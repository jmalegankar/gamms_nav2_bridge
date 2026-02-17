#!/usr/bin/env python3
"""
GAMMS-Nav2 Bridge: Step-by-step autonomous exploration
Runs GAMMS game loop synchronized with Nav2 navigation.

Usage:
  # Test mode (no ROS, just GAMMS visualization)
  python3 gamms_nav2_bridge.py back_of_fah.osm --origin-lat 32.8838 --origin-lon -117.2350 --test-mode
  
  # Full mode (with Nav2)
  python3 gamms_nav2_bridge.py back_of_fah.osm --origin-lat 32.8838 --origin-lon -117.2350
"""

import math
import time
import argparse

import gamms
import gamms.osm
import strategy  # Your agent_strategy.py
from pyproj import Transformer

# =============================================================================
# CONFIGURATION
# =============================================================================

MAP_FRAME = 'map'
NAV_TIMEOUT = 120.0  # seconds
MAX_RETRIES = 2
NODE_TOLERANCE = 1.0  # meters


# =============================================================================
# HELPER: Coordinate Conversion (only used in full mode)
# =============================================================================

class CoordConverter:
    """Convert between GAMMS UTM coords and Nav2 map frame."""
    
    def __init__(self, node, origin_lat: float, origin_lon: float, fromll_service: str):
        self.node = node
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.fromll_service = fromll_service

    
        utm_zone = int((origin_lon + 180) / 6) + 1
        self.wgs84_to_utm = Transformer.from_crs(
            'EPSG:4326', f'EPSG:326{utm_zone}', always_xy=True
        )
        self.utm_to_wgs84 = Transformer.from_crs(
            f'EPSG:326{utm_zone}', 'EPSG:4326', always_xy=True
        )
        
        self.origin_utm_x, self.origin_utm_y = self.wgs84_to_utm.transform(
            origin_lon, origin_lat
        )
        
        # FromLL service client
        from robot_localization.srv import FromLL
        self.fromll_client = node.create_client(FromLL, self.fromll_service)
        
    def wait_for_service(self, timeout=10.0):
        print(f"[Coords] Waiting for {self.fromll_service}...")
        if not self.fromll_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"Service {self.fromll_service} not available")
        print(f"[Coords] Service ready")
        
    def gamms_to_gps(self, x: float, y: float) -> tuple:
        utm_x = x + self.origin_utm_x
        utm_y = y + self.origin_utm_y
        lon, lat = self.utm_to_wgs84.transform(utm_x, utm_y)
        return lat, lon
    
    def gps_to_gamms(self, lat: float, lon: float) -> tuple:
        utm_x, utm_y = self.wgs84_to_utm.transform(lon, lat)
        x = utm_x - self.origin_utm_x
        y = utm_y - self.origin_utm_y
        return x, y
    
    def gamms_to_nav2(self, x: float, y: float):
        import rclpy
        from robot_localization.srv import FromLL
        from geometry_msgs.msg import PoseStamped
        
        lat, lon = self.gamms_to_gps(x, y)
        
        req = FromLL.Request()
        req.ll_point.latitude = lat
        req.ll_point.longitude = lon
        req.ll_point.altitude = 0.0
        
        future = self.fromll_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
        
        if future.result() is None:
            raise RuntimeError("fromLL service call failed")
        
        result = future.result()
        
        pose = PoseStamped()
        pose.header.frame_id = MAP_FRAME
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position = result.map_point
        pose.pose.orientation.w = 1.0
        
        return pose


# =============================================================================
# MAIN BRIDGE CLASS
# =============================================================================

class GAMMSNav2Bridge:
    """Bridge between GAMMS game loop and Nav2."""
    
    def __init__(self, osm_file: str, origin_lat: float, origin_lon: float, 
                 test_mode: bool = False, start_node: int = None, namespace: str = ""):
        self.osm_file = osm_file
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.test_mode = test_mode
        self.start_node_override = start_node
        
        # ROS node (only in full mode)
        self.ros_node = None
        self.nav_client = None
        self.coords = None
        self.current_gps = None
        
        # Navigation state
        self.nav_complete = False
        self.nav_success = False
        self.goal_handle = None
        
        # Step counter
        self.step = 0
        
        # Agent naming (strategy expects 'agent_0' format)
        self.agent_name = 'agent_0'

        self.namespace = namespace
        pfx = self._ns_prefix(namespace)
        
        self.gps_topic = f"{pfx}/gps/filtered"
        self.fromll_service = f"{pfx}/fromLL"
        self.nav2_action = f"{pfx}/follow_waypoints"
        
        
        
        print("\n" + "="*60)
        print("GAMMS-NAV2 BRIDGE" + (" [TEST MODE]" if test_mode else ""))
        print("="*60)
    
    def _ns_prefix(self, ns: str) -> str:
            ns = (ns or "").strip().strip("/")
            return f"/{ns}" if ns else ""
        
    def initialize(self):
        """Full initialization sequence."""
        if not self.test_mode:
            self._init_ros()
        self._init_gamms()
        self._init_strategy()
        self._find_start_node()
        self._create_agent()
        
    def _init_ros(self):
        """Initialize ROS2 components."""
        import rclpy
        from rclpy.node import Node
        from rclpy.action import ActionClient
        from sensor_msgs.msg import NavSatFix
        from nav2_msgs.action import FollowWaypoints
        
        print("\n[ROS2] Initializing...")
        
        rclpy.init()
        self.ros_node = Node('gamms_nav2_bridge')
        
        # Coordinate converter
        self.coords = CoordConverter(self.ros_node, self.origin_lat, self.origin_lon, self.fromll_service)
        self.coords.wait_for_service()
        
        # Nav2 action client
        self.nav_client = ActionClient(self.ros_node, FollowWaypoints, self.nav2_action)
        print(f"[ROS2] Waiting for {self.nav2_action}...")
        if not self.nav_client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError(f"Nav2 action {self.nav2_action} not available")
        print(f"[ROS2] Nav2 ready")
        
        # GPS subscriber
        self.gps_sub = self.ros_node.create_subscription(
            NavSatFix, self.gps_topic, self._gps_callback, 10
        )
        
        # Wait for GPS
        print(f"[ROS2] Waiting for GPS on {self.gps_topic}...")
        while self.current_gps is None:
            rclpy.spin_once(self.ros_node, timeout_sec=0.1)
        print(f"[ROS2] GPS: {self.current_gps[0]:.6f}, {self.current_gps[1]:.6f}")
        
    def _gps_callback(self, msg):
        self.current_gps = (msg.latitude, msg.longitude)
        
    def _init_gamms(self):
        """Initialize GAMMS context and load graph."""
        print("\n[GAMMS] Initializing...")
        
        self.ctx = gamms.create_context(
            vis_engine=gamms.visual.Engine.PYGAME,
            logger_config={'level': 'ERROR'}
        )
        
        print(f"[GAMMS] Loading {self.osm_file}...")
        self.G = gamms.osm.graph_from_xml(
            self.osm_file,
            resolution=5.0,
            bidirectional=True,
            retain_all=False,
            tolerance=8.25
        )
        print(f"[GAMMS] Graph: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges")
        
        self.ctx.graph.attach_networkx_graph(self.G)
        
        self.ctx.visual.set_graph_visual(
            width=1920, height=1080,
            node_size=2.0, edge_width=1.0,
            node_color=(80, 80, 80),
            edge_color=(120, 120, 120),
        )
        
    def _init_strategy(self):
        """Extract graph data for strategy module."""
        print("\n[Strategy] Building knowledge base...")
        
        strategy.global_nodes.clear()
        strategy.global_edges.clear()
        strategy.visited_nodes.clear()
        strategy.recent_nodes.clear()
        
        utm_zone = int((self.origin_lon + 180) / 6) + 1
        wgs84_to_utm = Transformer.from_crs(
            'EPSG:4326', f'EPSG:326{utm_zone}', always_xy=True
        )
        origin_utm_x, origin_utm_y = wgs84_to_utm.transform(
            self.origin_lon, self.origin_lat
        )
        
        for node_id, data in self.G.nodes(data=True):
            rel_x = data['x'] - origin_utm_x
            rel_y = data['y'] - origin_utm_y
            strategy.global_nodes[node_id] = (rel_x, rel_y)
            
        for u, v in self.G.edges():
            strategy.global_edges.add((u, v))
            strategy.global_edges.add((v, u))
            
        print(f"[Strategy] {len(strategy.global_nodes)} nodes, {len(strategy.global_edges)} edges")
        
    def _find_start_node(self):
        """Find start node from GPS or override."""
        print("\n[Init] Finding start node...")
        
        # Use override if provided
        if self.start_node_override is not None:
            if self.start_node_override in strategy.global_nodes:
                self.start_node = self.start_node_override
                x, y = strategy.global_nodes[self.start_node]
                print(f"[Init] Using override start node: {self.start_node} at ({x:.1f}, {y:.1f})")
                return
            else:
                # Show some valid node IDs to help user
                sample_nodes = list(strategy.global_nodes.keys())[:10]
                print(f"[Init] Warning: Node {self.start_node_override} not found!")
                print(f"[Init] Available nodes (sample): {sample_nodes}")
                print(f"[Init] Falling back to GPS/random...")
        
        # Test mode: pick random node
        if self.test_mode:
            import random
            self.start_node = random.choice(list(strategy.global_nodes.keys()))
            x, y = strategy.global_nodes[self.start_node]
            print(f"[Init] Test mode - random start: node {self.start_node} at ({x:.1f}, {y:.1f})")
            return
        
        # Full mode: use GPS
        utm_zone = int((self.origin_lon + 180) / 6) + 1
        wgs84_to_utm = Transformer.from_crs(
            'EPSG:4326', f'EPSG:326{utm_zone}', always_xy=True
        )
        origin_utm_x, origin_utm_y = wgs84_to_utm.transform(
            self.origin_lon, self.origin_lat
        )
        
        gps_utm_x, gps_utm_y = wgs84_to_utm.transform(
            self.current_gps[1], self.current_gps[0]
        )
        robot_x = gps_utm_x - origin_utm_x
        robot_y = gps_utm_y - origin_utm_y
        
        print(f"[Init] Robot position: ({robot_x:.1f}, {robot_y:.1f})")
        
        min_dist = float('inf')
        nearest_node = None
        
        for node_id, (nx, ny) in strategy.global_nodes.items():
            dist = math.sqrt((nx - robot_x)**2 + (ny - robot_y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node_id
                
        if nearest_node is None:
            raise RuntimeError("No nodes in graph!")
            
        if min_dist > NODE_TOLERANCE:
            raise RuntimeError(
                f"Robot too far from any node: {min_dist:.1f}m > {NODE_TOLERANCE}m"
            )
            
        self.start_node = nearest_node
        nx, ny = strategy.global_nodes[nearest_node]
        print(f"[Init] Start node: {nearest_node} at ({nx:.1f}, {ny:.1f}), {min_dist:.1f}m away")
        
    def _create_agent(self):
        """Create GAMMS agent at start node."""
        print("\n[Agent] Creating agent...")
        
        # Strategy expects 'neighbor_sensor_0' format for get_moves()
        self.ctx.sensor.create_sensor(
            'neighbor_sensor_0',
            gamms.sensor.SensorType.NEIGHBOR
        )
        
        # Strategy expects 'agent_0' format for get_moves()
        self.ctx.agent.create_agent(
            self.agent_name,
            start_node_id=self.start_node,
            sensors=['neighbor_sensor_0'],
            meta={'team': 0}
        )
        
        self.agent = self.ctx.agent.get_agent(self.agent_name)
        
        self.ctx.visual.set_agent_visual(
            self.agent_name,
            color=(0, 255, 0),
            size=3
        )
        
        strategy.visited_nodes.add(self.start_node)
        strategy.recent_nodes.append(self.start_node)
        
        print(f"[Agent] Created at node {self.start_node}")
        
    # =========================================================================
    # NAVIGATION
    # =========================================================================
    
    def send_nav_goal(self, node_id: int) -> bool:
        """Send navigation goal and wait for completion."""
        import rclpy
        from nav2_msgs.action import FollowWaypoints
        
        if node_id not in strategy.global_nodes:
            print(f"[Nav2] ERROR: Unknown node {node_id}")
            return False
            
        x, y = strategy.global_nodes[node_id]
        lat, lon = self.coords.gamms_to_gps(x, y)
        
        print(f"[Nav2] Goal: node {node_id}")
        print(f"       GAMMS: ({x:.1f}, {y:.1f})")
        print(f"       GPS:   ({lat:.6f}, {lon:.6f})")
        
        try:
            pose = self.coords.gamms_to_nav2(x, y)
        except Exception as e:
            print(f"[Nav2] ERROR: Coord conversion failed: {e}")
            return False
            
        print(f"       Map:   ({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f})")
        
        # Create goal with single waypoint
        goal = FollowWaypoints.Goal()
        goal.poses = [pose]
        
        self.nav_complete = False
        self.nav_success = False
        
        send_future = self.nav_client.send_goal_async(
            goal, 
            feedback_callback=self._nav_feedback
        )
        
        rclpy.spin_until_future_complete(self.ros_node, send_future, timeout_sec=10.0)
        self.goal_handle = send_future.result()
        
        if not self.goal_handle or not self.goal_handle.accepted:
            print("[Nav2] ERROR: Goal rejected")
            return False
            
        print("[Nav2] Goal accepted, navigating...")
        
        result_future = self.goal_handle.get_result_async()
        
        start_time = time.time()
        while not self.nav_complete:
            rclpy.spin_once(self.ros_node, timeout_sec=0.05)
            self.ctx.visual.simulate()
            
            if result_future.done():
                result = result_future.result().result
                
                # Check missed_waypoints - empty means success
                if len(result.missed_waypoints) == 0:
                    self.nav_success = True
                    print("[Nav2] ✓ Goal reached!")
                else:
                    self.nav_success = False
                    print(f"[Nav2] ✗ Missed waypoints: {result.missed_waypoints}")
                    
                self.nav_complete = True
                
            if time.time() - start_time > NAV_TIMEOUT:
                print(f"[Nav2] ERROR: Timeout ({NAV_TIMEOUT}s)")
                self._cancel_goal()
                return False
                
        return self.nav_success
    
    def _nav_feedback(self, feedback_msg):
        fb = feedback_msg.feedback
        print(f"[Nav2] Processing waypoint {fb.current_waypoint}", end='\r')
            
    def _cancel_goal(self):
        if self.goal_handle:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None
            
    # =========================================================================
    # TEST MODE SIMULATION
    # =========================================================================
    
    def simulate_nav(self, node_id: int) -> bool:
        """Simulate navigation in test mode (always succeeds after delay)."""
        if node_id not in strategy.global_nodes:
            print(f"[TestNav] ERROR: Unknown node {node_id}")
            return False
            
        x, y = strategy.global_nodes[node_id]
        current_node = self.agent.current_node_id
        cx, cy = strategy.global_nodes[current_node]
        
        dist = math.sqrt((x - cx)**2 + (y - cy)**2)
        
        print(f"[TestNav] Simulating: node {current_node} → {node_id} ({dist:.1f}m)")
        
        # Simulate travel time (0.5m/s walking speed)
        travel_time = min(dist / 0.5, 3.0)  # Cap at 3 seconds
        steps = int(travel_time * 10)
        
        for i in range(steps):
            time.sleep(0.1)
            self.ctx.visual.simulate()
            print(f"[TestNav] Progress: {100*(i+1)/steps:.0f}%", end='\r')
            
        print(f"[TestNav] ✓ Arrived at node {node_id}           ")
        self.agent.prev_node_id = self.agent.current_node_id
        return True
            
    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    
    def run(self):
        """Main game loop."""
        print("\n" + "="*60)
        print("STARTING EXPLORATION" + (" [TEST MODE]" if self.test_mode else ""))
        print("="*60 + "\n")
        
        self.ctx.visual.simulate()
        
        while not self.ctx.is_terminated():
            self.step += 1
            
            # === GAMMS PHASE ===
            state = self.agent.get_state()
            current_node = state['curr_pos']
            
            print(f"\n[Step {self.step}] At node {current_node}")
            
            strategy.agent_strategy(state, self.agent_name, self.step)
            next_node = state.get('action')
            
            if next_node is None or next_node == current_node:
                print(f"[Step {self.step}] No valid move - checking termination...")
                
                # Check if exploration complete
                frontiers = strategy.get_all_frontiers()
                if not frontiers:
                    print(f"[Step {self.step}] No frontiers remaining - exploration complete!")
                    break
                    
                self.ctx.visual.simulate()
                time.sleep(0.5)
                continue
                
            print(f"[Step {self.step}] Strategy chose: node {next_node}")
            
            # === NAV PHASE ===
            success = False
            for attempt in range(MAX_RETRIES + 1):
                if attempt > 0:
                    print(f"[Step {self.step}] Retry {attempt}/{MAX_RETRIES}")
                
                if self.test_mode:
                    success = self.simulate_nav(next_node)
                else:
                    success = self.send_nav_goal(next_node)
                    
                if success:
                    break
                time.sleep(1.0)
                
            # === SYNC PHASE ===
            if success:
                self.agent.set_state()
                strategy.visited_nodes.add(next_node)
                strategy.recent_nodes.append(next_node)
                print(f"[Step {self.step}] ✓ Moved to node {next_node}")
            else:
                print(f"[Step {self.step}] ✗ Failed to reach node {next_node}")
                
            self.ctx.visual.simulate()
            
            # Stats
            visited = len(strategy.visited_nodes)
            total = len(strategy.global_nodes)
            pct = 100.0 * visited / total if total > 0 else 0
            frontiers = len(strategy.get_all_frontiers())
            print(f"[Stats] Explored: {visited}/{total} ({pct:.1f}%) | Frontiers: {frontiers}")
            
        print("\n" + "="*60)
        print("EXPLORATION COMPLETE")
        print(f"Total steps: {self.step}")
        print(f"Nodes visited: {len(strategy.visited_nodes)}/{len(strategy.global_nodes)}")
        print("="*60)
        
    def shutdown(self):
        """Clean shutdown."""
        if not self.test_mode and self.ros_node:
            import rclpy
            rclpy.shutdown()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='GAMMS-Nav2 Bridge')
    parser.add_argument('osm_file', help='Path to OSM XML file')
    parser.add_argument('--origin-lat', type=float, required=True,
                        help='Map origin latitude')
    parser.add_argument('--origin-lon', type=float, required=True,
                        help='Map origin longitude')
    parser.add_argument('--namespace', default='',
                        help='Robot namespace (default: empty)')
    parser.add_argument('--test-mode', action='store_true',
                        help='Test mode: run GAMMS only, simulate navigation')
    parser.add_argument('--start-node', type=int, default=None,
                        help='Override start node ID')
    parser.add_argument('--list-nodes', action='store_true',
                        help='List all node IDs and exit')
    args = parser.parse_args()
    
    # Handle --list-nodes: just load graph and show nodes
    if args.list_nodes:
        print(f"\n[List] Loading {args.osm_file}...")
        G = gamms.osm.graph_from_xml(
            args.osm_file,
            resolution=5.0,
            bidirectional=True,
            retain_all=False,
            tolerance=8.25
        )
        print(f"[List] Found {len(G.nodes)} nodes:\n")
        for node_id in sorted(G.nodes()):
            data = G.nodes[node_id]
            print(f"  {node_id}: ({data['x']:.1f}, {data['y']:.1f})")
        return
    
    bridge = None
    try:
        bridge = GAMMSNav2Bridge(
            osm_file=args.osm_file,
            origin_lat=args.origin_lat,
            origin_lon=args.origin_lon,
            test_mode=args.test_mode,
            start_node=args.start_node,
            namespace=args.namespace
        )
        bridge.initialize()
        bridge.run()
    except KeyboardInterrupt:
        print("\n[Bridge] Interrupted by user")
    except Exception as e:
        print(f"\n[Bridge] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if bridge:
            bridge.shutdown()


if __name__ == '__main__':
    main()

