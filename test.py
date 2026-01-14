#!/usr/bin/env python3

import osmnx as ox
import networkx as nx
import math
import argparse

# ROS2 imports - only loaded when needed (not in test mode)
# import rclpy
# from rclpy.node import Node
# from nav2_simple_commander.robot_navigator import BasicNavigator
# from geometry_msgs.msg import PoseStamped
# from sensor_msgs.msg import NavSatFix
# from robot_localization.srv import FromLL


def parse_osm_with_osmnx(file_path):
    """Extract nodes from OSM XML file using osmnx"""
    print(f"\n{'='*60}")
    print(f"PARSING OSM FILE WITH OSMNX: {file_path}")
    print(f"{'='*60}")
    
    # Load graph from OSM XML file
    G = ox.graph_from_xml(file_path, bidirectional=True, simplify=False, retain_all=True)
    
    print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    # Extract node data (osmnx stores lat/lon as 'y'/'x' attributes)
    nodes = []
    for node_id, data in G.nodes(data=True):
        nodes.append({
            'id': str(node_id),
            'lat': data['y'],  # osmnx stores latitude as 'y'
            'lon': data['x']   # osmnx stores longitude as 'x'
        })
    
    print(f"Total nodes extracted: {len(nodes)}")
    
    if nodes:
        print(f"\nFirst 5 nodes:")
        for n in nodes[:5]:
            print(f"  ID: {n['id']}, Lat: {n['lat']:.7f}, Lon: {n['lon']:.7f}")
    
    # Calculate bounds
    if nodes:
        lats = [n['lat'] for n in nodes]
        lons = [n['lon'] for n in nodes]
        print(f"\nMap bounds:")
        print(f"  Latitude:  {min(lats):.7f} to {max(lats):.7f}")
        print(f"  Longitude: {min(lons):.7f} to {max(lons):.7f}")
    
    print(f"{'='*60}\n")
    return nodes, G


def find_nearest(nodes, current_lat, current_lon):
    """Find nearest node using haversine distance"""
    print(f"\n{'='*60}")
    print(f"FINDING NEAREST NODE")
    print(f"{'='*60}")
    print(f"Current position: Lat={current_lat:.7f}, Lon={current_lon:.7f}")
    
    def distance(lat1, lon1, lat2, lon2):
        R = 6371000  # Earth radius in meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    # Calculate distances for all nodes
    node_distances = []
    for n in nodes:
        d = distance(current_lat, current_lon, n['lat'], n['lon'])
        node_distances.append((n, d))
    
    # Sort by distance
    node_distances.sort(key=lambda x: x[1])
    
    # Show top 5 nearest nodes
    print(f"\nTop 5 nearest nodes:")
    for i, (n, d) in enumerate(node_distances[:5]):
        print(f"  {i+1}. ID: {n['id']}, Distance: {d:.2f}m, Lat: {n['lat']:.7f}, Lon: {n['lon']:.7f}")
    
    nearest = node_distances[0][0]
    nearest_dist = node_distances[0][1]
    
    print(f"\n✓ SELECTED NEAREST NODE:")
    print(f"  ID: {nearest['id']}")
    print(f"  Distance: {nearest_dist:.2f} meters")
    print(f"  Lat: {nearest['lat']:.7f}, Lon: {nearest['lon']:.7f}")
    print(f"{'='*60}\n")
    
    return nearest


class Navigator:
    """Navigator class - only used in full ROS2 mode"""
    def __init__(self):
        # Import ROS2 modules here to avoid import errors in test mode
        import rclpy
        from rclpy.node import Node as ROS2Node
        from nav2_simple_commander.robot_navigator import BasicNavigator
        from sensor_msgs.msg import NavSatFix
        from robot_localization.srv import FromLL
        
        # Create ROS2 node
        class NavigatorNode(ROS2Node):
            def __init__(inner_self):
                super().__init__('osm_navigator')
        
        self.node = NavigatorNode()
        self.nav = BasicNavigator()
        self.gps = None
        
        self.node.create_subscription(
            NavSatFix, '/gps/fix', 
            lambda msg: setattr(self, 'gps', msg), 10
        )
        self.ll_client = self.node.create_client(FromLL, '/fromLL')
        
        while not self.ll_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Waiting for /fromLL service...')
    
    def wait_gps(self):
        """Wait for GPS fix"""
        import rclpy
        while self.gps is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return self.gps.latitude, self.gps.longitude
    
    def gps_to_pose(self, lat, lon):
        """Convert GPS to map coordinates"""
        import rclpy
        from robot_localization.srv import FromLL
        from geometry_msgs.msg import PoseStamped
        
        req = FromLL.Request()
        req.ll_point.latitude = lat
        req.ll_point.longitude = lon
        req.ll_point.altitude = 0.0
        
        future = self.ll_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position = future.result().map_point
        pose.pose.orientation.w = 1.0
        return pose
    
    def go_to_node(self, node):
        """Navigate to OSM node"""
        import rclpy
        from nav2_simple_commander.robot_navigator import BasicNavigator
        
        self.node.get_logger().info(
            f"Going to node {node['id']} at ({node['lat']}, {node['lon']})"
        )
        
        goal = self.gps_to_pose(node['lat'], node['lon'])
        self.nav.goToPose(goal)
        
        while not self.nav.isTaskComplete():
            rclpy.spin_once(self.node, timeout_sec=0.1)
        
        result = self.nav.getResult()
        success = result == BasicNavigator.TaskResult.SUCCEEDED
        self.node.get_logger().info('Success!' if success else 'Failed')
        return success


def main():
    parser = argparse.ArgumentParser(description='Navigate to nearest OSM node using Nav2')
    parser.add_argument('osm_file', help='Path to OSM XML file')
    parser.add_argument('--test-mode', action='store_true', 
                        help='Test mode: skip ROS2/Nav2 initialization')
    parser.add_argument('--lat', type=float, default=32.8838,
                        help='Test GPS latitude (default: 32.8838)')
    parser.add_argument('--lon', type=float, default=-117.2350,
                        help='Test GPS longitude (default: -117.2350)')
    args = parser.parse_args()
    
    # Parse OSM file with osmnx
    nodes, graph = parse_osm_with_osmnx(args.osm_file)
    
    if not nodes:
        print("ERROR: No nodes found in OSM file!")
        return
    
    # Use provided test coordinates
    test_lat = args.lat
    test_lon = args.lon
    
    print(f"\n{'='*60}")
    print(f"USING TEST GPS POSITION")
    print(f"{'='*60}")
    print(f"Test GPS: Lat={test_lat:.7f}, Lon={test_lon:.7f}")
    print(f"(This simulates robot starting position)")
    print(f"{'='*60}\n")
    
    # Find nearest node
    nearest = find_nearest(nodes, test_lat, test_lon)
    
    if args.test_mode:
        print("\n✓ TEST MODE: Parsing and node selection successful!")
        print("  To run actual navigation, remove --test-mode flag")
        print("\nGraph info:")
        print(f"  Nodes: {len(graph.nodes)}")
        print(f"  Edges: {len(graph.edges)}")
        return
    
    # Full ROS2 navigation mode (only runs if not in test mode)
    try:
        import rclpy
        from nav2_simple_commander.robot_navigator import BasicNavigator
    except ImportError:
        print("\nERROR: ROS2 packages not found!")
        print("  Install ROS2 and required packages, or use --test-mode")
        return
    
    rclpy.init()
    navigator = Navigator()
    
    # Wait for Nav2
    navigator.get_logger().info('Waiting for Nav2...')
    navigator.nav.waitUntilNav2Active(localizer='robot_localization')
    
    # Navigate
    navigator.go_to_node(nearest)
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()