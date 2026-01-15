import osmnx as ox
import math
import argparse


def parse_osm(file_path):
    """Parse OSM file and extract all nodes"""
    print(f"\n{'='*60}")
    print(f"PARSING: {file_path}")
    print(f"{'='*60}")
    
    G = ox.graph_from_xml(file_path, bidirectional=True, simplify=False, retain_all=True)
    
    nodes = []
    for node_id, data in G.nodes(data=True):
        nodes.append({
            'id': str(node_id),
            'lat': data['y'],  # osmnx stores lat as 'y'
            'lon': data['x']   # osmnx stores lon as 'x'
        })
    
    print(f"Found {len(nodes)} nodes")
    if nodes:
        print(f"Sample: ID={nodes[0]['id']}, Lat={nodes[0]['lat']:.6f}, Lon={nodes[0]['lon']:.6f}")
    print(f"{'='*60}\n")
    
    return nodes


def find_nearest_node(nodes, lat, lon):
    """Find nearest OSM node to GPS position"""
    print(f"Current GPS: Lat={lat:.6f}, Lon={lon:.6f}")
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Earth radius in meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    # Find nearest
    nearest = min(nodes, key=lambda n: haversine(lat, lon, n['lat'], n['lon']))
    distance = haversine(lat, lon, nearest['lat'], nearest['lon'])
    
    print(f"✓ Nearest node: ID={nearest['id']}, Distance={distance:.1f}m")
    print(f"  Target GPS: Lat={nearest['lat']:.6f}, Lon={nearest['lon']:.6f}\n")
    
    return nearest


def gps_to_map_coordinates(osm_node, fromLL_client, ros_node):
    """Convert GPS lat/lon to map frame using robot_localization"""
    import rclpy
    from robot_localization.srv import FromLL
    from geometry_msgs.msg import PoseStamped
    
    # Call /fromLL service
    request = FromLL.Request()
    request.ll_point.latitude = osm_node['lat']
    request.ll_point.longitude = osm_node['lon']
    request.ll_point.altitude = 0.0
    
    future = fromLL_client.call_async(request)
    rclpy.spin_until_future_complete(ros_node, future)
    
    result = future.result()
    if result is None:
        return None
    
    # Create PoseStamped in map frame
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = ros_node.get_clock().now().to_msg()
    pose.pose.position = result.map_point
    pose.pose.orientation.w = 1.0
    
    return pose


def navigate_to_waypoint(pose, waypoint_action_client, node):
    """Send waypoint to Nav2 and wait for completion"""
    import rclpy
    from nav2_msgs.action import FollowWaypoints
    
    # Create goal
    goal = FollowWaypoints.Goal()
    goal.poses = [pose]
    
    print("Sending waypoint to Nav2...")
    send_future = waypoint_action_client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future)
    
    goal_handle = send_future.result()
    if not goal_handle.accepted:
        print("✗ Goal rejected by Nav2")
        return False
    
    print("Goal accepted, navigating...")
    
    # Wait for result
    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    
    result = result_future.result().result
    success = len(result.missed_waypoints) == 0
    
    print("✓ Navigation complete!" if success else "✗ Failed to reach waypoint")
    return success


def main():
    parser = argparse.ArgumentParser(description='Navigate to nearest OSM node')
    parser.add_argument('osm_file', help='Path to OSM XML file')
    parser.add_argument('--test-mode', action='store_true', help='Test parsing only (no ROS2)')
    parser.add_argument('--lat', type=float, help='Override GPS latitude for testing')
    parser.add_argument('--lon', type=float, help='Override GPS longitude for testing')
    parser.add_argument('--gps-topic', default='/j100_0004/gps/filtered', 
                        help='GPS topic (default: /j100_0004/gps/filtered)')
    parser.add_argument('--fromll-service', default='/fromLL',
                        help='robot_localization fromLL service (default: /fromLL)')
    parser.add_argument('--namespace', default='j100_0004',
                        help='Robot namespace for follow_waypoints action (default: j100_0004)')
    args = parser.parse_args()
    
    # Parse OSM file
    nodes = parse_osm(args.osm_file)
    if not nodes:
        print("ERROR: No nodes in OSM file")
        return
    
    # TEST MODE: Just parse and find nearest to test coordinates
    if args.test_mode:
        test_lat = args.lat or 32.8838
        test_lon = args.lon or -117.2350
        print(f"TEST MODE: Using coordinates {test_lat:.6f}, {test_lon:.6f}")
        nearest = find_nearest_node(nodes, test_lat, test_lon)
        print("\n✓ Test complete! To navigate, run without --test-mode")
        return
    
    # NAVIGATION MODE: Initialize ROS2
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.action import ActionClient
        from sensor_msgs.msg import NavSatFix
        from robot_localization.srv import FromLL
        from nav2_msgs.action import FollowWaypoints
    except ImportError:
        print("ERROR: ROS2 packages not found. Install ROS2 or use --test-mode")
        return
    
    rclpy.init()
    node = Node('osm_waypoint_navigator')
    
    # Get current GPS position
    print(f"Waiting for GPS on {args.gps_topic}...")
    current_gps = {'data': None}
    
    def gps_callback(msg):
        current_gps['data'] = msg
    
    gps_sub = node.create_subscription(NavSatFix, args.gps_topic, gps_callback, 10)
    
    # Wait for GPS
    while current_gps['data'] is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    
    # Use override coordinates if provided, otherwise use GPS
    if args.lat and args.lon:
        print(f"Using override coordinates: {args.lat:.6f}, {args.lon:.6f}")
        current_lat = args.lat
        current_lon = args.lon
    else:
        current_lat = current_gps['data'].latitude
        current_lon = current_gps['data'].longitude
        print(f"Got GPS: {current_lat:.6f}, {current_lon:.6f}")
    
    # Find nearest OSM node
    nearest = find_nearest_node(nodes, current_lat, current_lon)
    
    # Setup services and actions
    fromLL_client = node.create_client(FromLL, args.fromll_service)
    print(f"Waiting for {args.fromll_service} service...")
    while not fromLL_client.wait_for_service(timeout_sec=1.0):
        print("Still waiting...")
    
    waypoint_action = ActionClient(node, FollowWaypoints, f'/{args.namespace}/follow_waypoints')
    print(f"Waiting for /{args.namespace}/follow_waypoints action...")
    while not waypoint_action.wait_for_server(timeout_sec=1.0):
        print("Still waiting...")
    
    print("\n✓ All services ready!\n")
    
    # Convert GPS to map coordinates
    print("Converting GPS to map coordinates...")
    pose = gps_to_map_coordinates(nearest, fromLL_client, node)
    
    if pose is None:
        print("ERROR: Failed to convert GPS to map coordinates")
        rclpy.shutdown()
        return
    
    print(f"Map position: x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}")
    
    # Navigate!
    navigate_to_waypoint(pose, waypoint_action, node)
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()