#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix
from robot_localization.srv import FromLL
import xml.etree.ElementTree as ET
import math
import argparse


def parse_osm(file_path):
    """Extract nodes from OSM XML file"""
    tree = ET.parse(file_path)
    nodes = []
    for node in tree.getroot().findall('node'):
        nodes.append({
            'id': node.get('id'),
            'lat': float(node.get('lat')),
            'lon': float(node.get('lon'))
        })
    return nodes


def find_nearest(nodes, current_lat, current_lon):
    """Find nearest node using haversine distance"""
    def distance(lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return min(nodes, key=lambda n: distance(current_lat, current_lon, n['lat'], n['lon']))


class Navigator(Node):
    def __init__(self):
        super().__init__('osm_navigator')
        self.nav = BasicNavigator()
        self.gps = None
        self.create_subscription(NavSatFix, '/gps/fix', lambda msg: setattr(self, 'gps', msg), 10)
        self.ll_client = self.create_client(FromLL, '/fromLL')
        
        while not self.ll_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /fromLL service...')
    
    def wait_gps(self):
        """Wait for GPS fix"""
        while self.gps is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.gps.latitude, self.gps.longitude
    
    def gps_to_pose(self, lat, lon):
        """Convert GPS to map coordinates"""
        req = FromLL.Request()
        req.ll_point.latitude = lat
        req.ll_point.longitude = lon
        req.ll_point.altitude = 0.0
        
        future = self.ll_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position = future.result().map_point
        pose.pose.orientation.w = 1.0
        return pose
    
    def go_to_node(self, node):
        """Navigate to OSM node"""
        self.get_logger().info(f"Going to node {node['id']} at ({node['lat']}, {node['lon']})")
        
        goal = self.gps_to_pose(node['lat'], node['lon'])
        self.nav.goToPose(goal)
        
        while not self.nav.isTaskComplete():
            rclpy.spin_once(self, timeout_sec=0.1)
        
        result = self.nav.getResult()
        success = result == BasicNavigator.TaskResult.SUCCEEDED
        self.get_logger().info('Success!' if success else 'Failed')
        return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('osm_file', help='Path to OSM XML file')
    args = parser.parse_args()
    
    rclpy.init()
    navigator = Navigator()
    
    # Wait for Nav2
    navigator.get_logger().info('Waiting for Nav2...')
    navigator.nav.waitUntilNav2Active(localizer='robot_localization')
    
    # Get current GPS position
    navigator.get_logger().info('Waiting for GPS fix...')
    lat, lon = navigator.wait_gps()
    navigator.get_logger().info(f'Current position: ({lat}, {lon})')
    
    # Parse OSM and find nearest node
    navigator.get_logger().info(f'Parsing {args.osm_file}...')
    nodes = parse_osm(args.osm_file)
    navigator.get_logger().info(f'Found {len(nodes)} nodes')
    
    nearest = find_nearest(nodes, lat, lon)
    navigator.get_logger().info(f'Nearest node: {nearest}')
    
    # Navigate
    navigator.go_to_node(nearest)
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()