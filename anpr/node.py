#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import easyocr
import numpy as np


class LicensePlateDetectorNode(Node):
    def __init__(self):
        super().__init__('license_plate_detector')
        
        # Declare parameters
        self.declare_parameter('model_path', 'license_plate_detector.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('use_gpu', True)
        
        # Get parameters
        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        use_gpu = self.get_parameter('use_gpu').value
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Load YOLO model
        self.get_logger().info(f'Loading YOLO model from {model_path}...')
        self.model = YOLO(model_path)
        
        # Load OCR reader
        self.get_logger().info('Loading EasyOCR reader...')
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        
        # Create subscriber for camera images
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',  # Change to your camera topic
            self.image_callback,
            10
        )
        
        # Create publishers
        self.annotated_image_pub = self.create_publisher(
            Image,
            '/license_plate/annotated_image',
            10
        )
        
        self.plate_text_pub = self.create_publisher(
            String,
            '/license_plate/text',
            10
        )
        
        self.detections_pub = self.create_publisher(
            Detection2DArray,
            '/license_plate/detections',
            10
        )
        
        self.get_logger().info('License Plate Detector Node initialized!')
    
    def image_callback(self, msg):
        """Process incoming images and detect license plates"""
        try:
            # Convert ROS Image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Run YOLO detection
            results = self.model(image, conf=self.conf_threshold, verbose=False)
            
            # Prepare detection array message
            detection_array = Detection2DArray()
            detection_array.header = msg.header
            
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    self.get_logger().info(
                        f'License Plate detected (confidence: {confidence:.2f})'
                    )
                    
                    # Crop the license plate
                    plate_img = image[y1:y2, x1:x2]
                    
                    if plate_img.size == 0:
                        continue
                    
                    # Preprocess for OCR
                    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(
                        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    
                    # Perform OCR
                    ocr_results = self.reader.readtext(thresh)
                    
                    # Extract text
                    if ocr_results:
                        text = ' '.join([r[1] for r in ocr_results])
                        text = text.replace(' ', '').upper()
                        self.get_logger().info(f'Plate Text: {text}')
                        
                        # Publish plate text
                        text_msg = String()
                        text_msg.data = text
                        self.plate_text_pub.publish(text_msg)
                    else:
                        text = "UNREADABLE"
                        self.get_logger().warn('Could not read text from plate')
                    
                    # Draw bounding box and label
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f"{text} ({confidence:.2f})"
                    cv2.putText(
                        image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                    )
                    
                    # Create Detection2D message
                    detection = Detection2D()
                    detection.bbox.center.position.x = float((x1 + x2) / 2)
                    detection.bbox.center.position.y = float((y1 + y2) / 2)
                    detection.bbox.size_x = float(x2 - x1)
                    detection.bbox.size_y = float(y2 - y1)
                    
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = text
                    hypothesis.hypothesis.score = confidence
                    detection.results.append(hypothesis)
                    
                    detection_array.detections.append(detection)
            
            # Publish annotated image
            annotated_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            annotated_msg.header = msg.header
            self.annotated_image_pub.publish(annotated_msg)
            
            # Publish detections
            if detection_array.detections:
                self.detections_pub.publish(detection_array)
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = LicensePlateDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()