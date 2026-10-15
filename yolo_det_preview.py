#!/usr/bin/env python3
import argparse
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

class YoloPreview(Node):
    def __init__(self, weights, topic, conf, only_banana):
        super().__init__('yolo_preview')
        self.bridge = CvBridge()
        self.model = self._load_model(weights)
        self.names = self.model.model.names if hasattr(self.model, 'model') else {}
        self.conf = float(conf)
        self.only_banana = only_banana
        self.last_t = time.time()
        self.fps = 0.0

        self.sub = self.create_subscription(Image, topic, self.cb, 10)
        self.get_logger().info(f"[YOLO] Loaded: {weights}")
        self.get_logger().info(f"Subscribed to: {topic}")
        cv2.namedWindow('YOLO Preview', cv2.WINDOW_NORMAL)

    def _load_model(self, w):
        try:
            return YOLO(w)             # 자동 다운로드 지원
        except Exception as e:
            self.get_logger().warn(f"Load {w} failed: {e}. Falling back to yolov8n.pt")
            return YOLO('yolov8n.pt')

    def cb(self, msg):
        # ROS Image -> OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # FPS
        now = time.time()
        dt = max(1e-6, now - self.last_t)
        self.last_t = now
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

        # Inference
        res = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        boxes = res.boxes
        draw = frame.copy()
        n_draw = 0

        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls = int(boxes.cls[i]) if boxes.cls is not None else -1
                label = self.names.get(cls, str(cls))
                if self.only_banana and label.lower() != 'banana':
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].int().tolist()
                conf = float(boxes.conf[i]) if boxes.conf is not None else 0.0
                cv2.rectangle(draw, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(draw, f"{label}:{conf:.2f}", (x1, max(0, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                n_draw += 1

        cv2.putText(draw, f"FPS:{self.fps:.1f}  det:{n_draw}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow('YOLO Preview', draw)
        k = cv2.waitKey(1) & 0xFF
        if k in [ord('q'), 27]:
            self.get_logger().info("Quit.")
            cv2.destroyAllWindows()
            rclpy.shutdown()
        elif k == ord('s'):
            path = f"yolo_frame_{int(time.time())}.png"
            cv2.imwrite(path, draw)
            self.get_logger().info(f"Saved {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='yolov8s.pt', help="Ultralytics weights or model name")
    ap.add_argument('--topic', default='/camera/camera/color/image_raw', help="ROS image topic")
    ap.add_argument('--conf', type=float, default=0.5, help="confidence threshold")
    ap.add_argument('--only_banana', action='store_true', help="show only 'banana' detections")
    args = ap.parse_args()

    rclpy.init()
    node = YoloPreview(args.weights, args.topic, args.conf, args.only_banana)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
