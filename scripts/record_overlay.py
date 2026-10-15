#!/usr/bin/env python3
"""Record the tracker overlay to MP4 and the published pose to CSV.

    python scripts/record_overlay.py --pose_topic /green_cube_pose --out runs/green_cube --duration 60
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped


class OverlayRecorder(Node):
    def __init__(self, a):
        super().__init__('overlay_recorder')
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        self.mp4, self.fps, self.vw, self.n = a.out + '.mp4', a.fps, None, 0
        self.t0 = time.time()
        self.f = open(a.out + '_poses.csv', 'w', newline='')
        self.w = csv.writer(self.f)
        self.w.writerow(['t_sec', 'stamp_sec', 'frame_id', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        self.create_subscription(Image, a.overlay_topic, self.on_img, 50)
        self.create_subscription(PoseStamped, a.pose_topic, self.on_pose, 50)

    def on_img(self, m):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * 3]
        a = a.reshape(m.height, m.width, 3)
        if m.encoding == 'rgb8':
            a = a[..., ::-1]
        if self.vw is None:
            self.vw = cv2.VideoWriter(self.mp4, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (m.width, m.height))
        self.vw.write(np.ascontiguousarray(a))
        self.n += 1

    def on_pose(self, m):
        p, q = m.pose.position, m.pose.orientation
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.w.writerow([f'{time.time() - self.t0:.3f}', f'{stamp:.3f}', m.header.frame_id,
                         f'{p.x:.5f}', f'{p.y:.5f}', f'{p.z:.5f}',
                         f'{q.x:.5f}', f'{q.y:.5f}', f'{q.z:.5f}', f'{q.w:.5f}'])

    def close(self):
        if self.vw is not None:
            self.vw.release()
        self.f.close()
        print(f'[record_overlay] {self.n} frames -> {self.mp4}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pose_topic', required=True, help='e.g. /green_cube_pose')
    p.add_argument('--overlay_topic', default='/inference/overlay')
    p.add_argument('--out', default='runs/replay/overlay', help='output path prefix')
    p.add_argument('--duration', type=float, default=60.0, help='seconds')
    p.add_argument('--fps', type=float, default=15.0)
    a = p.parse_args()
    rclpy.init()
    node = OverlayRecorder(a)
    end = time.time() + a.duration
    try:
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    node.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
