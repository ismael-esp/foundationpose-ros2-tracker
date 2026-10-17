#!/usr/bin/env python3
import rclpy, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import threading, time, struct

class RGBDProbe(Node):
    def __init__(self):
        super().__init__("rgbd_probe")
        self.rgb=None; self.depth=None; self.ci=None
        qos = 10
        self.sub_rgb = self.create_subscription(Image, "/camera/color/image_raw", self.cb_rgb, qos)
        self.sub_depth = self.create_subscription(Image, "/camera/aligned_depth_to_color/image_raw", self.cb_depth, qos)
        self.sub_ci = self.create_subscription(CameraInfo, "/camera/color/camera_info", self.cb_ci, qos)
        self.timer = self.create_timer(0.5, self.tick)

    def cb_rgb(self, msg):  self.rgb = msg
    def cb_depth(self, msg): self.depth = msg
    def cb_ci(self, msg):    self.ci = msg

    def _img_stats(self, img):
        return f"{img.encoding} {img.width}x{img.height}, stamp={img.header.stamp.sec}.{img.header.stamp.nanosec:09d}"

    def tick(self):
        if self.rgb and self.depth:
            # RGB
            rgb = np.frombuffer(self.rgb.data, dtype=np.uint8).reshape(self.rgb.height, self.rgb.width, -1)
            m, s = float(rgb.mean()), float(rgb.std())
            # DEPTH
            d = np.frombuffer(self.depth.data, dtype=np.float32).reshape(self.depth.height, self.depth.width)
            finite = np.isfinite(d)
            nz = (d>0) & finite
            nz_ratio = float(nz.sum())/float(d.size)

            print("RGB:", self._img_stats(self.rgb), f"mean={m:.1f} std={s:.2f}")
            print("DEP:", self._img_stats(self.depth), f"nonzero_ratio={nz_ratio*100:.1f}%")
            if self.ci:
                K = list(self.ci.k)
                print(f"CI : {self.ci.width}x{self.ci.height} K=[{K[0]:.2f},0,{K[2]:.2f}; 0,{K[4]:.2f},{K[5]:.2f}; 0,0,1]")
            print("-"*60)

def main():
    rclpy.init()
    n = RGBDProbe()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == "__main__":
    main()
