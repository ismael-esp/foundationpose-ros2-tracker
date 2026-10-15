#!/usr/bin/env python3
# yoloworld_preview.py
# Standalone preview of the EXACT YOLO-World detector used by foundation_pose.py
# (YoloWorldDetector, foundation_pose.py:485-536). Draws every box so you can
# see what YOLO-World detects for a given text prompt, and highlights the box
# the pipeline would actually pick (highest conf -> fed to SAM2 -> register()).
#
# It does NOT import foundation_pose.py (avoids the heavy estimater/nvdiffrast
# cold start) — it replicates only the YOLO-World call, byte-for-byte:
#   model = YOLOWorld("yolov8x-worldv2.pt")
#   model.set_classes([object])
#   model.predict(rgb[..., ::-1], conf=det_conf, verbose=False, device=DEV)[0]
#   pick = argmax(conf)
# Parity note: the pipeline feeds predict() **BGR** (it passes rgb[..., ::-1]
# where rgb is RGB). This script does the same so detections match exactly.
#
# ----------------------------------------------------------------------------
# RUN (same env as foundation_pose.py: fp_ros + ROS2 humble + CycloneDDS)
#
#   conda activate fp_ros
#   source /opt/ros/humble/setup.bash
#   export ROS_DOMAIN_ID=10
#   export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
#   export RMW_CYCLONEDDS_USE_ICEORYX=0
#   export RMW_CYCLONEDDS_LOG_LEVEL=error
#   export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
#   cd <repo root>
#
#   # live on the head camera (default — same topic foundation_pose.py uses):
#   python3 yoloworld_preview.py --object "red cube"
#
#   # front camera instead:
#   python3 yoloworld_preview.py --object "red cube" \
#       --color_topic /front_right/cam1/color/image_raw
#
#   # sweep the confidence threshold to see weak detections:
#   python3 yoloworld_preview.py --object "red cube" --det_conf 0.05
#
#   # try several prompts at once (pipeline only uses the single --object):
#   python3 yoloworld_preview.py --object "red cube,cube,box,red block"
#
#   # offline on a saved image (no ROS needed):
#   python3 yoloworld_preview.py --object "red cube" --image debug_frames/x.png
#
# Window keys: q/ESC quit · s save current annotated frame · +/- conf step
# ----------------------------------------------------------------------------

import os
import sys
import time
import argparse

import cv2
import numpy as np
import torch
from ultralytics import YOLOWorld


# --- device probe: identical to foundation_pose.py:_probe_det_device ---------
def probe_det_device() -> str:
    if not torch.cuda.is_available():
        return 'cpu'
    try:
        m = torch.nn.Conv2d(1, 1, 1).cuda()
        x = torch.zeros(1, 1, 4, 4).cuda()
        m(x)
        del m, x
        torch.cuda.empty_cache()
        return 'cuda:0'
    except Exception:
        torch.cuda.empty_cache()
        return 'cpu'


# --- manual ROS Image -> numpy (NO cv_bridge; verbatim from fp.py/foundation_pose.py) ---
def imgmsg_to_numpy(msg):
    if msg.encoding == 'rgb8':
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    elif msg.encoding == 'bgr8':
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def parse_args():
    p = argparse.ArgumentParser(
        description="Preview the YOLO-World detector used by foundation_pose.py")
    p.add_argument('--object', type=str, required=True,
                   help="Text prompt(s). Single value = exactly what the "
                        "pipeline uses (--object). Comma-separated = debug "
                        "multiple classes at once.")
    p.add_argument('--color_topic', type=str,
                   default='/head/cam2/color/image_raw',
                   help="ROS2 color topic (default = foundation_pose.py head "
                        "preset). Front: /front_right/cam1/color/image_raw")
    p.add_argument('--det_conf', type=float, default=0.30,
                   help="Confidence threshold (foundation_pose.py default 0.30).")
    p.add_argument('--weights', type=str, default='yolov8x-worldv2.pt',
                   help="YOLO-World weights (pipeline uses yolov8x-worldv2.pt).")
    p.add_argument('--device', type=str, default=None,
                   help="cuda:0 / cpu. Default = same auto-probe as the pipeline.")
    p.add_argument('--image', type=str, default=None,
                   help="Run once on an image file instead of the ROS topic.")
    p.add_argument('--save_dir', type=str, default='yoloworld_preview_out',
                   help="Where 's' / --image annotated frames are written.")
    return p.parse_args()


def annotate(bgr, dets, classes, conf_thr):
    """Draw every detection; thicken + label the one the pipeline would pick."""
    vis = bgr.copy()
    pick = int(np.argmax([d[5] for d in dets])) if dets else -1
    for i, (x1, y1, x2, y2, label, cf) in enumerate(dets):
        chosen = (i == pick)
        color = (0, 0, 255) if chosen else (0, 200, 0)   # BGR: red=picked
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3 if chosen else 2)
        tag = f"{label} {cf:.2f}" + ("  <- PIPELINE PICKS THIS" if chosen else "")
        yt = max(14, y1 - 6)
        cv2.putText(vis, tag, (x1, yt), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, tag, (x1, yt), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
    hud = (f"prompt={classes}  conf>={conf_thr:.2f}  "
           f"{len(dets)} det(s)  red=pipeline pick")
    cv2.putText(vis, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(vis, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def run_yoloworld(model, bgr, conf, device):
    """EXACT pipeline call (foundation_pose.py:514-523). `bgr` is what predict
    receives — the pipeline passes rgb[..., ::-1], i.e. BGR."""
    res = model.predict(bgr, conf=conf, verbose=False, device=device)[0]
    names = getattr(model, 'names', {}) or {}
    dets = []
    nb = 0 if res.boxes is None else len(res.boxes)
    for j in range(nb):
        bx = res.boxes.xyxy[j].int().tolist()
        cf = float(res.boxes.conf[j].item())
        cl = int(res.boxes.cls[j].item()) if res.boxes.cls is not None else -1
        dets.append((bx[0], bx[1], bx[2], bx[3], str(names.get(cl, '?')), cf))
    return dets


def main():
    a = parse_args()
    device = a.device or probe_det_device()
    classes = [s.strip() for s in a.object.split(',') if s.strip()]
    os.makedirs(a.save_dir, exist_ok=True)

    print(f"[YOLO-World] weights={a.weights} device={device} "
          f"conf>={a.det_conf} classes={classes}")
    model = YOLOWorld(a.weights)
    model.set_classes(classes)

    # ---- offline single image ----
    if a.image:
        bgr = cv2.imread(a.image)            # cv2 reads BGR == what predict gets
        if bgr is None:
            print(f"ERROR: cannot read {a.image}")
            sys.exit(1)
        dets = run_yoloworld(model, bgr, a.det_conf, device)
        for d in dets:
            print(f"  {d[4]:<16} conf={d[5]:.3f}  box=({d[0]},{d[1]},{d[2]},{d[3]})")
        if not dets:
            print("  (no detections — try lower --det_conf or a different prompt)")
        vis = annotate(bgr, dets, classes, a.det_conf)
        out = os.path.join(a.save_dir, 'image_annotated.png')
        cv2.imwrite(out, vis)
        print(f"saved -> {out}")
        try:
            cv2.imshow('yoloworld_preview', vis)
            cv2.waitKey(0)
        except Exception:
            pass
        return

    # ---- live ROS2 topic ----
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    class Prev(Node):
        def __init__(self):
            super().__init__('yoloworld_preview')
            self.latest = None
            self.create_subscription(Image, a.color_topic, self._cb, 10)
            self.get_logger().info(f"Subscribed {a.color_topic}; waiting for frames...")

        def _cb(self, msg):
            rgb = imgmsg_to_numpy(msg)
            if rgb is not None:
                # pipeline parity: predict() is fed rgb[..., ::-1] (BGR)
                self.latest = np.ascontiguousarray(rgb[..., ::-1])

    rclpy.init()
    node = Prev()
    conf = a.det_conf
    n, t0, fps = 0, time.time(), 0.0
    last_log = 0.0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.latest is None:
                continue
            bgr = node.latest
            dets = run_yoloworld(model, bgr, conf, device)
            n += 1
            if time.time() - t0 >= 1.0:
                fps, n, t0 = n / (time.time() - t0), 0, time.time()
            if time.time() - last_log >= 1.0:
                summ = ", ".join(f"{d[4]}:{d[5]:.2f}" for d in dets) or "none"
                node.get_logger().info(
                    f"conf>={conf:.2f} {len(dets)} det(s) [{fps:.1f} Hz] -> {summ}")
                last_log = time.time()
            vis = annotate(bgr, dets, classes, conf)
            cv2.putText(vis, f"{fps:.1f} Hz", (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(vis, f"{fps:.1f} Hz", (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('yoloworld_preview', vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('s'):
                p = os.path.join(a.save_dir, f"frame_{int(time.time())}.png")
                cv2.imwrite(p, vis)
                node.get_logger().info(f"saved -> {p}")
            elif k in (ord('+'), ord('=')):
                conf = min(0.95, conf + 0.05)
            elif k in (ord('-'), ord('_')):
                conf = max(0.01, conf - 0.05)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
