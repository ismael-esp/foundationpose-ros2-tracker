#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import time
from pathlib import Path

import numpy as np
import zmq

import rosbag2_py
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image, CameraInfo
from builtin_interfaces.msg import Time

# --------------------------
# Optional resize (OpenCV)
# --------------------------
try:
    import cv2
except Exception:
    cv2 = None


def _parse_resize(s: str):
    if not s or s.lower() == "keep":
        return None
    s = s.lower().replace(" ", "")
    if "x" not in s:
        raise ValueError("--resize 형식은 'WxH' 입니다. 예) 320x180")
    w, h = s.split("x")
    return (int(w), int(h))


def _resize_image(arr: np.ndarray, wh, is_depth=False) -> np.ndarray:
    """RGB/Depth 공용 리사이즈. depth는 NEAREST, RGB는 LINEAR."""
    if wh is None:
        return arr
    if cv2 is None:
        raise RuntimeError("리사이즈가 필요하지만 OpenCV가 없습니다. `pip install opencv-python` 하거나 --resize keep 사용하세요.")
    w, h = wh
    interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
    return cv2.resize(arr, (w, h), interpolation=interp)


def _time_msg_from_ns(t_ns: int) -> Time:
    m = Time()
    m.sec = int(t_ns // 1_000_000_000)
    m.nanosec = int(t_ns % 1_000_000_000)
    return m


def _next_free_dir(base: str) -> str:
    """base가 비어있으면 base, 있으면 base_001, _002 ..."""
    p = Path(base)
    if not p.exists():
        return str(p)
    i = 1
    while True:
        cand = p.parent / f"{p.name}_{i:03d}"
        if not cand.exists():
            return str(cand)
        i += 1


def _to_uint16_mm_from_float_m(depth_m: np.ndarray) -> np.ndarray:
    """32FC1(m) -> 16UC1(mm)"""
    d = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    d = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)
    return d


class LazyBagWriter:
    def __init__(self, base_dir: str, ns: str, frame_id: str,
                 depth_out_encoding: str = "16UC1",
                 resize_wh: tuple[int, int] | None = None):
        self.base_dir = base_dir
        self.ns = ns.rstrip("/")
        self.frame_id = frame_id
        self.dir = None
        self.writer = None
        self.opened = False

        self.depth_out_encoding = depth_out_encoding.upper()
        assert self.depth_out_encoding in ("16UC1", "32FC1")
        self.resize_wh = resize_wh

    # --------------------------
    # rosbag2 open
    # --------------------------
    def _try_open_at(self, uri: str):
        self.writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(uri=uri, storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        )
        self.writer.open(storage_options, converter_options)

        for name, typ in [
            (f"{self.ns}/color/image_raw", "sensor_msgs/msg/Image"),
            (f"{self.ns}/aligned_depth_to_color/image_raw", "sensor_msgs/msg/Image"),
            (f"{self.ns}/color/camera_info", "sensor_msgs/msg/CameraInfo"),
        ]:
            md = rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format="cdr")
            self.writer.create_topic(md)

    def _open(self):
        # 경쟁 상황에서도 안전하게 여러 번 재시도
        retries = 12
        delay = 0.05
        for _ in range(retries):
            cand = _next_free_dir(self.base_dir)
            try:
                self._try_open_at(cand)
                self.dir = cand
                self.opened = True
                print(f"[bag] writing to: {self.dir}", flush=True)
                return
            except RuntimeError as e:
                msg = str(e)
                if "already exists" in msg or "exists" in msg:
                    time.sleep(delay)
                    continue
                raise
        raise RuntimeError("Failed to open rosbag2 after multiple attempts.")

    # --------------------------
    # write one synchronized triplet
    # --------------------------
    def write(self, rgb_u8: np.ndarray, depth_f32_m: np.ndarray, K_in: np.ndarray, t_ns: int):
        if not self.opened:
            self._open()

        # 원본 크기
        h, w = rgb_u8.shape[:2]
        dh, dw = depth_f32_m.shape[:2]

        # ---------------- resize (선택) ----------------
        # RGB/Depth 모두 동일 목표 해상도로 맞춤 + K 스케일
        if self.resize_wh is not None:
            rgb_out = _resize_image(rgb_u8, self.resize_wh, is_depth=False)
            depth_m = _resize_image(depth_f32_m, self.resize_wh, is_depth=True)
            W_out, H_out = self.resize_wh[0], self.resize_wh[1]

            sx = W_out / float(w)
            sy = H_out / float(h)

            # Intrinsics 스케일
            K = np.array(K_in, dtype=np.float32).copy()
            K[0, 0] *= sx
            K[1, 1] *= sy
            K[0, 2] *= sx
            K[1, 2] *= sy
        else:
            rgb_out = rgb_u8
            depth_m = depth_f32_m
            H_out, W_out = h, w
            K = np.array(K_in, dtype=np.float32)

        # ---------------- CameraInfo ----------------
        info = CameraInfo()
        info.header.stamp = _time_msg_from_ns(t_ns)
        info.header.frame_id = self.frame_id
        info.height = int(H_out)
        info.width = int(W_out)
        info.k = [
            float(K[0, 0]), 0.0,            float(K[0, 2]),
            0.0,            float(K[1, 1]), float(K[1, 2]),
            0.0,            0.0,            1.0,
        ]
        info.p = [
            float(K[0, 0]), 0.0,            float(K[0, 2]), 0.0,
            0.0,            float(K[1, 1]), float(K[1, 2]), 0.0,
            0.0,            0.0,            1.0,            0.0,
        ]
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5

        # ---------------- RGB Image ----------------
        rgb = Image()
        rgb.header = info.header
        rgb.height = int(H_out)
        rgb.width = int(W_out)
        rgb.encoding = "rgb8"
        rgb.is_bigendian = 0
        rgb.step = int(W_out * 3)
        rgb.data = np.ascontiguousarray(rgb_out).tobytes()

        # ---------------- Depth Image ----------------
        depth = Image()
        depth.header = info.header
        depth.height = int(depth_m.shape[0])
        depth.width = int(depth_m.shape[1])
        depth.is_bigendian = 0

        if self.depth_out_encoding == "16UC1":
            d16 = _to_uint16_mm_from_float_m(depth_m)
            depth.encoding = "16UC1"
            depth.step = int(depth.width * 2)
            depth.data = np.ascontiguousarray(d16).tobytes()
        else:
            depth.encoding = "32FC1"
            depth.step = int(depth.width * 4)
            depth.data = np.ascontiguousarray(depth_m.astype(np.float32)).tobytes()

        ts = int(t_ns)
        self.writer.write(f"{self.ns}/color/camera_info", serialize_message(info), ts)
        self.writer.write(f"{self.ns}/color/image_raw", serialize_message(rgb), ts)
        self.writer.write(f"{self.ns}/aligned_depth_to_color/image_raw", serialize_message(depth), ts)


def main():
    parser = argparse.ArgumentParser(
        description="Receive RGB-D via ZMQ and write rosbag2(.db3) with auto-index dir and robust EOS/idle handling."
    )
    parser.add_argument("--bind", type=str, default="tcp://*:5555", help="ZMQ PULL bind address")
    parser.add_argument("--bag_dir", type=str, default="rgbd_bag", help="Base output dir (auto-index)")
    parser.add_argument("--ns", type=str, default="/camera/camera")
    parser.add_argument("--frame_id", type=str, default="cam_front_color_optical_frame")
    parser.add_argument("--drop", type=int, default=0, help="Drop every N frames (>0 means keep only 1 of N+1)")
    parser.add_argument("--idle_timeout", type=float, default=20.0, help="초 단위. 0이면 무한대기, >0이면 해당 시간 무입력이면 종료")
    parser.add_argument("--depth_out_encoding", type=str, default="16UC1", choices=["16UC1", "32FC1"],
                        help="뎁스 출력 인코딩 (기본: 16UC1 = 데모 스타일)")
    parser.add_argument("--resize", type=str, default="keep", help="출력 해상도. 예: 320x180, 기본 keep")
    args = parser.parse_args()

    resize_wh = _parse_resize(args.resize)

    # ---- ZMQ (PULL) ----
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PULL)
    sock.bind(args.bind)
    print(f"[ZMQ] Recorder bound at {args.bind}", flush=True)

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    idle_ms = int(args.idle_timeout * 1000.0) if args.idle_timeout and args.idle_timeout > 0 else None

    bag = LazyBagWriter(
        base_dir=args.bag_dir,
        ns=args.ns,
        frame_id=args.frame_id,
        depth_out_encoding=args.depth_out_encoding,
        resize_wh=resize_wh,
    )

    got_any = False
    n = 0

    while True:
        # idle timeout 처리
        if idle_ms is None:
            socks = dict(poller.poll())  # 무한 대기
        else:
            socks = dict(poller.poll(idle_ms))
            if not socks:
                if got_any:
                    print("⌛ idle timeout reached. Closing.", flush=True)
                    break
                else:
                    # 아직 아무 것도 못받았으면 계속 대기
                    continue

        if socks.get(sock) != zmq.POLLIN:
            continue

        parts = sock.recv_multipart()
        try:
            header = json.loads(parts[0].decode("utf-8"))
        except Exception as e:
            print(f"[ERR] header parse failed: {e}", flush=True)
            continue

        # EOS 처리
        if header.get("eos", False):
            if got_any:
                print("✅ EOS received. Closing.", flush=True)
                break
            else:
                print("⚠️ EOS received before any frame. Waiting for frames...", flush=True)
                continue

        # 페이로드 파싱
        try:
            rgb_bytes = parts[1]
            depth_bytes = parts[2]
        except IndexError:
            print("[ERR] multipart length < 3", flush=True)
            continue

        try:
            w, h = int(header["w"]), int(header["h"])
            dw, dh = int(header["dw"]), int(header["dh"])
            K = np.array(header["K"], dtype=np.float32).reshape(3, 3)
            t_ns = int(header["stamp_ns"])
        except Exception as e:
            print(f"[ERR] header fields missing/invalid: {e}", flush=True)
            continue

        try:
            rgb_u8 = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(h, w, 3)
            depth_f32 = np.frombuffer(depth_bytes, dtype=np.float32).reshape(dh, dw)
        except Exception as e:
            print(f"[ERR] reshape failed: {e} (rgb {len(rgb_bytes)} bytes, depth {len(depth_bytes)} bytes, "
                  f"rgb shape target {h}x{w}x3, depth shape target {dh}x{dw})", flush=True)
            continue

        # frame dropping
        if args.drop > 0 and (n % (args.drop + 1)) != 0:
            n += 1
            continue

        # write
        bag.write(rgb_u8, depth_f32, K, t_ns)
        got_any = True

        if n % 30 == 0:
            nz = int((depth_f32 > 0).sum())
            mean = float(rgb_u8.mean())
            std = float(rgb_u8.std())
            print(
                f"[bag] wrote {n:06d} @ {t_ns/1e9:.3f}s  "
                f"rgb(mean={mean:.1f},std={std:.2f}) depth(nz={nz}) "
                f"out_enc={bag.depth_out_encoding} "
                f"size={'%dx%d'%(rgb_u8.shape[1], rgb_u8.shape[0])}"
                f"{'' if bag.resize_wh is None else ' -> %dx%d'%bag.resize_wh}",
                flush=True,
            )
        n += 1


if __name__ == "__main__":
    main()
