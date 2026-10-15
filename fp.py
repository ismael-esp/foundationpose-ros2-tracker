# fp_v1.py  (FoundationPose + YOLO + SAM; ROS 2; no GUI)
# Fixed-topic publishing per label:
#   - Each label always publishes to /<label>_pose (no ..._2_pose).
#   - Reactivation reuses the same topic/publisher.
# Other features preserved (YOLO debug, inactivity gating, reacquire, etc.)

import sys
sys.path.append('./FoundationPose')
sys.path.append('./FoundationPose/nvdiffrast')

import rclpy
from rclpy.node import Node
from estimater import *   # FoundationPose, ScorePredictor, PoseRefinePredictor, draw_posed_3d_box, draw_xyz_axis
import cv2
import numpy as np
import trimesh
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
import argparse
import os, time, glob, threading, queue, math, re
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.transform import Rotation as R
from ultralytics import SAM, YOLO
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from tf2_ros import TransformBroadcaster, Buffer, TransformListener, TransformException

# -------------------------------
# Fixed parameters (adjust if needed)
# -------------------------------
BASE_FRAME = "world"
# CAM_FRAME is overridden after args parsing below (see "CAM_FRAME = args.cam_frame").
# This placeholder keeps type checkers happy but is always replaced at runtime.
CAM_FRAME  = "cam2_color_optical_frame"

# BASE->CAM calibration used ONLY when --self_broadcast_tf is set (standalone, no Docker).
# With Docker running, world→cam_frame is already in the TF tree — self-broadcast is OFF.
#
# # Head camera (cam2, top-down, /head/cam2/...):
# CALIB_T = {
#     'x': 0.007763,
#     'y': 0.439065,
#     'z': 1.96248,
#     'qx': 0.0057113,
#     'qy': 0.9983262,
#     'qz': 0.056996,
#     'qw': 0.0079736,
# }

# extra angle camera (cam2, top-down, /head/cam2/...):
CALIB_T = {
    'x': 0.536218,
    'y': 0.897494,
    'z': 1.438932,
    'qx': -0.300892,
    'qy': -0.146301,
    'qz': 0.881543,
    'qw': -0.333080,
}

# Front/D415 calib fro the right front camera
# CALIB_T = {
#     'x': -0.20351833,
#     'y': 1.49654685,
#     'z': 1.54438168,
#     'qx': -0.03041232,
#     'qy': 0.79420454,
#     'qz': -0.60483816,
#     'qw': 0.04985004,
# }

# -------------------------------
# FoundationPose: tag 'is_register' after register()
# -------------------------------
original_init = FoundationPose.__init__
original_register = FoundationPose.register
def modified_init(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer=None, refiner=None, glctx=None, debug=0, debug_dir='./FoundationPose'):
    original_init(self, model_pts, model_normals, symmetry_tfs, mesh, scorer, refiner, glctx, debug, debug_dir)
    self.is_register = False
def modified_register(self, K, rgb, depth, ob_mask, iteration):
    pose = original_register(self, K, rgb, depth, ob_mask, iteration)
    self.is_register = True
    return pose
FoundationPose.__init__ = modified_init
FoundationPose.register = modified_register

# --- Detect GPU conv capability ---
# PyTorch 2.1.0+cu121 only ships kernels for SM 5.0–9.0. The RTX 5070 Ti is SM 12.0.
# matmul works via PTX JIT fallback, but Conv2d (used by YOLO/SAM) does not.
# FoundationPose uses its own custom CUDA extensions compiled locally with SM 12.0 support,
# so it keeps running on GPU. YOLO and SAM fall back to CPU automatically.
def _probe_det_device() -> str:
    """Return 'cuda:0' if Conv2d runs on the GPU, 'cpu' otherwise."""
    if not torch.cuda.is_available():
        return 'cpu'
    try:
        _m = torch.nn.Conv2d(1, 1, 1).cuda()
        _x = torch.zeros(1, 1, 4, 4).cuda()
        _m(_x)
        del _m, _x
        torch.cuda.empty_cache()
        return 'cuda:0'
    except Exception:
        torch.cuda.empty_cache()
        return 'cpu'

_DET_DEVICE = _probe_det_device()

# -------------------------------
# CLI
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--objects', type=str, default='',
    help='Comma-separated list of object labels to load (mesh filename stems, lowercased). '
         'Empty = load all meshes in demo_data/. '
         'Example: --objects "red cube,mustard,peach,jammanbo"')
parser.add_argument('--mode', choices=['auto','auto_debug'], default='auto',
    help="auto: automatic registration (no GUI). auto_debug: same with extra logs.")
parser.add_argument('--detector_weights', type=str, default='yolo11_06.pt', #boxes_best #yolo11_06
    help="YOLO weights/name (auto-download).")
parser.add_argument('--det_conf', type=float, default=0.80, help='YOLO confidence threshold')
parser.add_argument('--viz_yolo', action='store_true', help='Overlay YOLO boxes/labels on the main overlay image')

# --- New debug overlay toggle ---
parser.add_argument('--publish_yolo_debug', action='store_true',
    help='Publish a second overlay (/inference/yolo_overlay) with YOLO boxes only')
parser.add_argument('--viz_local', action='store_true',
    help='Show result in a local popup window (cv2.imshow) instead of publishing to ROS2 topics (faster)')

parser.add_argument('--viz_yolo_font', type=float, default=0.6, help='OpenCV font scale for YOLO label overlay')

# Tracking / refine
parser.add_argument('--track_refine_iter_base', type=int, default=1, help='Base refine iterations per frame')
parser.add_argument('--track_refine_iter_max',  type=int, default=2, help='Max refine iterations')
parser.add_argument('--iter_gain_per_sec',      type=float, default=1.5, help='Extra refine iters per elapsed sec')

# YOLO anchoring / drift / reacquire
parser.add_argument('--yolo_anchor_period', type=int, default=40, help='Run YOLO every N frames to check drift / manage state')
parser.add_argument('--drift_px_thresh',    type=float, default=40.0, help='Drift thresh (pixels) to trigger reacquire')
parser.add_argument('--lost_reacquire',     action='store_true', default=True, help='Enable background reacquire when lost')

# Registration (initial & reacquire) iterations
parser.add_argument('--est_refine_iter', type=int, default=2, help='Refine iterations used on register()')

# --- Inactivity & gating params ---
parser.add_argument('--expire_sec', type=float, default=4.0,
    help='Seconds without a YOLO match before marking an object INACTIVE (stop track/visualize/publish)')
parser.add_argument('--occlusion_max_frames', type=int, default=45,
    help='Max frames to glide on constant-velocity when track_one fails (default 45 = 1.5 s @ 30 fps)')
parser.add_argument('--stable_ticks_new', type=int, default=2,
    help='YOLO anchor ticks required before running SAM/register for a NEW/REAPPEARING object')
parser.add_argument('--new_min_conf', type=float, default=0.70,
    help='Minimum YOLO confidence for candidates to count toward stability')
parser.add_argument('--gate_radius_px', type=float, default=64.0,
    help='Association radius (px) for matching YOLO dets to tracked objects on anchor ticks')

# Topics
parser.add_argument('--overlay_topic',    type=str, default='/inference/overlay',    help='Main overlay (RELIABLE)')
parser.add_argument('--overlay_topic_be', type=str, default='/inference/overlay_be', help='Main overlay (BEST_EFFORT)')
parser.add_argument('--yolo_overlay_topic', type=str, default='/inference/yolo_overlay', help='YOLO-only overlay (BEST_EFFORT)')

# Camera topics — change these to switch cameras.
# head (cam2, top-down):  /head/cam2/...
# front D415 (cam1, tripod): /front/cam1/...
# back (cam3):            /back/cam3/...
parser.add_argument('--color_topic',      type=str, default='/extra/cam2/color/image_raw')
parser.add_argument('--depth_topic',      type=str, default='/extra/cam2/aligned_depth_to_color/image_raw')
parser.add_argument('--camera_info_topic',type=str, default='/extra/cam2/color/camera_info')

# Camera TF frame.
# 'AUTO' (default): derived from the color_topic — /head/cam2/... → cam2_color_optical_frame.
# This keeps the TF lookup always in sync with the chosen topic; no manual update needed
# when switching cameras. Override only if your frame naming differs from the standard pattern.
parser.add_argument('--cam_frame', type=str, default='AUTO',
    help="TF frame for the camera optical frame. 'AUTO' derives it from --color_topic "
         "(e.g. /head/cam2/... → cam2_color_optical_frame). Override only if your TF "
         "frame name differs from the <camera_name>_color_optical_frame pattern.")

# Gravity alignment: strip roll/pitch from world-frame object pose, keep only yaw.
# Required by FOUNDATIONPOSE_REQUIREMENTS.md — objects on the table must have Z pointing up.
# Physically correct for any object sitting flat on the table (pick-and-place use case).
parser.add_argument('--gravity_align', action='store_true', default=True,
    help='Force object Z-axis to point up in world frame (correct for flat-on-table objects)')
parser.add_argument('--no_gravity_align', dest='gravity_align', action='store_false')

# TF self-broadcast control.
# When Docker is running and already provides world→cam_frame via its TF tree, set False.
# When running standalone (no Docker TF), set True and fill in CALIB_T below.
parser.add_argument('--self_broadcast_tf', action='store_true', default=False,
    help='Self-broadcast world→cam_frame TF using CALIB_T (only needed without Docker TF)')
parser.add_argument('--no_self_broadcast_tf', dest='self_broadcast_tf', action='store_false')

args = parser.parse_args()

# Resolve AUTO cam_frame from the color topic.
# /head/cam2/color/image_raw  → parts[1]='cam2' → 'cam2_color_optical_frame'
# /front/cam1/color/image_raw → parts[1]='cam1' → 'cam1_color_optical_frame'
# /back/cam3/color/image_raw  → parts[1]='cam3' → 'cam3_color_optical_frame'
if args.cam_frame == 'AUTO':
    _topic_parts = args.color_topic.strip('/').split('/')
    _cam_name = _topic_parts[1] if len(_topic_parts) > 1 else 'camera'
    args.cam_frame = f'{_cam_name}_color_optical_frame'

# CAM_FRAME module-level constant used throughout the class.
CAM_FRAME = args.cam_frame

# --- Manual Image Conversion (Bypasses cv_bridge NumPy conflict) ---
def imgmsg_to_numpy(msg):
    if msg.encoding == 'rgb8':
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    elif msg.encoding == 'bgr8':
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif msg.encoding == '16UC1':
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
    elif msg.encoding == '32FC1':
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    return None

def numpy_to_imgmsg(img, encoding, frame_id, stamp):
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = img.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = msg.width * (img.itemsize * (img.shape[2] if len(img.shape) > 2 else 1))
    msg.data = img.tobytes()
    return msg

# -------------------------------
# Node
# -------------------------------
class PoseEstimationNode(Node):
    def __init__(self, mesh_paths):
        super().__init__('pose_estimation_node')

        # --- Subscriptions ---
        self.image_sub = self.create_subscription(Image, args.color_topic, self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, args.depth_topic, self.depth_callback, 10)
        self.info_sub  = self.create_subscription(CameraInfo, args.camera_info_topic, self.camera_info_callback, 10)

        # --- Publishers (overlay) ---
        qos_rel = QoSProfile(depth=1);   qos_rel.reliability = ReliabilityPolicy.RELIABLE;    qos_rel.history = HistoryPolicy.KEEP_LAST; qos_rel.durability = DurabilityPolicy.VOLATILE
        qos_be  = QoSProfile(depth=1);   qos_be.reliability  = ReliabilityPolicy.BEST_EFFORT;  qos_be.history  = HistoryPolicy.KEEP_LAST; qos_be.durability  = DurabilityPolicy.VOLATILE

        self.overlay_pub    = self.create_publisher(Image, args.overlay_topic, qos_rel)
        self.overlay_pub_be = self.create_publisher(Image, args.overlay_topic_be, qos_be)

        self.yolo_overlay_pub = None
        if args.publish_yolo_debug:
            self.yolo_overlay_pub = self.create_publisher(Image, args.yolo_overlay_topic, qos_be)

        self.cam_K  = None

        # Buffers & worker
        self._frame_lock = threading.Lock()
        self._latest = {'color': None, 'depth': None, 'stamp': None}
        self._proc_busy = False
        self._stop = False
        self._proc_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._proc_thread.start()

        # Meshes
        self.mesh_files  = mesh_paths
        self.meshes      = [trimesh.load(m) for m in self.mesh_files]
        self.bounds      = [trimesh.bounds.oriented_bounds(m) for m in self.meshes]
        self.bboxes      = [np.stack([-extents/2, extents/2], axis=0).reshape(2, 3) for (_, extents) in self.bounds]
        # labels from filenames
        self.mesh_labels = [os.path.splitext(os.path.basename(p))[0].lower() for p in self.mesh_files]
        self.get_logger().info(f"Loaded {len(self.meshes)} meshes: {self.mesh_labels}")

        # FoundationPose
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()

        # SAM & YOLO
        self.get_logger().info(
            f"[Detection] device={_DET_DEVICE}  "
            f"(GPU SM={torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A'}; "
            f"YOLO+SAM use Conv2d which requires a compiled kernel — falling back to CPU on SM 12.0 with PyTorch 2.1)"
        )
        self.seg_model = SAM("sam2.1_b.pt")
        try:
            self.detector = YOLO(args.detector_weights or 'yolo11_06.pt')
            self.get_logger().info(f"[YOLO] loaded: {args.detector_weights}")
        except Exception as e:
            self.detector = None
            self.get_logger().warn(f"[YOLO] load failed: {e}")

        # State
        self.tracked = {}            # oid -> dict
        self.pose_publishers = {}    # topic -> publisher (reused)
        self.frame_count = 0
        self.first_registered = False

        # FIXED TOPIC PER LABEL (no suffixing)
        self._label_topic_map = {}   # label -> "/label_pose"

        # Simple YOLO gating cache
        self._cands = {}             # key: (label, gx, gy) -> stats

        # Async YOLO worker
        self._yolo_res = None
        self._yolo_lock = threading.Lock()
        self._yolo_event = threading.Event()   # set by main loop when a fresh YOLO run is wanted
        self._yolo_thread = threading.Thread(target=self._yolo_worker, daemon=True)
        self._yolo_thread.start()

        # Parallel executor for tracking
        self._tp_executor = ThreadPoolExecutor(max_workers=4)

        # Reacquire worker
        self._rq = queue.Queue(maxsize=32)
        self._rq_thread = threading.Thread(target=self._reacquire_loop, daemon=True)
        self._rq_thread.start()

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Object pose TF broadcaster — always on; publishes world→<label>_frame for every tracked object.
        # Add the TF display in RViz to see the live object axes.
        self._obj_tf_broadcaster = TransformBroadcaster(self)
        # Self-broadcast world→cam_frame only when Docker TF is not available.
        # When Docker is running it already provides this via ra_base_link→cam*_link→cam*_color_optical_frame.
        self.get_logger().info(
            f"[Camera] topics={args.color_topic}  TF frame={CAM_FRAME}"
        )
        if args.self_broadcast_tf:
            self.tf_broadcaster = TransformBroadcaster(self)
            self._tf_timer = self.create_timer(1.0/30.0, self._broadcast_camera_tf)
            self.get_logger().info(f"[TF] Self-broadcasting world→{CAM_FRAME} using CALIB_T")
        else:
            self.get_logger().info(f"[TF] Using system TF for world→{CAM_FRAME} (Docker mode)")

    # --- CameraInfo ---
    def camera_info_callback(self, msg):
        if self.cam_K is None:
            self.cam_K = np.array(msg.k, dtype=np.float64).reshape((3, 3))
            self.get_logger().info(f"Camera K initialized: {self.cam_K}")

    # --- Latest frames ---
    def image_callback(self, msg):
        try:
            color = imgmsg_to_numpy(msg)
            if color is None: return
            with self._frame_lock:
                self._latest['color'] = np.ascontiguousarray(color)
                self._latest['stamp'] = time.time()
        except Exception as e:
            self.get_logger().warn(f"image_callback failed: {e}")

    def depth_callback(self, msg):
        try:
            depth_raw = imgmsg_to_numpy(msg)
            if depth_raw is None: return
            if depth_raw.dtype == np.uint16:
                depth = depth_raw.astype(np.float32) / 1000.0
            else:
                depth = depth_raw.astype(np.float32)
            with self._frame_lock:
                self._latest['depth'] = np.ascontiguousarray(depth)
                self._latest['stamp'] = time.time()
        except Exception as e:
            self.get_logger().warn(f"depth_callback failed: {e}")

    # --- Worker loop ---
    def _process_loop(self):
        self.get_logger().info("Worker thread started.")
        have_logged_ready = False
        while not self._stop:
            with self._frame_lock:
                ready = (self._latest['color'] is not None and
                         self._latest['depth'] is not None and
                         self.cam_K is not None)
                if ready:
                    color = self._latest['color']
                    depth = self._latest['depth']
                    ts    = self._latest['stamp']
            if not ready:
                if not have_logged_ready:
                    self.get_logger().info("Waiting for color+depth+camera_info ...")
                    have_logged_ready = True
                time.sleep(0.01)
                continue
            if self._proc_busy:
                time.sleep(0.001)
                continue

            self._proc_busy = True
            try:
                self._process_one(color, depth, ts)
            except Exception as e:
                self.get_logger().error(f"[process_loop] exception: {e}")
            finally:
                self._proc_busy = False

    # --- YOLO ---
    def _yolo_boxes_bgr(self, bgr_img, conf=None):
        out = []
        if self.detector is None:
            return out
        cc = args.det_conf if conf is None else conf
        try:
            res = self.detector.predict(bgr_img, conf=cc, verbose=False, device=_DET_DEVICE)[0]
            if len(res.boxes) == 0:
                return out
            names = getattr(self.detector.model, "names", {})
            for i in range(len(res.boxes)):
                x1, y1, x2, y2 = res.boxes.xyxy[i].int().tolist()
                cls = int(res.boxes.cls[i].item()) if res.boxes.cls is not None else -1
                cf  = float(res.boxes.conf[i].item()) if res.boxes.conf is not None else 0.0
                label = names.get(cls, "obj").lower()
                cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
                out.append((x1, y1, x2, y2, label, cf, cx, cy))
        except Exception as e:
            self.get_logger().warn(f"[YOLO] predict failed: {e}")
        return out

    # --- SAM helpers ---
    def _sam_mask_from_point(self, rgb, point_xy):
        if self.seg_model is None:
            return None
        try:
            x, y = int(point_xy[0]), int(point_xy[1])
            res = self.seg_model.predict(rgb, points=[[x, y]], labels=[1], verbose=False, device=_DET_DEVICE)
            if not res or not res[0] or getattr(res[0], "masks", None) is None:
                return None
            md = res[0].masks
            if hasattr(md, "areas") and md.areas is not None:
                idx = int(np.argmax(md.areas.detach().cpu().numpy()))
                m = md.data[idx].detach().cpu().numpy().astype(np.float32)
            else:
                data = md.data.detach().cpu().numpy().astype(np.float32)
                areas = [np.sum(data[i] > 0.5) for i in range(data.shape[0])]
                idx = int(np.argmax(areas)) if len(areas) else 0
                m = data[idx]
            H, W = rgb.shape[:2]
            m = (cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8) * 255
            return m
        except Exception as e:
            self.get_logger().warn(f"SAM(point) failed: {e}")
            return None

    def _sam_mask_from_box(self, rgb, box):
        if self.seg_model is None:
            return None
        try:
            x1, y1, x2, y2 = [int(v) for v in box]
            res = self.seg_model.predict(rgb, bboxes=[[x1, y1, x2, y2]], verbose=False, device=_DET_DEVICE)
            if not res or not res[0] or getattr(res[0], "masks", None) is None:
                return None
            md = res[0].masks
            if hasattr(md, "areas") and md.areas is not None:
                idx = int(np.argmax(md.areas.detach().cpu().numpy()))
                m = md.data[idx].detach().cpu().numpy().astype(np.float32)
            else:
                data = md.data.detach().cpu().numpy().astype(np.float32)
                areas = [np.sum(data[i] > 0.5) for i in range(data.shape[0])]
                idx = int(np.argmax(areas)) if len(areas) else 0
                m = data[idx]
            H, W = rgb.shape[:2]
            m = (cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8) * 255
            return m
        except Exception as e:
            self.get_logger().warn(f"SAM(box) failed: {e}")
            return None

    def _make_rect_mask(self, rgb_shape, box):
        H, W = rgb_shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        m = np.zeros((H, W), dtype=np.uint8)
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = 255
        return m

    # --- Initial auto register ---
    def _auto_register_on_first_frame(self, color, depth, debug=False):
        yolo = self._yolo_boxes_bgr(color[..., ::-1], conf=args.det_conf)
        if not yolo:
            self.get_logger().warn("AUTO: No YOLO detections; nothing to register.")
            return

        selected = []   # (mesh_index, det)
        for mi, mlabel in enumerate(self.mesh_labels):
            det = self._pick_best_det_for_label(yolo, mlabel)
            if det is not None:
                selected.append((mi, det))

        if not selected:
            self.get_logger().warn("AUTO: No mesh matched YOLO labels; nothing to register.")
            return

        reg_count = 0
        now = time.time()
        for out_idx, (mi, det) in enumerate(sorted(selected, key=lambda x: x[0])):
            temp_mesh = self.meshes[mi]
            to_origin, extents = self.bounds[mi]
            bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
            x1, y1, x2, y2, yolo_label, _, cx, cy = det

            ob_mask = self._sam_mask_from_point(color, (cx, cy))
            if ob_mask is None:
                ob_mask = self._sam_mask_from_box(color, (x1, y1, x2, y2))
            if ob_mask is None:
                ob_mask = self._make_rect_mask(color.shape, (x1, y1, x2, y2))

            try:
                pose_est = FoundationPose(
                    model_pts=temp_mesh.vertices,
                    model_normals=temp_mesh.vertex_normals,
                    mesh=temp_mesh,
                    scorer=self.scorer,
                    refiner=self.refiner,
                    glctx=self.glctx
                )
                pose_est.register(K=self.cam_K, rgb=color, depth=depth,
                                  ob_mask=ob_mask, iteration=args.est_refine_iter)

                pub_topic = self._topic_for_label(yolo_label)

                self.tracked[out_idx] = {
                    'pose_est': pose_est,
                    'to_origin': to_origin,
                    'bbox': bbox,
                    'last_ts': now,
                    'label': yolo_label,
                    'pub_topic': pub_topic,
                    'reacquiring': False,
                    'lock': threading.Lock(),
                    'active': True,
                    'last_yolo_ts': now,
                }
                reg_count += 1
                self.get_logger().info(f"AUTO: registered '{yolo_label}' → topic '{pub_topic}'")
            except Exception as e:
                self.get_logger().warn(f"AUTO: register failed (mesh idx={mi}, label='{yolo_label}'): {e}")

        if reg_count > 0:
            self.first_registered = True
            self.get_logger().info(f"AUTO: registered {reg_count} object(s).")
        else:
            self.get_logger().warn("AUTO: no objects registered.")

    # --- FIXED topic for a label (no numeric suffixes) ---
    def _topic_for_label(self, label: str) -> str:
        base = re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_') or 'object'
        # Fixed, deterministic topic name:
        topic = f"/{base}_pose"
        # Cache (not strictly necessary, but handy if you later want quick lookups)
        self._label_topic_map[label] = topic
        return topic

    def _yolo_worker(self):
        self.get_logger().info("YOLO worker thread started.")
        while not self._stop:
            self._yolo_event.wait(timeout=1.0)
            if self._stop:
                break
            self._yolo_event.clear()
            with self._frame_lock:
                color = self._latest['color'].copy() if self._latest['color'] is not None else None
            if color is None:
                continue
            res = self._yolo_boxes_bgr(color[..., ::-1])
            with self._yolo_lock:
                self._yolo_res = (res, time.time())

    def _track_single(self, oid: int, data: dict, color, depth, ts):
        """Run track_one for one object; returns (oid, pose_or_None, error_or_None)."""
        lock = data['lock']
        with lock:
            pose_est  = data['pose_est']
            to_origin = data['to_origin']
            last_ts   = data['last_ts']

            dt    = max(0.0, ts - last_ts)
            extra = int(max(0.0, dt * args.iter_gain_per_sec))
            iters = min(args.track_refine_iter_max, args.track_refine_iter_base + extra)

            try:
                pose = pose_est.track_one(rgb=color, depth=depth, K=self.cam_K, iteration=iters)
                # EMA velocity update
                if 'last_track_pose' in data:
                    alpha = 0.7
                    raw_vel = np.linalg.inv(data['last_track_pose']) @ pose
                    data['vel_T'] = alpha * raw_vel + (1 - alpha) * data.get('vel_T', np.eye(4))
                else:
                    data['vel_T'] = np.eye(4)
                data['last_track_pose'] = pose
                data['occlusion_frames'] = 0
                data['current_pose'] = pose @ np.linalg.inv(to_origin)
                data['last_ts'] = ts
                return oid, data['current_pose'], None
            except Exception as e:
                if 'vel_T' in data and data.get('occlusion_frames', 0) < args.occlusion_max_frames:
                    data['occlusion_frames'] = data.get('occlusion_frames', 0) + 1
                    data['last_track_pose'] = data['last_track_pose'] @ data['vel_T']
                    data['current_pose'] = data['last_track_pose'] @ np.linalg.inv(to_origin)
                    data['last_ts'] = ts
                    return oid, data['current_pose'], 'occluded'
                return oid, None, e

    # --- Main processing ---
    def _process_one(self, color, depth, ts):
        depth = depth.copy()
        depth[(depth < 0.1) | (~np.isfinite(depth))] = 0

        # Registration pass
        if not self.first_registered:
            self._auto_register_on_first_frame(color, depth, debug=(args.mode=='auto_debug'))
            if not self.first_registered:
                self._publish_overlay(color)
                return

        now = time.time()
        
        # Get latest YOLO results from worker
        yolo_boxes, yolo_ts = None, 0
        with self._yolo_lock:
            if self._yolo_res:
                yolo_boxes, yolo_ts = self._yolo_res

        # YOLO debug overlay (optional)
        if args.publish_yolo_debug and self.yolo_overlay_pub is not None and yolo_boxes is not None:
            dbg = self._draw_yolo_boxes_rgb(color, yolo_boxes)
            self._publish_yolo_overlay(dbg)

        # Cache TF once per frame
        T_base_cam = self._lookup_T(BASE_FRAME, CAM_FRAME)

        # Tracking loop — submit all active objects in parallel
        vis = color.copy()
        dead_ids = []

        active_items = [
            (oid, data) for oid, data in list(self.tracked.items())
            if not data.get('reacquiring', False) and data.get('active', True)
        ]
        futures = {
            oid: self._tp_executor.submit(self._track_single, oid, data, color, depth, ts)
            for oid, data in active_items
        }

        for oid, future in futures.items():
            data = self.tracked[oid]
            r_oid, pose, err = future.result()
            if pose is not None:
                vis = self._visualize_pose_rgb(vis, data['current_pose'], data['bbox'], T_base_cam)
                if T_base_cam is not None:
                    self._publish_pose_from_matrices(data['current_pose'], T_base_cam, data['pub_topic'])
                if err == 'occluded':
                    self.get_logger().info(f"[Occlusion] oid={oid} gliding (frame {data['occlusion_frames']})")
            else:
                self.get_logger().warn(f"Track oid={oid} failed/occluded too long → reacquire: {err}")
                if args.lost_reacquire:
                    self._queue_reacquire(oid)
                else:
                    dead_ids.append(oid)

        for oid in dead_ids:
            self.tracked.pop(oid, None)

        # --- Gating logic (run at YOLO update rate or fixed period) ---
        need_gating = (self.frame_count % args.yolo_anchor_period == 0)
        # Trigger YOLO worker only when needed — not every frame
        if need_gating or not self.first_registered:
            self._yolo_event.set()
        if need_gating and yolo_boxes is not None:
            self._update_candidates(yolo_boxes, now)
            used_det_idx = set()
            
            for oid, data in self.tracked.items():
                if not data.get('active', True):
                    continue
                cp = data.get('current_pose', None)
                if cp is None: continue
                
                px = self._project_center(cp, self.cam_K)
                best_j, best_d = None, 1e9
                for j, det in enumerate(yolo_boxes):
                    if j in used_det_idx or det[4] != data['label']: continue
                    cx, cy = det[6], det[7]
                    d = math.hypot(px[0]-cx, px[1]-cy) if px is not None else 1e9
                    if d < best_d: best_d, best_j = d, j
                
                if best_j is not None and best_d <= args.gate_radius_px:
                    used_det_idx.add(best_j)
                    data['last_yolo_ts'] = now
                    if best_d > args.drift_px_thresh and args.lost_reacquire:
                        self.get_logger().info(f"[Drift] oid={oid} {best_d:.1f}px > {args.drift_px_thresh} → reacquire")
                        self._queue_reacquire(oid)
                else:
                    if now - data.get('last_yolo_ts', now) > args.expire_sec:
                        data['active'] = False
                        self.get_logger().info(f"[INACTIVE] '{data['label']}' (oid={oid}) timed out.")

            self._register_stable_candidates(color, depth, yolo_boxes, used_det_idx, now)

        # Publish/Show overlay
        if args.viz_yolo and yolo_boxes is not None:
            vis = self._draw_yolo_boxes_rgb(vis, yolo_boxes)
        
        if args.viz_local:
            cv2.imshow('Pose Estimation & Tracking', vis[..., ::-1])
            cv2.waitKey(1)
        else:
            self._publish_overlay(vis)
        
        self.frame_count += 1

    # --- Candidates gating ---
    def _update_candidates(self, yolo_boxes, now):
        for det in yolo_boxes:
            x1, y1, x2, y2, label, conf, cx, cy = det
            if conf < args.new_min_conf:
                continue
            gx, gy = int(cx // 32), int(cy // 32)
            key = (label, gx, gy)
            entry = self._cands.get(key, {'count':0, 'conf_sum':0.0})
            entry['count'] = int(entry.get('count', 0)) + 1
            entry['last_ts'] = now
            entry['box'] = (x1, y1, x2, y2)
            entry['cx'] = cx; entry['cy'] = cy
            entry['conf_sum'] = float(entry.get('conf_sum', 0.0)) + float(conf)
            self._cands[key] = entry

        # Drop stale candidates
        stale = [k for k,v in self._cands.items() if now - v.get('last_ts', 0) > max(2.0, args.expire_sec*2)]
        for k in stale:
            self._cands.pop(k, None)

    def _register_stable_candidates(self, color, depth, yolo_boxes, used_det_idx, now):
        lbl2mesh = {lbl: i for i, lbl in enumerate(self.mesh_labels)}

        # Try each candidate that is stable enough
        for key, v in list(self._cands.items()):
            if v['count'] < max(1, int(args.stable_ticks_new)):
                continue
            label = key[0]
            if label not in lbl2mesh:
                continue

            # If there is an ACTIVE tracker of this label near the candidate, skip
            near_active = False
            for oid, data in self.tracked.items():
                if not data.get('active', False) or data['label'] != label:
                    continue
                cp = data.get('_center_pose', None)
                if cp is None:
                    continue
                px = self._project_center(cp, self.cam_K)
                if px is None: 
                    continue
                if math.hypot(px[0]-v['cx'], px[1]-v['cy']) <= args.gate_radius_px:
                    near_active = True
                    break
            if near_active:
                v['count'] = max(0, v['count'] - 1)
                continue

            # Reuse an INACTIVE tracker with same label (keeps topic identical)
            reuse_oid = None
            for oid, data in self.tracked.items():
                if data['label'] == label and not data.get('active', True):
                    reuse_oid = oid
                    break

            # Build a mask (SAM point -> SAM box -> rect)
            ob_mask = self._sam_mask_from_point(color, (v['cx'], v['cy']))
            if ob_mask is None:
                ob_mask = self._sam_mask_from_box(color, v['box'])
            if ob_mask is None:
                ob_mask = self._make_rect_mask(color.shape, v['box'])

            if reuse_oid is not None:
                data = self.tracked[reuse_oid]
                try:
                    with data['lock']:
                        data['pose_est'].register(K=self.cam_K, rgb=color, depth=depth,
                                                  ob_mask=ob_mask, iteration=args.est_refine_iter)
                        data['last_ts'] = now
                        data['last_yolo_ts'] = now
                        data['active'] = True
                        # pub_topic remains unchanged (fixed)
                        self.get_logger().info(f"[REACTIVATED] '{label}' (oid={reuse_oid}) → topic '{data['pub_topic']}'")
                except Exception as e:
                    self.get_logger().warn(f"Reactivation register failed for '{label}': {e}")
            else:
                # Create a NEW tracker using the corresponding mesh
                mi = lbl2mesh[label]
                temp_mesh = self.meshes[mi]
                to_origin, extents = self.bounds[mi]
                bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
                try:
                    pose_est = FoundationPose(
                        model_pts=temp_mesh.vertices,
                        model_normals=temp_mesh.vertex_normals,
                        mesh=temp_mesh,
                        scorer=self.scorer,
                        refiner=self.refiner,
                        glctx=self.glctx
                    )
                    pose_est.register(K=self.cam_K, rgb=color, depth=depth,
                                      ob_mask=ob_mask, iteration=args.est_refine_iter)
                    pub_topic = self._topic_for_label(label)  # FIXED name
                    new_oid = 0 if len(self.tracked)==0 else (max(self.tracked.keys()) + 1)
                    self.tracked[new_oid] = {
                        'pose_est': pose_est,
                        'to_origin': to_origin,
                        'bbox': bbox,
                        'last_ts': now,
                        'label': label,
                        'pub_topic': pub_topic,  # FIXED topic
                        'reacquiring': False,
                        'lock': threading.Lock(),
                        'active': True,
                        'last_yolo_ts': now,
                    }
                    self.get_logger().info(f"[NEW] '{label}' (oid={new_oid}) → topic '{pub_topic}'")
                except Exception as e:
                    self.get_logger().warn(f"New register failed for '{label}': {e}")

            # Consume the candidate so it must be stable again to retrigger
            self._cands.pop(key, None)

    # --- Overlays ---
    def _draw_yolo_boxes_rgb(self, img_rgb, dets):
        vis = img_rgb.copy()
        for (x1, y1, x2, y2, label, conf, cx, cy) in dets:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)   # RGB
            cv2.circle   (vis, (int(cx), int(cy)), 4,                   (255, 0,   0), -1)
            txt = f"{label}:{conf:.2f}"
            cv2.putText(vis, txt, (int(x1), max(0, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, float(args.viz_yolo_font), (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, "[YOLO tick]", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
        return vis

    def _visualize_pose_rgb(self, img_rgb, center_pose, bbox, T_base_cam=None):
        # Use the raw camera-frame pose for visualization so the 3D bounding box
        # projects accurately onto the actual cube in the image.
        # Gravity alignment is applied only to the published PoseStamped and TF.
        vis = draw_posed_3d_box(self.cam_K, img=img_rgb, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=self.cam_K, thickness=3, transparency=0, is_input_rgb=True)
        return vis

    def _publish_overlay(self, img_rgb):
        overlay_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        msg = numpy_to_imgmsg(overlay_bgr, 'bgr8', CAM_FRAME, self.get_clock().now().to_msg())
        self.overlay_pub.publish(msg)       # reliable
        self.overlay_pub_be.publish(msg)    # best effort

    def _publish_yolo_overlay(self, img_rgb):
        if self.yolo_overlay_pub is None:
            return
        overlay_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        msg = numpy_to_imgmsg(overlay_bgr, 'bgr8', CAM_FRAME, self.get_clock().now().to_msg())
        self.yolo_overlay_pub.publish(msg)

    # --- Utils ---
    def _pick_best_det_for_label(self, dets, label):
        cands = [d for d in dets if d[4] == label]
        if not cands:
            return None
        cands.sort(key=lambda d: (d[5], (d[2]-d[0])*(d[3]-d[1])), reverse=True)
        return cands[0]

    def _project_center(self, T, K):
        try:
            if T is None:
                return None
            c = T[:3, 3].reshape(3,1)
            if c[2,0] <= 0: return None
            uv = (K @ (c / c[2,0]))[:2, 0]
            return (float(uv[0]), float(uv[1]))
        except Exception:
            return None

    def _lookup_T(self, parent_frame, child_frame):
        try:
            t = self.tf_buffer.lookup_transform(parent_frame, child_frame, rclpy.time.Time())
        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed ({parent_frame}->{child_frame}): {e}")
            return None
        q_xyzw = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R.from_quat(q_xyzw).as_matrix()
        T[:3, 3] = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z], dtype=np.float64)
        return T

    def _publish_pose_from_matrices(self, T_cam_obj, T_base_cam, topic_name):
        if topic_name not in self.pose_publishers:
            qos_rel = QoSProfile(depth=10)
            qos_rel.reliability = ReliabilityPolicy.RELIABLE
            qos_rel.history = HistoryPolicy.KEEP_LAST
            qos_rel.durability = DurabilityPolicy.VOLATILE
            self.pose_publishers[topic_name] = self.create_publisher(PoseStamped, topic_name, qos_rel)

        # base->obj = (base->cam) * (cam->obj)
        T_base_obj = T_base_cam @ T_cam_obj

        # Gravity alignment: for objects flat on the table the Z axis must point UP in world frame.
        # FoundationPose can lock onto any face of a symmetric object, so without this the
        # published Z may point sideways or down — making DexGraspNet grasps wrong.
        # We extract only the yaw (rotation around world Z) and discard roll/pitch.
        if args.gravity_align:
            yaw = R.from_matrix(T_base_obj[:3, :3]).as_euler('ZYX')[0]
            T_base_obj[:3, :3] = R.from_euler('z', yaw).as_matrix()

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_FRAME

        pos = T_base_obj[:3, 3]
        quat_xyzw = R.from_matrix(T_base_obj[:3, :3]).as_quat()  # [x,y,z,w]

        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = float(quat_xyzw[0])
        msg.pose.orientation.y = float(quat_xyzw[1])
        msg.pose.orientation.z = float(quat_xyzw[2])
        msg.pose.orientation.w = float(quat_xyzw[3])

        self.pose_publishers[topic_name].publish(msg)

        # Broadcast TF: world → <label>_frame
        # Enables the TF display in RViz to show live object axes (X=red, Y=green, Z=blue).
        # Frame name derived from topic: /red_cube_pose → red_cube_frame
        obj_frame = topic_name.strip('/').removesuffix('_pose') + '_frame'
        tf_msg = TransformStamped()
        tf_msg.header.stamp = msg.header.stamp
        tf_msg.header.frame_id = BASE_FRAME
        tf_msg.child_frame_id = obj_frame
        tf_msg.transform.translation.x = float(pos[0])
        tf_msg.transform.translation.y = float(pos[1])
        tf_msg.transform.translation.z = float(pos[2])
        tf_msg.transform.rotation.x = float(quat_xyzw[0])
        tf_msg.transform.rotation.y = float(quat_xyzw[1])
        tf_msg.transform.rotation.z = float(quat_xyzw[2])
        tf_msg.transform.rotation.w = float(quat_xyzw[3])
        self._obj_tf_broadcaster.sendTransform(tf_msg)

    # --- Reacquire support (fast path thread) ---
    def _queue_reacquire(self, oid):
        data = self.tracked.get(oid)
        if data is None or data['reacquiring']:
            return
        data['reacquiring'] = True
        try:
            self._rq.put_nowait(oid)
        except queue.Full:
            self.get_logger().warn("Reacquire queue full; dropping request")
            data['reacquiring'] = False

    def _reacquire_loop(self):
        while not self._stop:
            try:
                oid = self._rq.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._do_reacquire(oid)
            except Exception as e:
                self.get_logger().warn(f"Reacquire oid={oid} failed: {e}")
            finally:
                if oid in self.tracked:
                    self.tracked[oid]['reacquiring'] = False
                self._rq.task_done()

    def _depth_mask_in_box(self, depth_img, box, eps=0.05, min_pix=200, morph=3):
        ix1, iy1, ix2, iy2 = [int(v) for v in box]
        ix1 = max(0, ix1); iy1 = max(0, iy1)
        ix2 = min(depth_img.shape[1], ix2); iy2 = min(depth_img.shape[0], iy2)
        if ix2 <= ix1 or iy2 <= iy1:
            return None

        roi = depth_img[iy1:iy2, ix1:ix2]
        valid = np.isfinite(roi) & (roi > 0)
        if int(valid.sum()) < min_pix:
            return None

        z = roi[valid]
        Zmed = float(np.median(z))
        m = np.zeros_like(roi, dtype=np.uint8)
        m[(np.abs(roi - Zmed) < eps) & valid] = 255

        if morph and m.any():
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

        if int(m.sum()) < min_pix:
            return None

        full = np.zeros(depth_img.shape, dtype=np.uint8)
        full[iy1:iy2, ix1:ix2] = m
        return full

    def _do_reacquire(self, oid):
        with self._frame_lock:
            color = None if self._latest['color'] is None else self._latest['color'].copy()
            depth = None if self._latest['depth'] is None else self._latest['depth'].copy()
            K = None if self.cam_K is None else self.cam_K.copy()
        if color is None or depth is None or K is None:
            return

        data = self.tracked.get(oid)
        if data is None:
            return
        label = data['label']
        lock = data['lock']

        yolo = self._yolo_boxes_bgr(color[..., ::-1], conf=args.det_conf)
        if not yolo:
            return
        dets = [d for d in yolo if d[4] == label]
        if not dets:
            return
        dets.sort(key=lambda d: (d[5], (d[2]-d[0])*(d[3]-d[1])), reverse=True)
        x1, y1, x2, y2, _, _, cx, cy = dets[0]

        ob_mask = self._depth_mask_in_box(depth, (x1, y1, x2, y2))
        if ob_mask is None:
            ob_mask = self._make_rect_mask(color.shape, (x1, y1, x2, y2))

        with lock:
            pose_est = data['pose_est']
            try:
                pose_est.register(K=K, rgb=color, depth=depth,
                                  ob_mask=ob_mask, iteration=args.est_refine_iter)
                data['last_ts'] = time.time()
                data['last_yolo_ts'] = time.time()
                data['active'] = True
                return
            except Exception as e:
                self.get_logger().warn(f"Fast reacquire failed (depth/rect). oid={oid} err={e}")

        ob_mask_sam = self._sam_mask_from_point(color, (cx, cy))
        if ob_mask_sam is None:
            ob_mask_sam = self._sam_mask_from_box(color, (x1, y1, x2, y2))
        if ob_mask_sam is None:
            return

        with lock:
            try:
                pose_est.register(K=K, rgb=color, depth=depth,
                                  ob_mask=ob_mask_sam, iteration=args.est_refine_iter)
                data['last_ts'] = time.time()
                data['last_yolo_ts'] = time.time()
                data['active'] = True
            except Exception as e:
                self.get_logger().warn(f"SAM reacquire failed. oid={oid} err={e}")

    # --- TF broadcaster: BASE->CAM ---
    def _broadcast_camera_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = BASE_FRAME
        t.child_frame_id = CAM_FRAME
        t.transform.translation.x = float(CALIB_T['x'])
        t.transform.translation.y = float(CALIB_T['y'])
        t.transform.translation.z = float(CALIB_T['z'])
        t.transform.rotation.x = float(CALIB_T['qx'])
        t.transform.rotation.y = float(CALIB_T['qy'])
        t.transform.rotation.z = float(CALIB_T['qz'])
        t.transform.rotation.w = float(CALIB_T['qw'])
        self.tf_broadcaster.sendTransform(t)

# -------------------------------
# main
# -------------------------------
def main(args_ros=None):
    source_directory = "demo_data"
    file_paths = glob.glob(os.path.join(source_directory, '**', '*.obj'), recursive=True) + \
                 glob.glob(os.path.join(source_directory, '**', '*.stl'), recursive=True) + \
                 glob.glob(os.path.join(source_directory, '**', '*.STL'), recursive=True)

    # Filter to only the requested objects (--objects "red cube,mustard,peach")
    # Each label is the mesh filename stem, lowercased — must match YOLO class names exactly.
    if args.objects.strip():
        wanted = {o.strip().lower() for o in args.objects.split(',') if o.strip()}
        file_paths = [p for p in file_paths
                      if os.path.splitext(os.path.basename(p))[0].lower() in wanted]
        missing = wanted - {os.path.splitext(os.path.basename(p))[0].lower() for p in file_paths}
        if missing:
            print(f"[WARN] --objects: no mesh file found for: {missing}")
        print(f"[INFO] Loading {len(file_paths)} mesh(es): "
              f"{[os.path.splitext(os.path.basename(p))[0].lower() for p in file_paths]}")
    else:
        print(f"[INFO] Loading all {len(file_paths)} meshes from {source_directory}/")

    if not file_paths:
        print("[ERROR] No mesh files found — check --objects labels and demo_data/ contents.")
        return

    mesh_paths = file_paths

    rclpy.init(args=args_ros)
    node = PoseEstimationNode(mesh_paths)
    try:
        rclpy.spin(node)
    finally:
        node._stop = True
        try:
            node._proc_thread.join(timeout=1.0)
            node._rq_thread.join(timeout=1.0)
        except:
            pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
