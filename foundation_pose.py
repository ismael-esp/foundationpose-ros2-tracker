#!/usr/bin/env python3
# foundation_pose.py
# Single-object FoundationPose 6D tracker for the BODex pick-and-place hardware port.
#
# Why this exists:
#   fp.py was tuned for a close, tilted camera and depends on a per-object-trained
#   YOLO11 plus a periodic "YOLO anchor" loop that makes it slow and hard to integrate.
#   The camera is now a FAR fixed front tripod (/front_right/cam1). This node:
#     - detects the target ONCE (training-free: depth-cluster -> [YOLO-World] -> click+SAM2)
#     - then runs a pure FoundationPose track_one loop (no periodic YOLO)
#     - holds the pose through hand-grasp occlusion via kinematic hand-attach
#     - always publishes /<slug>_pose in world frame with the object Z-axis UP
#       (hard BODex / FOUNDATIONPOSE_REQUIREMENTS.md contract)
#
# fp.py is intentionally left untouched.

import sys
sys.path.append('./FoundationPose')
sys.path.append('./FoundationPose/nvdiffrast')

# IMPORTANT: `from estimater import *` MUST come before the ROS2 message imports.
# estimater's wildcard exports a name `Image` (a module, e.g. PIL.Image); if it
# is imported AFTER `from sensor_msgs.msg import Image` it shadows the ROS2
# message class and create_subscription() fails with
# "type object 'module' has no attribute '_TYPE_SUPPORT'". fp.py relies on this
# same ordering. Keep ROS message imports LAST so they cannot be shadowed.
from estimater import *  # FoundationPose, ScorePredictor, PoseRefinePredictor, dr, torch, draw_posed_3d_box, draw_xyz_axis
from ultralytics import SAM

import os
import re
import math
import time
import glob
import argparse
import threading

import cv2
import numpy as np
import trimesh

try:
    import open3d as o3d
    _HAVE_O3D = True
except Exception:
    _HAVE_O3D = False

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from scipy.spatial.transform import Rotation as R
from scipy import ndimage as ndi
from tf2_ros import TransformBroadcaster, Buffer, TransformListener, TransformException
# ROS2 message imports LAST (must not be shadowed by `from estimater import *`).
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped

BASE_FRAME = "world"

# ---------------------------------------------------------------------------
# Camera presets: ROS2 topics + optical TF frame + extrinsic (pose of the
# camera optical frame expressed in /world == T_world_cam). FoundationPose
# returns the object pose in the camera frame; CALIB_T lifts it into /world
# (used by --cam_world_source calib; the live Docker TF tree is opt-in).
# Re-run cam_2_base_transform.py and update these if a camera moves.
# ---------------------------------------------------------------------------
CAMERAS = {
    # head, top-down (recommended for the table — better depth than the far
    # grazing front view).
    'head': {
        'color': '/head/cam2/color/image_raw',
        'depth': '/head/cam2/aligned_depth_to_color/image_raw',
        'info':  '/head/cam2/color/camera_info',
        'frame': 'cam2_color_optical_frame',
        'calib': {'x': 0.007763, 'y': 0.439065, 'z': 1.96248,
                  'qx': 0.0057113, 'qy': 0.9983262,
                  'qz': 0.056996, 'qw': 0.0079736},
    },
    # front-right D415 tripod (far, grazing — poor depth on small objects).
    'front': {
        'color': '/front_right/cam1/color/image_raw',
        'depth': '/front_right/cam1/aligned_depth_to_color/image_raw',
        'info':  '/front_right/cam1/color/camera_info',
        'frame': 'cam1_color_optical_frame',
        'calib': {'x': -0.20351833, 'y': 1.49654685, 'z': 1.54438168,
                  'qx': -0.03041232, 'qy': 0.79420454,
                  'qz': -0.60483816, 'qw': 0.04985004},
    },
    # extra-left D415 tripod (from the left, grazing — poor depth on small objects).
    'extra': {
        'color': '/extra/cam2/color/image_raw',
        'depth': '/extra/cam2/aligned_depth_to_color/image_raw',
        'info':  '/extra/cam2/color/camera_info',
        'frame': 'cam2_color_optical_frame',
        'calib': {'x': 1.033741, 'y':1.028537 , 'z': 1.550534,
                  'qx': 0.406223, 'qy': 0.796689,
                  'qz': -0.414394, 'qw': -0.168958},
    },
}

def calib_matrix(c):
    """4x4 T_world_cam from a camera preset's calib (quaternion xyzw + xyz)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat([c['qx'], c['qy'], c['qz'], c['qw']]).as_matrix()
    T[:3, 3] = [c['x'], c['y'], c['z']]
    return T


# ---------------------------------------------------------------------------
# FoundationPose: tag 'is_register' after register().
# NOTE: the real signature is register(self, K, rgb, depth, ob_mask, ob_id=None,
# glctx=None, iteration=5). fp.py's monkey-patch forwarded `iteration`
# POSITIONALLY into `ob_id` (so est_refine_iter was silently ignored and
# iteration stayed 5). Here we forward it by keyword — bug fixed.
# ---------------------------------------------------------------------------
_original_init = FoundationPose.__init__
_original_register = FoundationPose.register


def _modified_init(self, model_pts, model_normals, symmetry_tfs=None, mesh=None,
                    scorer=None, refiner=None, glctx=None, debug=0,
                    debug_dir='./FoundationPose'):
    _original_init(self, model_pts, model_normals, symmetry_tfs, mesh, scorer,
                   refiner, glctx, debug, debug_dir)
    self.is_register = False


def _modified_register(self, K, rgb, depth, ob_mask, iteration):
    pose = _original_register(self, K, rgb, depth, ob_mask, iteration=iteration)
    self.is_register = True
    return pose


FoundationPose.__init__ = _modified_init
FoundationPose.register = _modified_register


# ---------------------------------------------------------------------------
# Detection device probe (copied from fp.py:72-87).
# FoundationPose keeps the GPU via its own CUDA extensions; SAM/YOLO-World use
# Conv2d which may need a CPU fallback depending on the installed torch build.
# ---------------------------------------------------------------------------
def _probe_det_device() -> str:
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


# ---------------------------------------------------------------------------
# Manual image conversion — never reintroduce cv_bridge (NumPy 1.x conflict).
# Verbatim from fp.py:188-209.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Pose / geometry helpers
# ---------------------------------------------------------------------------
def invert_T(T):
    Ti = np.eye(4, dtype=np.float64)
    Rm = T[:3, :3]
    Ti[:3, :3] = Rm.T
    Ti[:3, 3] = -Rm.T @ T[:3, 3]
    return Ti


def lookup_T(tf_buffer, parent_frame, child_frame, logger=None):
    """world<-child 4x4, or None. Adapted from fp.py:862-872."""
    try:
        t = tf_buffer.lookup_transform(parent_frame, child_frame, rclpy.time.Time())
    except TransformException as e:
        if logger is not None:
            logger.warn(f"TF lookup failed ({parent_frame}->{child_frame}): {e}")
        return None
    q = [t.transform.rotation.x, t.transform.rotation.y,
         t.transform.rotation.z, t.transform.rotation.w]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(q).as_matrix()
    T[:3, 3] = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
    return T


def gravity_align_yaw(T_world_obj, yaw_axis='x', last_yaw=None, yaw_ema=0.0):
    """Force object Z-axis exactly up in world; keep only yaw.

    Robust replacement for fp.py's singular `as_euler('ZYX')[0]` (fp.py:890):
    project a body axis onto the world XY plane and atan2. Falls back to the
    other in-plane axis, then to the previous yaw, when the chosen axis is
    near-vertical (exactly the FoundationPose side-face lock-on case).
    Returns (T_aligned, yaw).
    """
    Rm = T_world_obj[:3, :3]
    primary = 0 if yaw_axis == 'x' else 1
    a = Rm[:, primary]
    if math.hypot(a[0], a[1]) >= 1e-6:
        yaw = math.atan2(a[1], a[0])
    else:
        b = Rm[:, 1 - primary]
        if math.hypot(b[0], b[1]) >= 1e-6:
            yaw = math.atan2(b[1], b[0])
        elif last_yaw is not None:
            yaw = last_yaw
        else:
            yaw = 0.0

    if yaw_ema > 0.0 and last_yaw is not None:
        # scalar smoothing on the shortest-arc delta (no matrix blending)
        d = math.atan2(math.sin(yaw - last_yaw), math.cos(yaw - last_yaw))
        yaw = last_yaw + (1.0 - yaw_ema) * d

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.from_euler('z', yaw).as_matrix()  # 3rd column == [0,0,1] exactly
    out[:3, 3] = T_world_obj[:3, 3]
    return out, yaw


def project_center(T_cam_obj, K):
    """Project the object origin to pixels. None if behind camera."""
    try:
        c = T_cam_obj[:3, 3].reshape(3, 1)
        if c[2, 0] <= 0:
            return None
        uv = (K @ (c / c[2, 0]))[:2, 0]
        return float(uv[0]), float(uv[1])
    except Exception:
        return None


def bbox_corners(bbox):
    """8 corners of an axis-aligned bbox given as [[xmin,ymin,zmin],[xmax,...]]."""
    lo, hi = bbox[0], bbox[1]
    return np.array([[x, y, z] for x in (lo[0], hi[0])
                     for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], dtype=np.float64)


def fill_depth_holes(depth, max_px=15):
    """Nearest-valid hole-fill of a sparse depth map (meters; 0/NaN = invalid).

    ROOT-CAUSE FIX for the 'frozen box' failure:
    stock FoundationPose.track_one (estimater.py:257-258) runs
    erode_depth(radius=2) + bilateral_filter on the depth, then the refiner
    aligns the model to the resulting xyz_map. A small object far from the
    RealSense (here a ~7.5 cm cube ~1.9 m top-down) returns only sparse, holey
    depth; erosion removes the few valid border pixels, the refiner gets a
    near-empty point cloud at the object, produces ~zero delta and just returns
    pose_last every frame -> the box stays exactly at the registration spot
    while the scene moves (exactly what error_status.mp4 shows). register()
    survived this only because it has a guess_translation (median mask depth)
    fallback; track_one has none.

    Densifying *only* within max_px of a real measurement reconstructs the
    local object surface so the post-erode xyz_map still has geometry, without
    fabricating far-field depth (we never bridge gaps wider than max_px)."""
    invalid = ~np.isfinite(depth) | (depth <= 0)
    if not invalid.any() or invalid.all():
        return depth
    dist, (iy, ix) = ndi.distance_transform_edt(
        invalid, return_distances=True, return_indices=True)
    out = depth.copy()
    fillable = invalid & (dist <= max_px)
    out[fillable] = depth[iy[fillable], ix[fillable]]
    return out


# ---------------------------------------------------------------------------
# PerfTimer (C4) — per-stage profiling with a 1 Hz aggregate log line.
# Zero behavior change; instrumentation only. Required first because the
# adaptive iter scheduler and the bf16 quality gate rely on its IoU output.
# ---------------------------------------------------------------------------
class PerfTimer:
    """Accumulate wall-time per named stage in a 1-second window; flush via
    `format_line()` when a window closes. Use as a context manager:

        with timer.stage('refine'):
            pose = pose_est.track_one(...)

    Cheap: a single `time.perf_counter()` pair per stage, dict update."""

    def __init__(self, window_sec=1.0):
        self._stages = {}     # name -> [total_sec_in_window, count_in_window]
        self._window = window_sec
        self._t_window_start = time.perf_counter()

    class _StageCtx:
        __slots__ = ('parent', 'name', 't0')
        def __init__(self, parent, name):
            self.parent = parent
            self.name = name
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *exc):
            dt = time.perf_counter() - self.t0
            s = self.parent._stages.setdefault(self.name, [0.0, 0])
            s[0] += dt
            s[1] += 1
            return False

    def stage(self, name):
        return PerfTimer._StageCtx(self, name)

    def window_due(self):
        return (time.perf_counter() - self._t_window_start) >= self._window

    def format_line(self):
        """Aggregate per-stage line; resets the window counters."""
        parts = []
        total_ms = 0.0
        for name, (tot, n) in self._stages.items():
            if n <= 0:
                continue
            mean_ms = (tot / n) * 1000.0
            parts.append(f"{name}={mean_ms:.1f}({n})")
            total_ms += mean_ms
        parts.append(f"total~{total_ms:.1f}ms")
        self._stages.clear()
        self._t_window_start = time.perf_counter()
        return "[perf] " + " ".join(parts)


# ---------------------------------------------------------------------------
# QualityScorer (B2 core) — IoU between a SAM2 mask and a rendered silhouette
# of the mesh at a given camera-frame pose. Used by:
#   * YoloWorldDetector to re-rank candidate boxes when a prior pose exists
#   * _SamHelper.pick_by_iou to choose among SAM2 multi-mask candidates
#   * _track_step / _hybrid_step to feed the adaptive iter scheduler and the
#     bf16 quality gate
#
# Silhouette: convex hull of the projected mesh vertices (sampled to <=N).
# Approximate (over-estimates concave shapes) but fast (no GPU tensor mgmt,
# no nvdiffrast clip-space math) and discriminates correct vs wrong poses
# very well — that's all we need for IoU as a quality signal.
# ---------------------------------------------------------------------------
class QualityScorer:
    def __init__(self, mesh, max_verts=384):
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        if len(verts) > max_verts:
            idx = np.linspace(0, len(verts) - 1, max_verts, dtype=np.int64)
            verts = verts[idx]
        self._verts = verts  # (N,3) object-frame
        # Per-frame cache: (pose_id, K_id) -> silhouette uint8
        self._cache = {}

    def reset_cache(self):
        self._cache = {}

    def silhouette(self, T_cam_obj, K, shape):
        """uint8 (H,W) binary silhouette of the mesh at T_cam_obj."""
        if T_cam_obj is None:
            return None
        key = (id(T_cam_obj), id(K))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        H, W = shape
        v = self._verts
        Nv = len(v)
        v_h = np.hstack([v, np.ones((Nv, 1))])
        v_cam = (T_cam_obj @ v_h.T).T[:, :3]
        front = v_cam[:, 2] > 1e-3
        if int(front.sum()) < 3:
            sil = np.zeros((H, W), dtype=np.uint8)
            self._cache[key] = sil
            return sil
        v_cam = v_cam[front]
        uv = (K @ (v_cam / v_cam[:, 2:3]).T).T[:, :2]
        uv = np.round(uv).astype(np.int32)
        uv[:, 0] = np.clip(uv[:, 0], 0, W - 1)
        uv[:, 1] = np.clip(uv[:, 1], 0, H - 1)
        try:
            hull = cv2.convexHull(uv.reshape(-1, 1, 2))
        except Exception:
            sil = np.zeros((H, W), dtype=np.uint8)
            self._cache[key] = sil
            return sil
        sil = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(sil, [hull], 255)
        self._cache[key] = sil
        return sil

    @staticmethod
    def iou(mask_a, mask_b):
        """Binary IoU of two uint8 masks (>0 ⇒ foreground). Returns 0..1."""
        if mask_a is None or mask_b is None:
            return 0.0
        if mask_a.shape != mask_b.shape:
            return 0.0
        a = mask_a > 0
        b = mask_b > 0
        inter = int(np.logical_and(a, b).sum())
        union = int(np.logical_or(a, b).sum())
        return (inter / float(union)) if union > 0 else 0.0

    def score(self, mask, T_cam_obj, K):
        """Score = IoU(mask, silhouette(pose)). Returns 0.0 with no prior."""
        if T_cam_obj is None or mask is None:
            return 0.0
        sil = self.silhouette(T_cam_obj, K, mask.shape[:2])
        return self.iou(mask, sil)


# ---------------------------------------------------------------------------
# Detectors — each exposes detect(rgb, depth, K, logger) -> uint8 mask | None.
# rgb is RGB (HxWx3 uint8); SAM/YOLO-World are fed BGR (fp.py:407,431 bug fixed).
# ---------------------------------------------------------------------------
class _SamHelper:
    """Shared SAM2 prompting (point / box). BGR in, binary 255-mask out.

    Tier B (2026-06-15) — multi-mask IoU pick:
      from_point/from_box now accept optional `prior_pose, K, scorer`.
      When all three are given, SAM2's multi-mask output (md.data is (N,H,W))
      is scored by IoU vs the rendered mesh silhouette at `prior_pose` and the
      best one is returned. Without a prior, fall back to the largest mask
      (original behavior; preserves SEARCHING / first-lock semantics)."""

    def __init__(self, seg_model):
        self.seg = seg_model

    @staticmethod
    def _all_candidates(md, H, W):
        """Decode SAM2 output into a list of uint8 (H,W) masks (resized)."""
        if md is None:
            return []
        data = md.data.detach().cpu().numpy().astype(np.float32)
        out = []
        for i in range(data.shape[0]):
            m = (cv2.resize(data[i], (W, H), interpolation=cv2.INTER_NEAREST)
                 > 0.5).astype(np.uint8) * 255
            out.append(m)
        return out

    @staticmethod
    def _pick_largest_from_list(masks):
        if not masks:
            return None
        areas = [int((m > 0).sum()) for m in masks]
        return masks[int(np.argmax(areas))] if areas else None

    @staticmethod
    def _pick_best_iou(masks, prior_pose, K, scorer):
        """Pick the mask whose IoU vs the rendered silhouette is highest.
        Returns (best_mask, best_iou) or (None, 0.0)."""
        if not masks or prior_pose is None or scorer is None:
            return None, 0.0
        sil = scorer.silhouette(prior_pose, K, masks[0].shape[:2])
        if sil is None or int((sil > 0).sum()) < 4:
            return None, 0.0
        ious = [QualityScorer.iou(m, sil) for m in masks]
        i = int(np.argmax(ious))
        return masks[i], float(ious[i])

    def _resolve_pick(self, masks, prior_pose, K, scorer, logger, where):
        if not masks:
            return None
        if prior_pose is not None and scorer is not None:
            best, iou = self._pick_best_iou(masks, prior_pose, K, scorer)
            if best is not None:
                if logger and len(masks) > 1:
                    logger.info(f"[SAM {where}] picked by IoU={iou:.2f} "
                                f"({len(masks)} candidates)")
                return best
        return self._pick_largest_from_list(masks)

    def from_point(self, rgb, xy, logger=None, prior_pose=None, K=None,
                   scorer=None):
        try:
            bgr = rgb[..., ::-1]
            x, y = int(xy[0]), int(xy[1])
            res = self.seg.predict(bgr, points=[[x, y]], labels=[1],
                                   verbose=False, device=_DET_DEVICE)
            if not res or not res[0] or getattr(res[0], "masks", None) is None:
                return None
            masks = self._all_candidates(res[0].masks, rgb.shape[0], rgb.shape[1])
            return self._resolve_pick(masks, prior_pose, K, scorer, logger, 'point')
        except Exception as e:
            if logger:
                logger.warn(f"SAM(point) failed: {e}")
            return None

    def from_box(self, rgb, box, logger=None, prior_pose=None, K=None,
                 scorer=None):
        try:
            bgr = rgb[..., ::-1]
            x1, y1, x2, y2 = [int(v) for v in box]
            res = self.seg.predict(bgr, bboxes=[[x1, y1, x2, y2]],
                                   verbose=False, device=_DET_DEVICE)
            if not res or not res[0] or getattr(res[0], "masks", None) is None:
                return None
            masks = self._all_candidates(res[0].masks, rgb.shape[0], rgb.shape[1])
            return self._resolve_pick(masks, prior_pose, K, scorer, logger, 'box')
        except Exception as e:
            if logger:
                logger.warn(f"SAM(box) failed: {e}")
            return None


class DepthClusterDetector:
    """Training-free auto primary: remove the tabletop plane (RANSAC) then take
    the foreground component selected by `cluster_select`. open3d is used for the
    robust plane fit; connected-components does the clustering on the cleaned 2D
    mask (far faster than DBSCAN on a CPU point cloud, same intent)."""

    def __init__(self, args):
        self.a = args
        self.last_info = None  # diagnostics: chosen component bbox/centroid/area
        self.obj_diameter = None  # set by the node from the mesh; enables size gate

    def detect(self, rgb, depth, K, logger=None):
        a = self.a
        H, W = depth.shape[:2]
        valid = np.isfinite(depth) & (depth > 0.1) & (depth < 3.0)
        if int(valid.sum()) < a.min_obj_pixels:
            return None

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        ys, xs = np.where(valid)
        z = depth[ys, xs].astype(np.float64)
        X = (xs - cx) / fx * z
        Y = (ys - cy) / fy * z
        pts = np.stack([X, Y, z], axis=1)

        # Plane fit on a strided subsample for speed.
        if not _HAVE_O3D:
            if logger:
                logger.warn("open3d not available; skipping depth-cluster detector")
            return None
        step = max(1, pts.shape[0] // 60000)
        sub = pts[::step]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sub)
        try:
            plane, _ = pcd.segment_plane(distance_threshold=a.plane_dist,
                                         ransac_n=3, num_iterations=200)
        except Exception as e:
            if logger:
                logger.warn(f"segment_plane failed: {e}")
            return None
        n = np.array(plane[:3], dtype=np.float64)
        d = float(plane[3])
        denom = np.linalg.norm(n) + 1e-9

        # Signed distance to the table plane and a plane (u,v) basis.
        sd = (pts @ n + d) / denom            # signed; table inliers ~ 0
        nn = n / denom
        a0 = np.array([1.0, 0, 0]) if abs(nn[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(nn, a0); u /= (np.linalg.norm(u) + 1e-9)
        v = np.cross(nn, u)
        pu = pts @ u
        pv = pts @ v

        # Table footprint from plane inliers (robust 1..99 percentile box).
        inl = np.abs(sd) <= a.plane_dist
        if int(inl.sum()) < 50:
            if logger:
                logger.info(f"[DepthCluster] table plane has only {int(inl.sum())} "
                            f"inliers; skip")
            return None
        umin, umax = np.percentile(pu[inl], [1, 99])
        vmin, vmax = np.percentile(pv[inl], [1, 99])
        mg = a.table_margin

        # Keep points that REST ON the table: a small height band above the
        # plane AND inside the table footprint. This excludes the robot/torso
        # (behind the far table edge / too tall) — root-cause fix for issue 1.
        h = np.abs(sd)
        on_table = ((h > a.table_h_lo) & (h < a.table_h_hi) &
                    (pu > umin - mg) & (pu < umax + mg) &
                    (pv > vmin - mg) & (pv < vmax + mg))
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[ys[on_table], xs[on_table]] = 255

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        num, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        cand = []
        img_c = np.array([W / 2.0, H / 2.0])
        rej_size = 0
        diam = self.obj_diameter
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < a.cluster_min_pts:
                continue
            cyi, cxi = np.where(lab == i)
            zc = depth[cyi, cxi].astype(np.float64)
            mvalid = zc > 0
            if int(mvalid.sum()) < 4:
                continue
            zc = zc[mvalid]
            Xc = (cxi[mvalid] - cx) / fx * zc
            Yc = (cyi[mvalid] - cy) / fy * zc
            size3d = float(max(np.ptp(Xc), np.ptp(Yc), np.ptp(zc)))
            if diam is not None:
                lo, hi = a.size_lo * diam, a.size_hi * diam
                if not (lo <= size3d <= hi):
                    rej_size += 1
                    continue
            if a.cluster_select == 'best_size' and diam is not None:
                key = -abs(size3d - diam)          # most object-sized wins
            elif a.cluster_select == 'largest':
                key = float(area)
            elif a.cluster_select == 'center':
                key = -float(np.linalg.norm(cents[i] - img_c))
            else:  # nearest (smallest mean depth)
                key = -float(np.mean(zc))
            cand.append((key, i, area, size3d))
        if not cand:
            self.last_info = None
            if logger:
                logger.info(
                    f"[DepthCluster] {num-1} on-table comp(s); none passed "
                    f"area>={a.cluster_min_pts}+size gate (rej_size={rej_size}, "
                    f"obj_diameter={diam})")
            return None
        cand.sort(reverse=True)
        best, best_area, best_sz = cand[0][1], cand[0][2], cand[0][3]
        bx, by = int(stats[best, cv2.CC_STAT_LEFT]), int(stats[best, cv2.CC_STAT_TOP])
        bw, bh = int(stats[best, cv2.CC_STAT_WIDTH]), int(stats[best, cv2.CC_STAT_HEIGHT])
        ccx, ccy = float(cents[best][0]), float(cents[best][1])
        self.last_info = {'bbox': (bx, by, bx + bw, by + bh),
                          'centroid': (ccx, ccy), 'area': best_area,
                          'ncand': len(cand), 'size3d': best_sz}
        if logger:
            logger.info(
                f"[DepthCluster] {len(cand)} on-table cand (rej_size={rej_size}), "
                f"select={a.cluster_select} -> area={best_area}px "
                f"size3d={best_sz:.3f}m (obj~{diam:.3f}m) "
                f"bbox=({bx},{by},{bx+bw},{by+bh}) centroid=({ccx:.0f},{ccy:.0f})")
        out = np.where(lab == best, 255, 0).astype(np.uint8)
        if int((out > 0).sum()) < a.cluster_min_pts:
            self.last_info = None
            return None
        return out


class YoloWorldDetector:
    """Open-vocabulary text-prompted detector (no per-object training).

    Tier B/C (2026-06-15):
      * --object_aliases CSV adds extra prompts (e.g. snorlax → "plush pokemon,
        stuffed toy"); set_classes is called with [primary, *aliases].
      * --yolo_world_size {x,l,m,s} selects the model size (default 'l' is
        ~2x faster than the previous hardcoded 'x' with marginal recall loss).
      * When a prior pose exists, candidate boxes are scored by
        `confidence × (0.1 + IoU(box, rendered_silhouette))` so a high-conf box
        on the wrong object loses to a slightly-lower-conf box on the right one.
    """

    def __init__(self, args, sam_helper, scorer_ref=None):
        self.a = args
        self.sam = sam_helper
        self.model = None
        # Diagnostics: all raw (x1,y1,x2,y2,label,conf,score). `score` is the
        # ranking value actually used (= conf without prior pose, conf*(0.1+IoU)
        # with prior pose).
        self.last_dets = []
        # Callable that returns the current prior pose (None when in SEARCHING
        # or after track loss). Allows the node to inject a live reference
        # without coupling this class to the node's lifecycle.
        self._prior_pose_fn = None
        self._K_fn = None
        self._scorer_ref = scorer_ref  # QualityScorer (None until node init)

    def set_context(self, prior_pose_fn, K_fn, scorer_ref):
        self._prior_pose_fn = prior_pose_fn
        self._K_fn = K_fn
        self._scorer_ref = scorer_ref

    def _ensure(self, logger=None):
        if self.model is not None:
            return True
        try:
            from ultralytics import YOLOWorld
            sz = getattr(self.a, 'yolo_world_size', 'l')
            model_name = f"yolov8{sz}-worldv2.pt"
            self.model = YOLOWorld(model_name)
            prompts = [self.a.object] + list(
                getattr(self.a, 'object_aliases_list', []) or [])
            self.model.set_classes(prompts)
            if logger:
                logger.info(f"[YOLO-World] model={model_name} prompts={prompts}")
            return True
        except Exception as e:
            if logger:
                logger.warn(f"[YOLO-World] init failed: {e}")
            return False

    def detect(self, rgb, depth, K, logger=None):
        self.last_dets = []
        if not self._ensure(logger):
            return None
        try:
            res = self.model.predict(rgb[..., ::-1], conf=self.a.det_conf,
                                     verbose=False, device=_DET_DEVICE)[0]
            names = getattr(self.model, 'names', {}) or {}
            nb = 0 if res.boxes is None else len(res.boxes)
            raw = []
            for j in range(nb):
                bx = res.boxes.xyxy[j].int().tolist()
                cf = float(res.boxes.conf[j].item())
                cl = int(res.boxes.cls[j].item()) if res.boxes.cls is not None else -1
                raw.append((bx[0], bx[1], bx[2], bx[3],
                            str(names.get(cl, self.a.object)), cf))
            if nb == 0:
                if logger:
                    logger.warning(f"[YOLO-World] prompts={[self.a.object]+ list(getattr(self.a,'object_aliases_list',[]) or [])} "
                                   f"conf>={self.a.det_conf}: 0 det(s)")
                return None

            # Re-rank by IoU vs rendered silhouette when a prior pose exists.
            prior_pose = (self._prior_pose_fn() if self._prior_pose_fn else None)
            scored = []
            if prior_pose is not None and self._scorer_ref is not None:
                H, W = rgb.shape[:2]
                sil = self._scorer_ref.silhouette(prior_pose, K, (H, W))
                for d in raw:
                    x1, y1, x2, y2, _, cf = d
                    bm = np.zeros((H, W), dtype=np.uint8)
                    bm[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = 255
                    iou_b = QualityScorer.iou(bm, sil)
                    s = cf * (0.1 + iou_b)
                    scored.append((s, iou_b, d))
            else:
                for d in raw:
                    scored.append((d[5], 0.0, d))  # plain confidence

            scored.sort(key=lambda x: -x[0])
            # Materialize last_dets in ranking order with the score attached.
            self.last_dets = [(d[0], d[1], d[2], d[3], d[4], d[5], s, iou_b)
                              for s, iou_b, d in scored]
            best = scored[0]
            if logger:
                summ = ", ".join(
                    f"{d[4]}:c={d[5]:.2f}{(',iou='+format(iou_b,'.2f')) if iou_b>0 else ''}"
                    for s, iou_b, d in scored[:5])
                logger.warning(f"[YOLO-World] {nb} det -> top by score "
                               f"({'iou-reranked' if prior_pose is not None else 'conf-only'}): "
                               f"{summ}")
            x1, y1, x2, y2 = best[2][:4]
        except Exception as e:
            if logger:
                logger.warn(f"[YOLO-World] predict failed: {e}")
            return None
        # Pass scorer + K + prior to SAM so multi-mask is picked by IoU too.
        return self.sam.from_box(rgb, (x1, y1, x2, y2), logger,
                                 prior_pose=prior_pose, K=K,
                                 scorer=self._scorer_ref)


class ClickSamDetector:
    """One-time interactive fallback. Click the object once -> SAM2 point mask.
    Adapted from foundationpose_ros_multi.py:_on_mouse (508-549)."""

    def __init__(self, args, sam_helper):
        self.a = args
        self.sam = sam_helper

    def detect(self, rgb, depth, K, logger=None):
        win = f"SELECT '{self.a.object}'  (click object | c/Enter confirm | r reset | q cancel)"
        state = {'pt': None}

        def _on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                state['pt'] = (int(x), int(y))

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, _on_mouse)
        confirmed = False
        while True:
            disp = rgb[..., ::-1].copy()
            if state['pt'] is not None:
                cv2.circle(disp, state['pt'], 6, (0, 0, 255), -1)
            cv2.putText(disp, f"target: {self.a.object}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(win, disp)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), 27):
                cv2.destroyWindow(win)
                if logger:
                    logger.info("click-select: canceled")
                return None
            if key == ord('r'):
                state['pt'] = None
            if key in (ord('c'), 13, 32) and state['pt'] is not None:
                confirmed = True
                break
        cv2.destroyWindow(win)
        if not confirmed or state['pt'] is None:
            return None
        m = self.sam.from_point(rgb, state['pt'], logger)
        if m is None:
            if logger:
                logger.warn("click-select: SAM returned no mask")
        return m


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Single-object FoundationPose tracker (BODex port)")
    p.add_argument('--object', type=str, required=True,
                   help='Object label = mesh stem, e.g. "red cube". '
                        'Mesh = demo_data/<object>/<object>.obj ; topic = /<slug>_pose')
    p.add_argument('--hand_frame', type=str, required=True,
                   help='REQUIRED. Grasping hand TF frame for kinematic attach '
                        '(e.g. ra_palm_link or la_palm_link).')

    p.add_argument('--camera', type=str, default='head',
                   choices=list(CAMERAS.keys()),
                   help="Camera preset (topics + optical frame + CALIB_T). "
                        "'head' = top-down (default, better table depth); "
                        "'front' = far front-right tripod.")
    # Default None -> filled from the --camera preset after parsing; pass these
    # only to override the preset.
    p.add_argument('--color_topic', type=str, default=None)
    p.add_argument('--depth_topic', type=str, default=None)
    p.add_argument('--camera_info_topic', type=str, default=None)
    p.add_argument('--cam_frame', type=str, default=None,
                   help='Camera optical frame (overlay header; TF lookup when '
                        '--cam_world_source uses TF). Defaults from --camera.')
    p.add_argument('--cam_world_source', type=str, default='calib',
                   choices=['calib', 'tf', 'auto'],
                   help="How to get world<-camera_optical (T_world_cam): "
                        "'calib'=the --camera preset's CALIB_T (default); "
                        "'tf'=Docker TF tree (world->cam_frame); "
                        "'auto'=TF if available else CALIB_T.")

    p.add_argument('--est_refine_iter', type=int, default=2)
    # Tier B+C (2026-06-15): adaptive iteration count. The old `--track_refine_iter`
    # becomes `--track_refine_iter_max`; an alias for back-compat is wired below.
    p.add_argument('--track_refine_iter_max', type=int, default=4,
                   help='FoundationPose refine iterations CAP per tracked frame. '
                        'Used as a constant when adaptive iter is off; the upper '
                        'bound the adaptive scheduler escalates to.')
    p.add_argument('--track_refine_iter_min', type=int, default=1,
                   help='Adaptive iter floor (frames with high IoU + low dT run '
                        'at this iteration count to save GPU).')
    p.add_argument('--track_refine_iter_boost_frames', type=int, default=10,
                   help='When the adaptive scheduler escalates (low IoU or big '
                        'dT), hold the max iter count for this many frames '
                        'before decaying back toward min.')
    p.add_argument('--track_refine_iter', type=int, default=None,
                   help='[deprecated alias for --track_refine_iter_max].')
    p.add_argument('--adaptive_iter', dest='adaptive_iter', action='store_true',
                   default=True,
                   help='Enable adaptive track_refine_iter scheduling '
                        '(default ON; based on rolling IoU and pose dT).')
    p.add_argument('--no_adaptive_iter', dest='adaptive_iter', action='store_false')
    p.add_argument('--adaptive_iou_lo', type=float, default=0.40,
                   help='Adaptive iter: IoU below this triggers boost to max.')
    p.add_argument('--adaptive_dt_hi', type=float, default=0.030,
                   help='Adaptive iter: per-frame pose delta (m) above this '
                        'triggers boost to max.')
    p.add_argument('--depth_fill', action='store_true', default=True,
                   help='Nearest-valid hole-fill of sparse depth before '
                        'track_one. ROOT-CAUSE fix for the frozen box: stock '
                        'track_one erodes sparse far-object depth to nothing so '
                        'the refiner cannot follow the object. On by default.')
    p.add_argument('--no_depth_fill', dest='depth_fill', action='store_false')
    p.add_argument('--depth_fill_max_px', type=int, default=15,
                   help='Max pixel radius to fill from a real depth '
                        'measurement (do not fabricate far-field depth).')
    p.add_argument('--track_mode', type=str, default='hybrid',
                   choices=['hybrid', 'reregister', 'track_one'],
                   help="Steady-state estimator. 'hybrid' (default, fp.py-proven "
                        "= fluent+robust): track_one EVERY frame (smooth, fast, "
                        "temporally continuous => NO flicker) + SAM2 re-anchor "
                        "for recovery/drift (YOLO-World only when there is no "
                        "prior lock). 'reregister' = register every frame (robust "
                        "but flickers on symmetric objects, slow). 'track_one' = "
                        "pure track loop, no recovery.")
    p.add_argument('--reanchor_period', type=int, default=30,
                   help='hybrid: re-anchor with SAM2->register every N frames to '
                        'correct track_one drift (also triggers on track loss).')
    p.add_argument('--reanchor_refine_iter', type=int, default=1,
                   help='hybrid/reregister: register() iterations on a re-anchor '
                        '(1 = fast; raise if re-anchored pose is poor).')
    p.add_argument('--reacquire_after', type=int, default=8,
                   help='hybrid: after this many consecutive failed recoveries, '
                        'accept the YOLO-World detection UNCONDITIONALLY (gate '
                        'bypass) so a lost track can never deadlock.')
    p.add_argument('--reanchor_gate_px', type=float, default=220.0,
                   help='Sanity radius (px) for a YOLO-World re-anchor vs the '
                        'last pose projection. Bypassed after --reacquire_after '
                        'fails (no deadlock) and unused by the SAM2@pose path.')
    p.add_argument('--reanchor_warmup', type=int, default=8,
                   help='reregister mode: frames after the initial confirmed '
                        'register during which the spatial gate is relaxed.')
    p.add_argument('--yaw_mode', type=str, default='lock',
                   choices=['lock', 'track'],
                   help="Published yaw (Z is ALWAYS forced up). 'lock' (default, "
                        "BODex-stable): freeze yaw at the first confirmed "
                        "register — a symmetric cube's yaw is arbitrary, this "
                        "kills axis flicker and gives BODex a steady value. "
                        "'track' = follow yaw with --yaw_ema smoothing.")
    p.add_argument('--pos_ema', type=float, default=0.3,
                   help='EMA weight on the PREVIOUS published position (0=off). '
                        '0.3 lightly smooths jitter for the BODex 7-sample '
                        'median latch without lag on the slow pick.')
    p.add_argument('--auto_attempts', type=int, default=5,
                   help='Auto-detection attempts before falling back to click.')
    p.add_argument('--max_mask_frac', type=float, default=0.15,
                   help='Reject any registration mask larger than this fraction '
                        'of the image (image-wide SAM2 mask -> giant register '
                        'crop -> CUDA OOM). 0.15 = 15%% of image.')
    p.add_argument('--detect_mode', type=str, default='auto_confirm',
                   choices=['auto_confirm', 'auto', 'click'],
                   help="auto_confirm (default): auto proposes a candidate, you "
                        "accept it with a key or click the correct object to "
                        "override. auto: accept first valid auto detection "
                        "(no confirm). click: always pick by clicking.")
    p.add_argument('--size_lo', type=float, default=0.3,
                   help='Min cluster real size as a multiple of the mesh diameter '
                        '(DepthCluster size gate; rejects the robot/torso blob).')
    p.add_argument('--size_hi', type=float, default=4.0,
                   help='Max cluster real size as a multiple of the mesh diameter.')
    p.add_argument('--auto_detector', type=str, default='yoloworld',
                   choices=['yoloworld', 'depthcluster'],
                   help="Auto detector: 'yoloworld' (default) = YOLO-World "
                        "open-vocab box on --object -> SAM2 mask (RGB only, no "
                        "depth). 'depthcluster' = the depth-tabletop method "
                        "(unreliable on this far/red setup — kept for fallback).")
    p.add_argument('--use_yolo_world', action='store_true',
                   help='Deprecated no-op (YOLO-World is the default auto '
                        'detector; use --auto_detector).')
    p.add_argument('--object_aliases', type=str, default='',
                   help='Comma-separated extra YOLO-World prompts (B1). '
                        'e.g. --object_aliases "plush pokemon,stuffed toy". '
                        'set_classes is called with [primary, *aliases]; the '
                        'best candidate is picked by confidence and (when a '
                        'prior pose exists) IoU vs the rendered silhouette.')
    p.add_argument('--yolo_world_size', type=str, default='l',
                   choices=['x', 'l', 'm', 's'],
                   help='YOLO-World model size (C3). Default `l` (~2x faster '
                        'than `x` with marginal recall loss for our catalog).')
    p.add_argument('--debug_detect', action='store_true', default=True,
                   help='Draw detector candidates (DepthCluster bbox + YOLO-World '
                        'raw boxes) on the SEARCHING overlay.')
    p.add_argument('--no_debug_detect', dest='debug_detect', action='store_false')
    p.add_argument('--det_conf', type=float, default=0.30)

    # depth-cluster detector
    p.add_argument('--plane_dist', type=float, default=0.01,
                   help='RANSAC table-plane inlier threshold (m).')
    p.add_argument('--cluster_min_pts', type=int, default=40,
                   help='Min component pixel area to be a candidate.')
    p.add_argument('--cluster_select', type=str, default='best_size',
                   choices=['best_size', 'largest', 'center', 'nearest'],
                   help="best_size (default): pick the on-table cluster whose "
                        "real 3D size best matches the mesh diameter.")
    p.add_argument('--table_h_lo', type=float, default=0.006,
                   help='Min height above the table plane to count as object (m).')
    p.add_argument('--table_h_hi', type=float, default=0.40,
                   help='Max height above the table plane (m) — caps tall things.')
    p.add_argument('--table_margin', type=float, default=0.05,
                   help='Footprint margin (m) added around the detected table '
                        'extent; only points inside the table footprint are kept '
                        '(excludes the robot behind the far table edge).')
    p.add_argument('--mask_depth_band', type=float, default=0.0,
                   help='If >0, clip the registration mask to pixels within this '
                        'depth band (m) of the seed/median depth. DEFAULT 0 = '
                        'DISABLED — use the SAM2 mask as-is (depth on this far/red '
                        'setup is too sparse and shredded the mask). FoundationPose '
                        'still uses depth internally for pose; this only controls '
                        'mask post-processing.')
    p.add_argument('--min_obj_pixels', type=int, default=80,
                   help='Min mask pixels to accept a detection / count as visible '
                        '(lowered for a far, small object).')

    # visibility / occlusion
    p.add_argument('--require_vis', action='store_true', default=False,
                   help='If set, drop to OCCLUDED when the depth-consistency '
                        'visibility ratio < --track_vis_thresh. DEFAULT OFF: '
                        'stay TRACKING as long as FoundationPose track_one '
                        'succeeds (the depth here is too poor to gate on — a '
                        'strict gate makes a perfectly-tracked object "lost").')
    p.add_argument('--track_vis_thresh', type=float, default=0.15)
    p.add_argument('--vis_depth_eps', type=float, default=0.05)
    p.add_argument('--relock_vis_thresh', type=float, default=0.30)
    p.add_argument('--relock_stable_frames', type=int, default=5)
    p.add_argument('--brief_hold_frames', type=int, default=3)
    p.add_argument('--occluded_max_sec', type=float, default=8.0)
    p.add_argument('--offset_avg_frames', type=int, default=5)

    # pose output
    p.add_argument('--yaw_axis', type=str, default='x', choices=['x', 'y'])
    p.add_argument('--yaw_ema', type=float, default=0.0)
    p.add_argument('--gravity_align', action='store_true', default=True,
                   help='[legacy] Force published mesh Z-axis up. Use '
                        '--orientation_mode for the modern 3-way switch.')
    p.add_argument('--no_gravity_align', dest='gravity_align',
                   action='store_false')
    p.add_argument('--orientation_mode', type=str, default=None,
                   choices=['locked', 'tracked', 'raw'],
                   help="Published orientation policy (one flag to rule three).\n"
                        "  locked  (default) = Z forced up + yaw FROZEN at the "
                        "first register. BODex-friendly; right for symmetric "
                        "objects (cube). Derives: --gravity_align "
                        "--yaw_mode lock.\n"
                        "  tracked = Z forced up + yaw follows the live "
                        "FoundationPose pose's yaw. Useful when the object can "
                        "spin on the table and downstream wants the spin. "
                        "Derives: --gravity_align --yaw_mode track.\n"
                        "  raw     = NO post-processing — publish the full 6D "
                        "FoundationPose pose verbatim. Pitch/roll/yaw all "
                        "follow the mesh frame; the published Z-axis is no "
                        "longer guaranteed up. Use to *see* what "
                        "FoundationPose really thinks. Derives: "
                        "--no_gravity_align. WARNING: BREAKS the BODex "
                        "contract (FOUNDATIONPOSE_REQUIREMENTS §2b); do not "
                        "run a pick-and-place against it.")
    p.add_argument('--stamp_mode', type=str, default='sensor', choices=['sensor', 'now'])

    p.add_argument('--viz_local', action='store_true')
    p.add_argument('--overlay_topic', type=str, default='/inference/overlay')
    p.add_argument('--overlay_topic_be', type=str, default='/inference/overlay_be')

    # Tier C: BF16 autocast + per-stage profiling
    p.add_argument('--precision', type=str, default='auto',
                   choices=['fp32', 'tf32', 'bf16', 'auto'],
                   help="Refiner numeric precision (C1). 'auto' = tf32 on "
                        "CUDA SM>=80 (Ampere/Ada/Blackwell), fp32 elsewhere. "
                        "'tf32' is the safe speedup (~25-35%% faster, no "
                        "ops broken). 'bf16' is RISKY: stock FoundationPose "
                        "calls torch.linalg.inv which does NOT support "
                        "BFloat16 — register/track_one will raise 'Low "
                        "precision dtypes' and we auto-revert to fp32 the "
                        "first time it does. 'fp32' = no acceleration.")
    p.add_argument('--precision_iou_drop_max', type=float, default=0.05,
                   help='Auto-revert threshold for bf16 quality gate.')
    p.add_argument('--prof_log', dest='prof_log', action='store_true',
                   default=True,
                   help='Emit 1 Hz per-stage timing breakdown (C4 PerfTimer).')
    p.add_argument('--no_prof_log', dest='prof_log', action='store_false')

    a = p.parse_args()
    # Back-compat alias: --track_refine_iter overrides --track_refine_iter_max.
    if a.track_refine_iter is not None:
        a.track_refine_iter_max = a.track_refine_iter
    a.track_refine_iter = a.track_refine_iter_max  # legacy attribute (read sites)
    # Parse aliases CSV
    a.object_aliases_list = [s.strip() for s in (a.object_aliases or '').split(',')
                             if s.strip()]
    # --orientation_mode is the modern face of --gravity_align + --yaw_mode.
    # When set, it OVERRIDES whatever those two were (so users don't have to
    # think about the legacy combination).
    if a.orientation_mode is not None:
        if a.orientation_mode == 'locked':
            a.gravity_align = True
            a.yaw_mode = 'lock'
        elif a.orientation_mode == 'tracked':
            a.gravity_align = True
            a.yaw_mode = 'track'
        elif a.orientation_mode == 'raw':
            a.gravity_align = False
            # yaw_mode is unused when gravity_align is off, leave as-is.
    else:
        # Reflect the resolved mode back so logs show what's actually active.
        if not a.gravity_align:
            a.orientation_mode = 'raw'
        else:
            a.orientation_mode = ('locked' if a.yaw_mode == 'lock'
                                  else 'tracked')
    # Fill camera topic/frame defaults from the --camera preset (only where the
    # user did not explicitly override them).
    cam = CAMERAS[a.camera]
    if a.color_topic is None:
        a.color_topic = cam['color']
    if a.depth_topic is None:
        a.depth_topic = cam['depth']
    if a.camera_info_topic is None:
        a.camera_info_topic = cam['info']
    if a.cam_frame is None:
        a.cam_frame = cam['frame']
    return a


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
WAIT_INPUTS, SEARCHING, TRACKING, OCCLUDED = 'WAIT_INPUTS', 'SEARCHING', 'TRACKING', 'OCCLUDED'


class SingleObjectPoseNode(Node):
    def __init__(self, args, mesh_path):
        super().__init__('foundation_pose_node')
        self.a = args

        # --- Subscriptions ---
        self.create_subscription(Image, args.color_topic, self._color_cb, 10)
        self.create_subscription(Image, args.depth_topic, self._depth_cb, 10)
        self.create_subscription(CameraInfo, args.camera_info_topic, self._info_cb, 10)

        # --- Publishers ---
        qos_rel = QoSProfile(depth=1)
        qos_rel.reliability = ReliabilityPolicy.RELIABLE
        qos_rel.history = HistoryPolicy.KEEP_LAST
        qos_rel.durability = DurabilityPolicy.VOLATILE
        qos_be = QoSProfile(depth=1)
        qos_be.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_be.history = HistoryPolicy.KEEP_LAST
        qos_be.durability = DurabilityPolicy.VOLATILE

        self.slug = re.sub(r'[^a-z0-9]+', '_', args.object.lower()).strip('_') or 'object'
        self.pose_topic = f"/{self.slug}_pose"
        self.obj_frame = f"{self.slug}_frame"
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, qos_rel)
        self.overlay_pub = self.create_publisher(Image, args.overlay_topic, qos_rel)
        self.overlay_pub_be = self.create_publisher(Image, args.overlay_topic_be, qos_be)

        # --- TF ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.T_world_cam_calib = calib_matrix(CAMERAS[args.camera]['calib'])

        # --- Frame buffers ---
        self.cam_K = None
        self._frame_lock = threading.Lock()
        self._latest = {'color': None, 'depth': None, 'stamp': None, 'wall': None}
        self._stop = False
        self._proc_busy = False

        # --- Mesh + FoundationPose ---
        mesh = trimesh.load(mesh_path, force='mesh')
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        self.mesh = mesh
        self.to_origin = to_origin
        self.bbox = np.stack([-extents / 2.0, extents / 2.0], axis=0).reshape(2, 3)
        self.get_logger().info(f"Loaded mesh '{args.object}' -> topic {self.pose_topic} "
                               f"(verts={len(mesh.vertices)}, extents={np.round(extents,4)})")

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.get_logger().info(f"[Detection] device={_DET_DEVICE}")
        self.seg_model = SAM("sam2.1_b.pt")
        sam_helper = _SamHelper(self.seg_model)
        self.sam_helper = sam_helper

        # Tier B+C foundation (2026-06-15): PerfTimer + QualityScorer
        self.perf = PerfTimer(window_sec=1.0)
        self.qscorer = QualityScorer(mesh, max_verts=384)

        # Auto detector (RGB-only by default): YOLO-World (open-vocab box on
        # --object) -> SAM2 mask. Depth-cluster kept only as an explicit fallback.
        self.obj_diameter = float(np.linalg.norm(extents))
        if args.auto_detector == 'depthcluster':
            dc = DepthClusterDetector(args)
            dc.obj_diameter = self.obj_diameter
            self.detectors = [dc]
        else:
            yw = YoloWorldDetector(args, sam_helper, scorer_ref=self.qscorer)
            # Inject "what is the current prior pose?" and "what is K?" as
            # callables so the detector can score candidates without coupling
            # back to the node's internals.
            yw.set_context(
                prior_pose_fn=lambda: getattr(self, 'cam_pose', None),
                K_fn=lambda: self.cam_K,
                scorer_ref=self.qscorer)
            self.detectors = [yw]
        self.click_detector = ClickSamDetector(args, sam_helper)  # SAM2-from-click
        self.get_logger().info(
            f"[Detectors] mode={args.detect_mode} auto={args.auto_detector} "
            f"(YOLO-World size='{args.yolo_world_size}' "
            f"prompts={[args.object]+args.object_aliases_list} -> SAM2; "
            f"click=SAM2 point) obj_diameter={self.obj_diameter:.3f}m "
            f"mask_depth_band={args.mask_depth_band} "
            f"({'OFF — raw SAM2 mask' if args.mask_depth_band<=0 else 'on'})")

        # ---- C1: precision plan (resolved from --precision) ----
        # 'auto' → 'tf32' on CUDA SM>=80 (Ampere/Ada/Blackwell), else 'fp32'.
        # 'tf32' is enabled via global flags (no autocast wrap) — it's the safe
        # speedup and does not break torch.linalg.inv. 'bf16' uses
        # torch.autocast and will trip on stock FoundationPose's linalg.inv;
        # we catch that exception once at runtime and auto-revert to fp32.
        prec = args.precision
        try:
            major, minor = (torch.cuda.get_device_capability(0)
                            if torch.cuda.is_available() else (0, 0))
        except Exception:
            major, minor = (0, 0)
        sm = major * 10 + minor
        if prec == 'auto':
            prec = 'tf32' if (torch.cuda.is_available() and sm >= 80) else 'fp32'
        self._precision = prec
        self._precision_locked = False  # True after a quality-gated revert
        # Apply TF32 globally (safe; one-shot at startup).
        if self._precision == 'tf32' and torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                try:
                    torch.set_float32_matmul_precision('high')
                except Exception:
                    pass
            except Exception as e:
                self.get_logger().warn(f"[Precision] TF32 enable failed: {e}")
        self.get_logger().info(
            f"[Precision] requested={args.precision} resolved={self._precision} "
            f"sm={sm} (bf16 auto-reverts on linalg.inv bf16 error)")
        # Make the orientation policy unmistakable in the log: this is the
        # difference between "Z always up, BODex-compatible" and "raw 6D, you
        # see what FoundationPose actually thinks".
        om = args.orientation_mode
        if om == 'raw':
            self.get_logger().warn(
                f"[Orientation] mode=raw  -> publishing FULL 6D FoundationPose "
                f"pose verbatim. Mesh-Z is NOT forced up; pitch/roll/yaw all "
                f"reflect the mesh frame. This BREAKS the BODex contract "
                f"(FOUNDATIONPOSE_REQUIREMENTS §2b) — do NOT run "
                f"bodex_hw/run_bodex_pick_place_hw.py against this topic.")
        else:
            self.get_logger().info(
                f"[Orientation] mode={om}  (gravity_align={args.gravity_align}, "
                f"yaw_mode={args.yaw_mode}) — published Z forced up"
                f"{', yaw frozen at first register' if om=='locked' else ''}")

        # ---- Adaptive iter state (C2) ----
        self._iter_now = args.track_refine_iter_min
        self._iter_boost_left = 0
        self._iou_window = []  # last K successful-track IoUs (bf16 quality gate)
        self._iou_window_max = 60

        self.pose_est = FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
            mesh=mesh, scorer=self.scorer, refiner=self.refiner, glctx=self.glctx)

        # --- State ---
        self.state = WAIT_INPUTS
        self.cam_pose = None          # raw 4x4 object-in-camera (TRACKING only)
        self.last_world_aligned = None  # last published gravity-aligned 4x4
        self.last_yaw = None
        self._yaw_lock = None         # frozen yaw for --yaw_mode lock
        self._pos_ema = None          # EMA state for --pos_ema
        self._frame_n = 0             # hybrid frame counter
        self._reacq_fail = 0          # hybrid consecutive recovery failures
        self._last_src = '?'          # what produced the last good pose
        self._anchor_px_initial = None  # mask px at the first confirmed register
        self._manual_redetect = False  # set when user presses 'n' in the viewer
        self.hand_offset = None       # T_hand_object (full 6D)
        self._solid_vis = 0
        self._relock_ok = 0
        self._brief_hold = 0
        self._auto_tries = 0
        self._last_detector = None
        self._reg_seed = None
        self._last_track_t = None
        self._occ_start = None
        self._tf_warned = False
        self._tf_ok_logged = False

        # rate/latency telemetry
        self._pub_times = []
        self._last_tele = time.time()

        self._proc_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._proc_thread.start()
        self.get_logger().info(
            f"[Camera] preset={args.camera} color={args.color_topic} "
            f"cam_frame={args.cam_frame} hand_frame={args.hand_frame} "
            f"cam_world_source={args.cam_world_source}")
        if args.cam_world_source in ('calib', 'auto'):
            self.get_logger().info(
                f"[Calib] T_world_cam (CALIB_T) t={np.round(self.T_world_cam_calib[:3,3],4)}")

    # ---- Callbacks ----
    def _info_cb(self, msg):
        if self.cam_K is None:
            self.cam_K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(f"Camera K initialized:\n{self.cam_K}")

    def _color_cb(self, msg):
        try:
            c = imgmsg_to_numpy(msg)
            if c is None:
                return
            with self._frame_lock:
                self._latest['color'] = np.ascontiguousarray(c)
                self._latest['stamp'] = msg.header.stamp
                self._latest['wall'] = time.time()
        except Exception as e:
            self.get_logger().warn(f"color_cb failed: {e}")

    def _depth_cb(self, msg):
        try:
            d = imgmsg_to_numpy(msg)
            if d is None:
                return
            d = d.astype(np.float32) / 1000.0 if d.dtype == np.uint16 else d.astype(np.float32)
            with self._frame_lock:
                self._latest['depth'] = np.ascontiguousarray(d)
        except Exception as e:
            self.get_logger().warn(f"depth_cb failed: {e}")

    # ---- Worker ----
    def _process_loop(self):
        self.get_logger().info("Worker thread started.")
        logged_wait = False
        while not self._stop:
            with self._frame_lock:
                ready = (self._latest['color'] is not None and
                         self._latest['depth'] is not None and
                         self.cam_K is not None)
                if ready:
                    color = self._latest['color']
                    depth = self._latest['depth']
                    stamp = self._latest['stamp']
                    wall = self._latest['wall']
            if not ready:
                if not logged_wait:
                    self.get_logger().info("Waiting for color+depth+camera_info ...")
                    logged_wait = True
                time.sleep(0.01)
                continue
            if self._proc_busy:
                time.sleep(0.001)
                continue
            self._proc_busy = True
            try:
                self._process_one(color, depth.copy(), stamp, wall)
            except Exception as e:
                self.get_logger().error(f"[process] exception: {e}")
            finally:
                self._proc_busy = False

    # ---- world<-camera_optical (T_world_cam) ----
    def _get_T_world_cam(self):
        """Resolve T_world_cam per --cam_world_source. CALIB_T does NOT need the
        Docker camera TF chain (the plan's blocking risk #1); the live TF tree is
        still used for --hand_frame regardless of this setting."""
        src = self.a.cam_world_source
        if src == 'calib':
            return self.T_world_cam_calib

        T = lookup_T(self.tf_buffer, BASE_FRAME, self.a.cam_frame, None)
        if T is not None:
            if self._tf_warned or not self._tf_ok_logged:
                self.get_logger().info(
                    f"[TF] {BASE_FRAME}->{self.a.cam_frame} OK  t={np.round(T[:3,3],4)}")
                self._tf_ok_logged = True
                self._tf_warned = False
            return T

        if src == 'auto':
            if not self._tf_warned:
                self.get_logger().warn(
                    f"[TF] {BASE_FRAME}->{self.a.cam_frame} unavailable; "
                    f"falling back to CALIB_T")
                self._tf_warned = True
            return self.T_world_cam_calib

        # src == 'tf'
        if not self._tf_warned:
            self.get_logger().error(
                f"[TF] {BASE_FRAME}->{self.a.cam_frame} unresolvable and "
                f"--cam_world_source=tf. Is the Docker TF chain "
                f"(world->cam1_link->{self.a.cam_frame}) up? Staying in WAIT_INPUTS.")
            self._tf_warned = True
        return None

    # ---- Main state machine ----
    def _process_one(self, color, depth, stamp, wall):
        # B2: per-frame silhouette cache. Keyed by pose-id; the prior pose
        # mutates each frame so the cached silhouette must not leak across.
        self.qscorer.reset_cache()
        depth[(depth < 0.1) | (~np.isfinite(depth))] = 0

        T_world_cam = self._get_T_world_cam()
        if T_world_cam is None:
            self.state = WAIT_INPUTS
            self._publish_overlay(color, "WAIT: no world->cam TF (--cam_world_source=tf)")
            return
        if self.state == WAIT_INPUTS:
            self.state = SEARCHING
            self.get_logger().info(
                f"Inputs ready (cam->world via {self.a.cam_world_source}) -> SEARCHING")

        # Manual 'n' override: drop the current lock and re-run the SAM2
        # selector so the user can click the object again. Honored from any
        # state except WAIT_INPUTS (which already lacks a lock to drop).
        if self._manual_redetect and self.state in (TRACKING, OCCLUDED, SEARCHING):
            self._manual_redetect = False
            # Reclaim GPU memory before the new register() — long-running
            # sessions fragment the allocator and the next register can OOM
            # even with a sane mask.
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            self.cam_pose = None
            self._yaw_lock = None
            self._pos_ema = None
            self._frame_n = 0
            self._reacq_fail = 0
            self._anchor_px_initial = None
            self._brief_hold = 0
            self._auto_tries = 0
            self._occ_start = None
            self.hand_offset = None
            self.state = SEARCHING
            self.get_logger().warn(
                f"[manual] 'n' -> SEARCHING (mode={self.a.detect_mode}; "
                "click the object to re-select)")

        if self.state == SEARCHING:
            self._do_search(color, depth)
            if self.a.debug_detect:
                self._draw_search_debug(color)
            else:
                self._publish_overlay(color, f"SEARCHING mode={self.a.detect_mode}")
            return

        # TRACKING / OCCLUDED: steady-state estimator. Default 'hybrid'
        # (track_one every frame + SAM2 re-anchor recovery; see _hybrid_step).
        if self.a.track_mode == 'hybrid':
            tracked_pose, vis_ratio = self._hybrid_step(color, depth)
        elif self.a.track_mode == 'reregister':
            tracked_pose, vis_ratio = self._reregister_step(color, depth)
        else:
            tracked_pose, vis_ratio = self._track_step(color, depth)

        ok_vis = (vis_ratio >= self.a.track_vis_thresh
                  if self.a.require_vis else True)
        if tracked_pose is not None and ok_vis:
            tnow = tracked_pose[:3, 3]
            dT = (float(np.linalg.norm(tnow - self._last_track_t)) * 1000.0
                  if self._last_track_t is not None else 0.0)
            self._last_track_t = tnow.copy()
            self.cam_pose = tracked_pose
            self.state = TRACKING
            self._brief_hold = 0
            self._occ_start = None
            T_world_obj = T_world_cam @ tracked_pose
            self._maybe_cache_offset(T_world_obj, vis_ratio)
            self._publish(T_world_obj, stamp, wall)
            # Overlay: raw box (accurate projection) + the gravity-aligned
            # PUBLISHED pose axes (what BODex receives — Z must be up).
            T_cam_pub = invert_T(T_world_cam) @ self.last_world_aligned
            zx, zy, zz = self.last_world_aligned[:3, 2]
            if self.a.track_mode == 'reregister':
                diag = (f"anchor={getattr(self,'_anchor_src','?')} "
                        f"mask={getattr(self,'_anchor_px',0)}px "
                        f"gateD={getattr(self,'_anchor_gate_d',-1):.0f}px")
            else:
                pf = getattr(self, '_obj_px_fill', (0, 0.0))
                diag = (f"src={getattr(self,'_last_src','?')} "
                        f"objpx={pf[0]} med={pf[1]:.3f}m "
                        f"yaw={'LOCK' if self.a.yaw_mode=='lock' else 'trk'}")
            status = (f"TRACKING [{self.a.track_mode}] dT={dT:.0f}mm/f "
                      f"camZ={tnow[2]:.3f}m\n"
                      f"{diag}  pubZ=({zx:.2f},{zy:.2f},{zz:.2f}) "
                      f"{'OK-up' if zz > 0.9 else 'CHECK-Z'}")
            if (time.time() - getattr(self, '_last_trk_log', 0.0)) > 1.0:
                self.get_logger().warning(
                    f"[track] {status.replace(chr(10), '  ')}")
                self._last_trk_log = time.time()
            self._draw_and_publish(color, tracked_pose, T_cam_pub, status)
            return

        # Vision failed or poor -> OCCLUDED handling.
        if self.state == TRACKING:
            self.get_logger().info(f"TRACKING -> OCCLUDED (vis={vis_ratio:.2f})")
            self.state = OCCLUDED
            self._occ_start = wall
            self._relock_ok = 0

        # Re-lock check (need stable good vision).
        if tracked_pose is not None and vis_ratio >= self.a.relock_vis_thresh:
            self._relock_ok += 1
            if self._relock_ok >= self.a.relock_stable_frames:
                self.cam_pose = tracked_pose
                self.state = TRACKING
                self._occ_start = None
                T_world_obj = T_world_cam @ tracked_pose
                self.hand_offset = None          # force fresh offset
                self._solid_vis = 0
                self._maybe_cache_offset(T_world_obj, vis_ratio)
                self._publish(T_world_obj, stamp, wall)
                T_cam_pub = invert_T(T_world_cam) @ self.last_world_aligned
                self._draw_and_publish(color, tracked_pose, T_cam_pub, "RE-LOCKED")
                self.get_logger().info("OCCLUDED -> TRACKING (re-locked)")
                return
        else:
            self._relock_ok = 0

        # Kinematic hand-attach while occluded.
        if self.hand_offset is not None:
            T_world_hand = lookup_T(self.tf_buffer, BASE_FRAME, self.a.hand_frame, None)
            if T_world_hand is not None:
                T_world_obj = T_world_hand @ self.hand_offset
                self._publish(T_world_obj, stamp, wall)
                self._publish_overlay(
                    color, f"OCCLUDED hand-attach (det={getattr(self,'_anchor_src','?')})\n"
                           f"holding object rigid to {self.a.hand_frame}")
                return

        # No offset: brief hold of the last good pose, else stale.
        if self.last_world_aligned is not None and self._brief_hold < self.a.brief_hold_frames:
            self._brief_hold += 1
            self._republish(self.last_world_aligned, stamp, wall)
            self._publish_overlay(
                color, f"OCCLUDED hold {self._brief_hold}/{self.a.brief_hold_frames} "
                       f"(det={getattr(self,'_anchor_src','?')})")
            return

        if self._occ_start is not None and (wall - self._occ_start) > self.a.occluded_max_sec:
            self.get_logger().warn("OCCLUDED too long, no hand offset -> SEARCHING")
            self.state = SEARCHING
            self._auto_tries = 0
        self._publish_overlay(
            color, f"OCCLUDED lost - no hand offset yet (det={getattr(self,'_anchor_src','?')})\n"
                   f"detector cannot see object & grasp not yet cached")

    # ---- Search / register ----
    def _auto_propose(self, color, depth):
        """Run the auto detectors; return (mask|None, source_name).

        Wrapped in PerfTimer.stage('det') — captures the full detector pass
        including its internal SAM2 call (so the 1 Hz log shows total detection
        cost). The standalone SAM2@pose path in _reanchor is timed separately
        under 'sam'."""
        with self.perf.stage('det'):
            for det in self.detectors:
                self._last_detector = det.__class__.__name__
                m = det.detect(color, depth, self.cam_K, self.get_logger())
                if m is not None and int((m > 0).sum()) >= self.a.min_obj_pixels:
                    ys2, xs2 = np.where(m > 0)
                    self.get_logger().info(
                        f"[detect] {det.__class__.__name__} candidate: {len(xs2)}px "
                        f"bbox=({xs2.min()},{ys2.min()},{xs2.max()},{ys2.max()}) "
                        f"centroid=({xs2.mean():.0f},{ys2.mean():.0f})")
                    return m, det.__class__.__name__
            return None, None

    def _confirm_or_select(self, color, depth, proposed, src):
        """Show the auto proposal; user accepts it, clicks the correct object to
        override, redoes, or cancels. Returns ('accept',mask)|('redo',None)|
        ('cancel',None)."""
        win = ("CONFIRM target  [Enter/Space=accept]  [click=pick correct]  "
               "[r=redo auto]  [q=cancel]")
        st = {'click': None, 'mask': proposed, 'src': src or 'none', 'seed': None}

        def _on_mouse(ev, x, y, flags, param):
            if ev == cv2.EVENT_LBUTTONDOWN:
                st['click'] = (int(x), int(y))

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, _on_mouse)
        result = ('cancel', None)
        while True:
            if st['click'] is not None:
                pt = st['click']
                st['click'] = None
                m = self.sam_helper.from_point(color, pt, self.get_logger())
                if m is not None and self.a.mask_depth_band > 0:
                    m = self._refine_mask_by_depth(m, depth, seed=pt)
                rej = self._reject_oversize_mask(m, color, where=f'click@{pt}')
                if m is not None and int((m > 0).sum()) >= self.a.min_obj_pixels \
                        and not rej:
                    st['mask'] = m
                    st['src'] = f'click@{pt}'
                    st['seed'] = pt
                    self.get_logger().info(
                        f"[confirm] SAM2 click override at {pt} "
                        f"({int((m>0).sum())}px"
                        f"{', depth-refined' if self.a.mask_depth_band>0 else ''})")
                elif rej:
                    # message already logged by _reject_oversize_mask; keep the
                    # previous proposal as st['mask'] so the user can re-click
                    pass
                else:
                    self.get_logger().warn(f"[confirm] SAM2 gave no usable "
                                           f"mask at {pt}")
            disp = color[..., ::-1].copy()
            if st['mask'] is not None:
                cnts, _ = cv2.findContours(st['mask'], cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(disp, cnts, -1, (0, 255, 0), 2)
                head = f"proposed by {st['src']} - Enter=accept / click to override"
            else:
                head = "NO auto proposal - click the object to select it"
            cv2.putText(disp, head, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(disp, "[Enter/Space]=accept  [click]=override  "
                        "[r]=redo auto  [q]=cancel", (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, disp)
            k = cv2.waitKey(20) & 0xFF
            if k in (13, 32, ord('c')) and st['mask'] is not None:
                self._reg_seed = st['seed']
                result = ('accept', st['mask'])
                break
            if k == ord('r'):
                result = ('redo', None)
                break
            if k in (ord('q'), 27):
                result = ('cancel', None)
                break
        cv2.destroyWindow(win)
        return result

    def _refine_mask_by_depth(self, mask, depth, seed=None):
        """Clip the mask to pixels whose depth is within --mask_depth_band of the
        object's depth (seed pixel if given, else mask median). Removes background
        / table leakage so FoundationPose.guess_translation (median mask depth,
        estimater.py:149) uses the true object depth -> correct scale & pose.
        Disabled by default (--mask_depth_band 0): depth here is too sparse and
        shredded the SAM2 mask; we register on the raw SAM2 mask instead."""
        if self.a.mask_depth_band <= 0:
            return mask
        valid = (mask > 0) & (depth > 0)
        nv = int(valid.sum())
        if nv < 4:
            return mask
        z0 = 0.0
        if seed is not None:
            sx, sy = int(seed[0]), int(seed[1])
            H, W = depth.shape[:2]
            if 0 <= sy < H and 0 <= sx < W:
                win = depth[max(0, sy-3):sy+4, max(0, sx-3):sx+4]
                win = win[win > 0]
                if win.size:
                    z0 = float(np.median(win))
        if z0 <= 0:
            z0 = float(np.median(depth[valid]))
        band = self.a.mask_depth_band
        keep = valid & (np.abs(depth - z0) <= band)
        nk = int(keep.sum())
        self.get_logger().info(
            f"[mask-refine] z0={z0:.3f}m band=±{band:.3f}m  {nv}px -> {nk}px "
            f"(seed={'yes' if seed is not None else 'median'})")
        if nk >= max(4, self.a.min_obj_pixels // 2):
            return (keep.astype(np.uint8) * 255)
        self.get_logger().warn("[mask-refine] too few px after clip; keep raw mask")
        return mask

    def _do_search(self, color, depth):
        mode = self.a.detect_mode
        self._reg_seed = None
        if mode == 'click':
            mask = self.click_detector.detect(color, depth, self.cam_K,
                                              self.get_logger())
        elif mode == 'auto':
            mask, src = self._auto_propose(color, depth)
            if mask is not None:
                self.get_logger().warning(f"[detect] auto-accept via {src} "
                                          f"(--detect_mode auto, no confirm)")
        else:  # auto_confirm (default)
            proposed, src = self._auto_propose(color, depth)
            res, mask = self._confirm_or_select(color, depth, proposed, src)
            if res != 'accept':
                self.get_logger().info(f"[confirm] {res} -> re-search next frame")
                return

        if mask is None:
            return
        # Depth-refine the registration mask (click already refined in confirm;
        # this also tightens auto/click-mode masks). Root-cause fix for issue 2.
        mask = self._refine_mask_by_depth(mask, depth, seed=self._reg_seed)
        # Hard upper-bound: an image-wide mask blows up the register render batch
        # (252 hypotheses x giant crop) and OOMs the GPU.
        if self._reject_oversize_mask(mask, color, where='search'):
            return
        valid = int(((depth > 0) & (mask > 0)).sum())
        if valid < 4:
            self.get_logger().warn(f"mask has only {valid} valid-depth px; retrying")
            return
        zmed = float(np.median(depth[(depth > 0) & (mask > 0)]))
        pose = self._safe_register(self.cam_K, color, depth, mask,
                                   self.a.est_refine_iter, where='search')
        if pose is None:
            return
        self.cam_pose = pose @ invert_T(self.to_origin)
        t = self.cam_pose[:3, 3]
        # Cache initial mask size as the sanity reference for later re-anchors:
        # a YOLO-World mask >4x or <0.25x this is almost certainly a different
        # object (or background), so reject it even if it gates through.
        self._anchor_px_initial = int((mask > 0).sum())
        self.get_logger().warning(
            f"[register OK] mask={valid}px mask_med_depth={zmed:.3f}m -> "
            f"obj cam_xyz=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f}) "
            f"|t|={np.linalg.norm(t):.3f}m  (obj diam {self.obj_diameter:.3f}m). "
            f"If cam_z >> mask_med_depth the mask still leaks background.")
        # Fresh lock -> recalibrate the stabilizers and hybrid counters.
        self._yaw_lock = None
        self._pos_ema = None
        self._frame_n = 0
        self._reacq_fail = 0
        self.state = TRACKING

    def _obj_depth_px(self, depth):
        """(#valid-depth px, median depth) inside the prior pose's projected
        bbox. The boundary measurement that proves whether the frozen box is
        depth starvation: this is exactly the signal track_one's refiner needs
        and the one number the previous logs never reported."""
        if getattr(self, 'cam_pose', None) is None:
            return (0, 0.0)
        pc = bbox_corners(self.bbox)
        ph = (self.cam_pose @ np.hstack([pc, np.ones((8, 1))]).T).T[:, :3]
        if np.any(ph[:, 2] <= 0):
            return (0, 0.0)
        uv = (self.cam_K @ (ph / ph[:, 2:3]).T).T[:, :2]
        H, W = depth.shape[:2]
        x1, y1 = np.floor(uv.min(0)).astype(int)
        x2, y2 = np.ceil(uv.max(0)).astype(int)
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return (0, 0.0)
        roi = depth[y1:y2, x1:x2]
        m = np.isfinite(roi) & (roi > 0)
        n = int(m.sum())
        return (n, float(np.median(roi[m])) if n else 0.0)

    # ---- Track one frame ----
    def _autocast_ctx(self):
        """torch.autocast in bf16 ONLY when --precision=bf16 and the gate hasn't
        tripped. TF32 needs no per-call wrap (global flags set in __init__)."""
        if (self._precision == 'bf16' and torch.cuda.is_available()
                and not self._precision_locked):
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        # null context manager
        class _Noop:
            def __enter__(self_): return None
            def __exit__(self_, *a): return False
        return _Noop()

    def _maybe_revert_bf16(self, exc, where=''):
        """If the exception is the 'BFloat16 not supported by linalg.inv' one,
        lock precision back to fp32 immediately. Returns True iff we reverted —
        the caller should retry once."""
        msg = str(exc)
        bad = ('BFloat16' in msg or 'Low precision dtypes' in msg
               or 'bf16' in msg.lower())
        if bad and self._precision == 'bf16' and not self._precision_locked:
            self._precision_locked = True
            self.get_logger().warn(
                f"[Precision][{where}] bf16 incompatible with stock "
                f"FoundationPose linalg.inv -> reverting to fp32 for the rest "
                f"of the run. Retrying once now.")
            return True
        return False

    def _adaptive_iter(self):
        """Return the track_refine_iter to use this frame (C2).

        Starts at --track_refine_iter_min; if the previous frame escalated, we
        ride the boost until --track_refine_iter_boost_frames elapses; otherwise
        the steady-state value is _iter_now (decayed). Escalation itself is set
        AFTER the frame in _track_step based on IoU/dT signals."""
        if not self.a.adaptive_iter:
            return self.a.track_refine_iter_max
        if self._iter_boost_left > 0:
            self._iter_boost_left -= 1
            return self.a.track_refine_iter_max
        return self._iter_now

    def _track_step(self, color, depth):
        # ROOT-CAUSE FIX: measure object-region depth density, then densify
        # before track_one so stock erode_depth + refiner still have geometry
        # (see fill_depth_holes / error_status.mp4 analysis).
        self._obj_px_raw = self._obj_depth_px(depth)
        if self.a.depth_fill:
            depth = fill_depth_holes(depth, self.a.depth_fill_max_px)
        self._obj_px_fill = self._obj_depth_px(depth)

        it = self._adaptive_iter()
        try:
            with self.perf.stage('refine'), self._autocast_ctx():
                pose = self.pose_est.track_one(
                    rgb=color, depth=depth, K=self.cam_K, iteration=it)
        except Exception as e:
            # Self-healing: bf16 + linalg.inv is incompatible with stock
            # FoundationPose; if that fires we revert to fp32 and retry once.
            if self._maybe_revert_bf16(e, where='track_one'):
                try:
                    with self.perf.stage('refine'):
                        pose = self.pose_est.track_one(
                            rgb=color, depth=depth, K=self.cam_K, iteration=it)
                except Exception as e2:
                    now = time.time()
                    if now - getattr(self, '_last_trkerr_log', 0.0) > 1.0:
                        self.get_logger().warn(f"[track_one] retry raised: {e2}")
                        self._last_trkerr_log = now
                    self._iter_boost_left = self.a.track_refine_iter_boost_frames
                    return None, 0.0
            else:
                now = time.time()
                if now - getattr(self, '_last_trkerr_log', 0.0) > 1.0:
                    self.get_logger().warn(f"[track_one] raised: {e}")
                    self._last_trkerr_log = now
                self._iter_boost_left = self.a.track_refine_iter_boost_frames
                return None, 0.0
        center = pose @ invert_T(self.to_origin)

        # ---- Adaptive iter feedback (C2) + IoU quality signal (B2/C1 gate) ----
        # Render the silhouette at the new pose, score against the SAM mask if
        # we have one (re-anchor frames), else against the cluster of high-depth
        # pixels inside the projected bbox as a rough surrogate. Both are
        # accumulated in _iou_window for the bf16 quality gate; the per-frame
        # IoU decides escalation/decay.
        iou_now = 0.0
        ref_mask = getattr(self, '_last_mask', None)
        if ref_mask is not None and ref_mask.shape[:2] == color.shape[:2]:
            iou_now = self.qscorer.score(ref_mask, center, self.cam_K)
        self._iou_last = iou_now
        if iou_now > 0.0:
            self._iou_window.append(iou_now)
            if len(self._iou_window) > self._iou_window_max:
                self._iou_window.pop(0)

        # dT vs previous successful track
        prev_t = self._last_track_t
        dT_m = (float(np.linalg.norm(center[:3, 3] - prev_t))
                if prev_t is not None else 0.0)
        if self.a.adaptive_iter:
            low_quality = (iou_now > 0.0 and iou_now < self.a.adaptive_iou_lo)
            big_jump = (dT_m > self.a.adaptive_dt_hi)
            if low_quality or big_jump:
                self._iter_boost_left = self.a.track_refine_iter_boost_frames
                self._iter_now = self.a.track_refine_iter_max
            elif self._iter_boost_left == 0 and self._iter_now > self.a.track_refine_iter_min:
                # Decay one step at a time so we don't ping-pong.
                self._iter_now = max(self.a.track_refine_iter_min, self._iter_now - 1)

        return center, self._visibility(center, depth)

    def _hybrid_step(self, color, depth):
        """Default fluent+robust estimator (fp.py-proven pattern, viable now the
        camera is close ~0.6 m so depth is dense):

          - track_one EVERY frame: smooth, fast, temporally continuous (uses
            pose_last) -> the axes do NOT flicker (register() re-guesses
            orientation each frame on a symmetric cube -> the flicker seen in
            error_status.mp4).
          - SAM2 re-anchor on track loss, when no prior lock, or every
            --reanchor_period frames to correct drift. register() updates
            pose_est.pose_last, so track_one resumes from the re-anchored pose.

        The old one-way gate deadlock is gone: recovery uses SAM2-at-the-
        predicted-point (locked-on, no gate) and a hard re-acquire escape after
        --reacquire_after failures.

        Drift correction (the error_status.mp4 #2/#3 bug — track_one keeps
        returning a pose-on-thin-air while the cube has moved): every
        --reanchor_period frames we run a YOLO-World **primary** re-anchor with
        the spatial gate DISABLED, so a correct YOLO-World detection far from
        the (drifted) track can snap the lock back to the cube. SAM2@pose is
        kept only for the track_one-failure fast path — using it on drift would
        just confirm the wrong pose's pixel."""
        self._frame_n += 1
        self._last_mask = None
        pose, vis = self._track_step(color, depth)
        track_lost = pose is None
        no_prior = self.cam_pose is None
        is_periodic = (not track_lost and not no_prior
                       and self._frame_n % max(1, self.a.reanchor_period) == 0)
        need = track_lost or no_prior or is_periodic
        if need:
            rp, rv = self._reanchor(
                color, depth,
                force_yolo=(no_prior or is_periodic),
                disable_gate=is_periodic)
            if rp is not None:
                self._reacq_fail = 0
                self._last_src = getattr(self, '_anchor_src', 'reanchor')
                return rp, rv
            if track_lost:
                self._reacq_fail += 1
        if pose is not None:
            self._reacq_fail = 0
            self._last_src = 'track_one'
        return pose, vis

    def _mask_size_ok(self, m, lo=0.25, hi=4.0):
        """True iff the candidate mask px count is within [lo,hi] x the initial
        registered mask px count. If we have no baseline (no first register
        yet), accept anything that passes min_obj_pixels."""
        init = getattr(self, '_anchor_px_initial', None)
        if not init or init <= 0:
            return True
        npx = int((m > 0).sum())
        ratio = npx / float(init)
        return lo <= ratio <= hi

    def _mask_size_label(self, src, m):
        init = getattr(self, '_anchor_px_initial', None) or 1
        ratio = int((m > 0).sum()) / float(init)
        return f'{src}!size{ratio:.2f}x'

    def _reject_oversize_mask(self, m, color, where=''):
        """True iff the mask covers more than --max_mask_frac of the image.
        A 502k-px / 55% mask (the user clicking the table by mistake) drives
        FoundationPose's 252-hypothesis render batch huge -> CUDA OOM."""
        if m is None:
            return False
        H, W = color.shape[:2]
        imgpx = max(1, H * W)
        npx = int((m > 0).sum())
        frac = npx / float(imgpx)
        if frac > self.a.max_mask_frac:
            self.get_logger().warn(
                f"[{where}] mask {npx}px ({100 * frac:.1f}% of image) > "
                f"--max_mask_frac {100 * self.a.max_mask_frac:.0f}% — "
                f"reject (would OOM register). Click the OBJECT, not the table.")
            return True
        return False

    def _safe_register(self, K, rgb, depth, mask, iteration, where=''):
        """Wrap pose_est.register with empty_cache before + an OOM-recovery
        retry. The OOM in this branch came from a giant click mask; the size
        cap above is the primary defense, this is the belt-and-suspenders.

        Also: BF16 autocast (C1) is applied here for the register path."""
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            with self.perf.stage('refine'), self._autocast_ctx():
                return self.pose_est.register(
                    K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=iteration)
        except torch.cuda.OutOfMemoryError as e:
            self.get_logger().warn(
                f"[{where}] register OOM ({e}); clearing cache, retrying "
                f"iter={max(1, iteration - 1)}")
            try:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                with self.perf.stage('refine'), self._autocast_ctx():
                    return self.pose_est.register(
                        K=K, rgb=rgb, depth=depth, ob_mask=mask,
                        iteration=max(1, iteration - 1))
            except Exception as e2:
                self.get_logger().warn(f"[{where}] register retry failed: {e2}")
                return None
        except Exception as e:
            # Self-healing: bf16 + linalg.inv is incompatible with stock
            # FoundationPose. If that's the exception, revert to fp32 and retry.
            if self._maybe_revert_bf16(e, where=where):
                try:
                    with self.perf.stage('refine'):
                        return self.pose_est.register(
                            K=K, rgb=rgb, depth=depth, ob_mask=mask,
                            iteration=iteration)
                except Exception as e2:
                    self.get_logger().warn(
                        f"[{where}] register fp32-retry failed: {e2}")
                    return None
            now = time.time()
            if now - getattr(self, '_last_trkerr_log', 0.0) > 1.0:
                self.get_logger().warn(f"[{where}] register raised: {e}")
                self._last_trkerr_log = now
            return None

    def _reanchor(self, color, depth, force_yolo, disable_gate=False):
        """Re-anchor the pose via SAM2->register(). Two paths:

        * Track-loss recovery (force_yolo=False): SAM2-from-point at the
          predicted object pixel. Fast, locked-on, only meaningful while the
          track is still on the object — useless once it has drifted off, but
          the YOLO-World fallback below catches that case.
        * Periodic drift check / no-prior lock (force_yolo=True): skip
          SAM2@pose entirely and run YOLO-World -> SAM2. With
          disable_gate=True (periodic check) the spatial gate is OFF so a
          correct detection far from the drifted pose can pull the lock back.
          With disable_gate=False (no-prior) the gate is applied until
          --reacquire_after failures bypass it.

        Both YOLO-World paths sanity-check the candidate mask against the
        initial registered mask size: <0.25x or >4x the initial px count is
        rejected (different object / background leak)."""
        mask, src = None, None
        if not force_yolo and self.cam_pose is not None:
            pc = project_center(self.cam_pose, self.cam_K)
            if pc is not None:
                with self.perf.stage('sam'):
                    # Pass prior pose + scorer so SAM2 multi-mask is picked by
                    # IoU vs rendered silhouette (B2 full).
                    m = self.sam_helper.from_point(
                        color, (int(pc[0]), int(pc[1])), self.get_logger(),
                        prior_pose=self.cam_pose, K=self.cam_K,
                        scorer=self.qscorer)
                if m is not None and int((m > 0).sum()) >= self.a.min_obj_pixels:
                    # Size sanity: SAM2 will happily return ANY blob at the
                    # click point (table, robot, ...) — reject if it's wildly
                    # different from the registered object.
                    if self._mask_size_ok(m):
                        mask, src = m, 'SAM2@pose'
                    else:
                        self._anchor_src = self._mask_size_label('SAM2@pose', m)
        if mask is None:
            m2, s2 = self._auto_propose(color, depth)
            if m2 is not None:
                ok = self._mask_size_ok(m2)
                if not ok:
                    self._anchor_src = self._mask_size_label(s2 or 'yolo', m2)
                if (ok and not disable_gate
                        and self.cam_pose is not None
                        and self._reacq_fail < self.a.reacquire_after):
                    pc = project_center(self.cam_pose, self.cam_K)
                    ys, xs = np.where(m2 > 0)
                    if pc is not None and xs.size:
                        d = math.hypot(xs.mean() - pc[0], ys.mean() - pc[1])
                        if d > self.a.reanchor_gate_px:
                            ok = False
                            self._anchor_src = f'{s2}!gate{d:.0f}'
                if ok:
                    mask, src = m2, (s2 or 'yolo')
        if mask is None:
            return None, 0.0
        self._anchor_src = src
        self._anchor_px = int((mask > 0).sum())
        self._last_mask = mask
        if int(((depth > 0) & (mask > 0)).sum()) < 4:
            return None, 0.0
        # Defensive size cap (the periodic re-anchor mask passed _mask_size_ok,
        # but a missing _anchor_px_initial baseline could let a huge mask
        # through). Same OOM defense as in _do_search.
        if self._reject_oversize_mask(mask, color, where='reanchor'):
            return None, 0.0
        rp = self._safe_register(self.cam_K, color, depth, mask,
                                 self.a.reanchor_refine_iter, where='reanchor')
        if rp is None:
            return None, 0.0
        center = rp @ invert_T(self.to_origin)
        return center, self._visibility(center, depth)

    def _reregister_step(self, color, depth):
        """ROBUST steady-state estimator (default --track_mode reregister).

        Re-detect (YOLO-World->SAM2, training-free) and re-register() EVERY
        frame. register() derives translation from the SAM2 mask centroid +
        median mask depth (estimater.py guess_translation), so it follows a
        textureless cube top-down where track_one's refiner is degenerate and
        froze the box (error_status.mp4). This is the self-healing mechanism
        that made the old fp.py robust (periodic SAM2 re-anchor, fp.py:656-796),
        minus the per-object-trained YOLO.

        fp.py-style spatial gate keeps the lock on the confirmed object: a fresh
        mask whose centroid jumps >--reanchor_gate_px from the last pose
        projection is rejected (returns None -> existing OCCLUDED hand-attach /
        brief-hold path), so a stray YOLO-World box on the robot/another red
        blob can't hijack the track."""
        mask, src = self._auto_propose(color, depth)
        if mask is None:
            self._anchor_src, self._anchor_px, self._anchor_gate_d = 'none', 0, -1.0
            return None, 0.0
        ys, xs = np.where(mask > 0)
        npx = int(xs.size)
        mcx, mcy = float(xs.mean()), float(ys.mean())
        self._reanchor_n = getattr(self, '_reanchor_n', 0) + 1
        gate_d = -1.0
        if self.cam_pose is not None and self._reanchor_n > self.a.reanchor_warmup:
            pc = project_center(self.cam_pose, self.cam_K)
            if pc is not None:
                gate_d = math.hypot(mcx - pc[0], mcy - pc[1])
                if gate_d > self.a.reanchor_gate_px:
                    self._anchor_src = f'{src}!GATE{gate_d:.0f}px'
                    self._anchor_px, self._anchor_gate_d = npx, gate_d
                    return None, 0.0
        self._anchor_src = src or '?'
        self._anchor_px, self._anchor_gate_d = npx, gate_d
        self._last_mask = mask

        if self.a.mask_depth_band > 0:
            mask = self._refine_mask_by_depth(mask, depth)
        if int(((depth > 0) & (mask > 0)).sum()) < 4:
            return None, 0.0
        try:
            with self.perf.stage('refine'), self._autocast_ctx():
                pose = self.pose_est.register(
                    K=self.cam_K, rgb=color, depth=depth, ob_mask=mask,
                    iteration=self.a.est_refine_iter)
        except Exception as e:
            if self._maybe_revert_bf16(e, where='reregister'):
                try:
                    with self.perf.stage('refine'):
                        pose = self.pose_est.register(
                            K=self.cam_K, rgb=color, depth=depth, ob_mask=mask,
                            iteration=self.a.est_refine_iter)
                except Exception as e2:
                    self.get_logger().warn(f"[reregister] fp32-retry failed: {e2}")
                    return None, 0.0
            else:
                now = time.time()
                if now - getattr(self, '_last_trkerr_log', 0.0) > 1.0:
                    self.get_logger().warn(f"[reregister] register raised: {e}")
                    self._last_trkerr_log = now
                return None, 0.0
        center = pose @ invert_T(self.to_origin)
        return center, self._visibility(center, depth)

    def _visibility(self, T_cam_obj, depth):
        """Depth-consistency ratio inside the projected object bbox."""
        pc = bbox_corners(self.bbox)
        pc_h = (T_cam_obj @ np.hstack([pc, np.ones((8, 1))]).T).T[:, :3]
        if np.any(pc_h[:, 2] <= 0):
            return 0.0
        uv = (self.cam_K @ (pc_h / pc_h[:, 2:3]).T).T[:, :2]
        x1, y1 = np.floor(uv.min(0)).astype(int)
        x2, y2 = np.ceil(uv.max(0)).astype(int)
        H, W = depth.shape[:2]
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        roi = depth[y1:y2, x1:x2]
        z_exp = float(T_cam_obj[2, 3])
        good = np.isfinite(roi) & (roi > 0) & (np.abs(roi - z_exp) < self.a.vis_depth_eps)
        n_good = int(good.sum())
        if n_good < self.a.min_obj_pixels:
            return 0.0
        return n_good / float(roi.size)

    # ---- Hand-attach offset ----
    def _maybe_cache_offset(self, T_world_obj_raw, vis_ratio):
        # When the depth-visibility gate is off (default), trust consecutive
        # successful track_one frames instead of the unreliable depth ratio.
        gate = self.a.relock_vis_thresh if self.a.require_vis else -1.0
        if vis_ratio < gate:
            self._solid_vis = 0
            return
        self._solid_vis += 1
        if self._solid_vis < self.a.offset_avg_frames:
            return
        T_world_hand = lookup_T(self.tf_buffer, BASE_FRAME, self.a.hand_frame, None)
        if T_world_hand is None:
            return
        self.hand_offset = invert_T(T_world_hand) @ T_world_obj_raw  # overwrite (no SE3 blend)

    # ---- Publishing ----
    def _publish(self, T_world_obj, stamp, wall):
        if self.a.gravity_align:
            T_aligned, yaw = gravity_align_yaw(
                T_world_obj, self.a.yaw_axis, self.last_yaw, self.a.yaw_ema)
            # yaw_mode=lock: freeze yaw at the first confirmed register. A
            # symmetric cube's yaw is arbitrary; this kills the per-frame axis
            # flicker and gives BODex one steady value (Z stays forced up).
            if self.a.yaw_mode == 'lock':
                if self._yaw_lock is None:
                    self._yaw_lock = yaw
                yaw = self._yaw_lock
                T_aligned = T_aligned.copy()
                T_aligned[:3, :3] = R.from_euler('z', yaw).as_matrix()
            self.last_yaw = yaw
        else:
            T_aligned = T_world_obj
        # Light position EMA -> smooth values for the BODex 7-sample median
        # latch (no lag concern on the slow pre-grasp pick).
        if self.a.pos_ema > 0.0 and self._pos_ema is not None:
            T_aligned = T_aligned.copy()
            T_aligned[:3, 3] = (self.a.pos_ema * self._pos_ema
                                + (1.0 - self.a.pos_ema) * T_aligned[:3, 3])
        self._pos_ema = T_aligned[:3, 3].copy()
        self.last_world_aligned = T_aligned
        self._republish(T_aligned, stamp, wall)

    def _republish(self, T_aligned, stamp, wall):
        with self.perf.stage('pub'):
            use_stamp = stamp if (self.a.stamp_mode == 'sensor' and stamp is not None
                                  and (stamp.sec or stamp.nanosec)) else self.get_clock().now().to_msg()
            pos = T_aligned[:3, 3]
            q = R.from_matrix(T_aligned[:3, :3]).as_quat()  # x,y,z,w

            m = PoseStamped()
            m.header.stamp = use_stamp
            m.header.frame_id = BASE_FRAME
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, pos)
            m.pose.orientation.x, m.pose.orientation.y = float(q[0]), float(q[1])
            m.pose.orientation.z, m.pose.orientation.w = float(q[2]), float(q[3])
            self.pose_pub.publish(m)

            tf = TransformStamped()
            tf.header.stamp = use_stamp
            tf.header.frame_id = BASE_FRAME
            tf.child_frame_id = self.obj_frame
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, pos)
            tf.transform.rotation.x, tf.transform.rotation.y = float(q[0]), float(q[1])
            tf.transform.rotation.z, tf.transform.rotation.w = float(q[2]), float(q[3])
            self.tf_broadcaster.sendTransform(tf)

        self._telemetry(wall)

    def _telemetry(self, wall):
        now = time.time()
        self._pub_times.append(now)
        if now - self._last_tele >= 1.0:
            self._pub_times = [t for t in self._pub_times if now - t <= 1.0]
            hz = len(self._pub_times)
            lat = (now - wall) * 1000.0 if wall else float('nan')
            # Rolling IoU + adaptive iter state (Tier B+C signal)
            iou_mean = (sum(self._iou_window) / len(self._iou_window)
                        if self._iou_window else 0.0)
            iter_state = (f"iter={self._iter_now}"
                          f"{'+boost' if self._iter_boost_left>0 else ''} "
                          f"max={self.a.track_refine_iter_max}")
            self.get_logger().info(
                f"[telemetry] {self.state} ~{hz} Hz, latency {lat:.0f} ms, "
                f"hand_offset={'yes' if self.hand_offset is not None else 'no'} "
                f"prec={self._precision}"
                f"{'(LOCKED-fp32)' if self._precision_locked else ''}  "
                f"iou30={iou_mean:.2f}(N={len(self._iou_window)})  "
                f"{iter_state}")
            if self.a.prof_log and self.perf.window_due():
                self.get_logger().info(self.perf.format_line())
            # ---- BF16 quality gate (C1) ----
            # Simple, conservative absolute-floor revert: if we asked for bf16
            # and the rolling IoU mean over a full sample window stays below
            # --precision_iou_drop_max, lock to fp32 for the rest of the run.
            # (We don't capture an fp32 baseline to compare against — that
            # would require running fp32 for N frames at startup which adds
            # latency to the first lock. Pass --precision fp32 explicitly to
            # disable bf16 from the start.)
            if (self._precision == 'bf16' and not self._precision_locked
                    and len(self._iou_window) >= self._iou_window_max
                    and iou_mean < self.a.precision_iou_drop_max):
                self._precision_locked = True
                self.get_logger().warn(
                    f"[Precision] bf16 quality gate tripped: iou30={iou_mean:.2f} "
                    f"< {self.a.precision_iou_drop_max} -> falling back to fp32 "
                    f"for the rest of the run.")
            self._last_tele = now

    # ---- Detection debug overlay (what each detector produced) ----
    def _draw_search_debug(self, color):
        vis = color.copy()
        for det in self.detectors:
            info = getattr(det, 'last_info', None)
            if info and info.get('bbox'):
                x1, y1, x2, y2 = info['bbox']
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cx, cy = info['centroid']
                cv2.circle(vis, (int(cx), int(cy)), 5, (0, 255, 255), -1)
                cv2.putText(vis, f"DepthCluster {info['area']}px "
                            f"({info.get('ncand','?')} cand)",
                            (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 2, cv2.LINE_AA)
            for d in getattr(det, 'last_dets', []) or []:
                # New layout (B1): (x1,y1,x2,y2,label,conf,score,iou_b) — 8.
                # Stay tolerant of the old 6-tuple format for safety.
                if len(d) >= 8:
                    x1, y1, x2, y2, lb, cf, sc, iou_b = d
                    label = (f"{lb}:c={cf:.2f} iou={iou_b:.2f}"
                             if iou_b > 0 else f"{lb}:c={cf:.2f}")
                else:
                    x1, y1, x2, y2, lb, cf = d[:6]
                    label = f"{lb}:{cf:.2f}"
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(vis, label, (x1, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
        self._publish_overlay(
            vis, f"SEARCHING mode={self.a.detect_mode} last={self._last_detector}  "
                 f"cyan=DepthCluster  magenta=YOLO-World")

    # ---- Visualization (raw camera-frame pose, like fp.py:822-834) ----
    def _draw_and_publish(self, color, T_cam_raw, T_cam_pub, status):
        with self.perf.stage('render'):
            return self._draw_and_publish_impl(color, T_cam_raw, T_cam_pub, status)

    def _draw_and_publish_impl(self, color, T_cam_raw, T_cam_pub, status):
        try:
            vis = draw_posed_3d_box(self.cam_K, img=color, ob_in_cam=T_cam_raw,
                                    bbox=self.bbox)
            # thin = raw FoundationPose pose; thick = gravity-aligned PUBLISHED
            # pose (the BODex output). Z (blue) of the thick axes must point up.
            vis = draw_xyz_axis(vis, ob_in_cam=T_cam_raw, scale=0.07, K=self.cam_K,
                                thickness=2, transparency=0, is_input_rgb=True)
            vis = draw_xyz_axis(vis, ob_in_cam=T_cam_pub, scale=0.12, K=self.cam_K,
                                thickness=4, transparency=0, is_input_rgb=True)
            # show the SAM2 lock that drove register() (reregister: every frame;
            # hybrid: only on re-anchor frames — _last_mask is cleared otherwise).
            m = getattr(self, '_last_mask', None)
            if self.a.track_mode in ('reregister', 'hybrid') and m is not None:
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, (255, 128, 0), 2)
        except Exception:
            vis = color
        self._publish_overlay(vis, status)

    def _publish_overlay(self, img_rgb, status):
        vis = img_rgb.copy()
        # Multi-line: the cv2 window covers the terminal in screen recordings,
        # so all diagnostics must be legible ON the image.
        for i, line in enumerate(str(status).split('\n')):
            y = 28 + i * 30
            cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 5, cv2.LINE_AA)          # outline for contrast
            cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)
        if self.a.viz_local:
            cv2.imshow('foundation_pose', vis[..., ::-1])
            k = cv2.waitKey(1) & 0xFF
            # 'n' = drop track and re-open the SAM2 selector so the user can
            # click the object again (current track is wrong / object swapped).
            if k == ord('n') and not self._manual_redetect:
                self._manual_redetect = True
                self.get_logger().warn(
                    "[manual] 'n' pressed -> re-detection requested next frame")
            return
        bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        msg = numpy_to_imgmsg(bgr, 'bgr8', self.a.cam_frame, self.get_clock().now().to_msg())
        self.overlay_pub.publish(msg)
        self.overlay_pub_be.publish(msg)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _resolve_mesh(object_name):
    base = os.path.join("demo_data", object_name)
    cands = []
    for ext in ('obj', 'stl', 'STL'):
        cands += glob.glob(os.path.join(base, f"*.{ext}"))
    if not cands:
        # fall back to slug-named dir
        slug = re.sub(r'[^a-z0-9]+', '_', object_name.lower()).strip('_')
        for ext in ('obj', 'stl', 'STL'):
            cands += glob.glob(os.path.join("demo_data", slug, f"*.{ext}"))
    # prefer "<stem>.obj" exactly, else the first candidate
    for c in cands:
        if os.path.splitext(os.path.basename(c))[0].lower() == object_name.lower():
            return c
    return cands[0] if cands else None


def main():
    args = parse_args()
    mesh_path = _resolve_mesh(args.object)
    if mesh_path is None:
        print(f"[ERROR] No mesh found for '{args.object}' under demo_data/. "
              f"Expected demo_data/{args.object}/<name>.obj")
        return
    print(f"[INFO] Using mesh: {mesh_path}")

    rclpy.init()
    node = SingleObjectPoseNode(args, mesh_path)
    try:
        rclpy.spin(node)
    finally:
        node._stop = True
        try:
            node._proc_thread.join(timeout=1.0)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
