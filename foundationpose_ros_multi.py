# foundationpose_ros_multi.py

import sys
sys.path.append('./FoundationPose')
sys.path.append('./FoundationPose/nvdiffrast')

import rclpy
from rclpy.node import Node
from estimater import *   # FoundationPose, ScorePredictor, PoseRefinePredictor, draw_*
import cv2
import numpy as np
import trimesh
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge
import argparse
import os, time, glob, threading, queue, math
from scipy.spatial.transform import Rotation as R
from ultralytics import SAM, YOLO
from cam_2_base_transform import *  # transformation(pose_array)
import tkinter as tk
from tkinter import Listbox, END, Button
from tf2_ros import TransformBroadcaster, Buffer, TransformListener, TransformException

# -------------------------------
# 고정 파라미터(네 환경에 맞게 필요 시 조정)
# -------------------------------
BASE_FRAME = "world"                          # RViz Fixed Frame로 쓸 부모 프레임
CAM_FRAME  = "cam_front_color_optical_frame"      # CameraInfo.frame_id와 동일해야 함

# Top view
# CALIB_T = {
#     'x': -0.007763, 'y': 0.439065, 'z': 1.96248,
#     'qx': 0.0057113, 'qy': 0.9983262, 'qz': -0.0569960, 'qw': 0.0079736,
# }


# # Front camera pose (IsaacLab env axes: +Y forward, +X left, +Z up)
CALIB_T = {
    'x': 0.028505,
    'y': 0.854822,
    'z': 1.346442,
    'qx': 0.00702263,
    'qy': 0.95571907,
    'qz': -0.29234591,
    'qw': -0.02798306,
    # RPY (deg): roll = -145.9409°, pitch = -2.8314°, yaw = 178.2906°
}
# -------------------------------
# FoundationPose monkey patch
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

# -------------------------------
# 파일 순서 선택 GUI (수동 모드에서만 사용)
# -------------------------------
class FileSelectorGUI:
    def __init__(self, master, file_paths):
        self.master = master
        self.master.title("Library: Sequence Selector")
        self.file_paths = file_paths
        self.reordered_paths = None

        self.listbox = Listbox(master, selectmode="extended", width=50, height=10)
        self.listbox.pack()

        for fp in self.file_paths:
            self.listbox.insert(END, os.path.splitext(os.path.basename(fp))[0])

        self.up_button = Button(master, text="Move Up", command=self.move_up)
        self.up_button.pack(side="left", padx=5, pady=5)
        self.down_button = Button(master, text="Move Down", command=self.move_down)
        self.down_button.pack(side="left", padx=5, pady=5)
        self.done_button = Button(master, text="Done", command=self.done)
        self.done_button.pack(side="left", padx=5, pady=5)

    def move_up(self):
        selected = list(self.listbox.curselection())
        for i in selected:
            if i > 0:
                name = self.listbox.get(i)
                self.listbox.delete(i)
                self.listbox.insert(i-1, name)
                self.listbox.selection_set(i-1)

    def move_down(self):
        selected = list(self.listbox.curselection())
        for i in reversed(selected):
            if i < self.listbox.size() - 1:
                name = self.listbox.get(i)
                self.listbox.delete(i)
                self.listbox.insert(i+1, name)
                self.listbox.selection_set(i+1)

    def done(self):
        names = self.listbox.get(0, END)
        mp = {os.path.splitext(os.path.basename(f))[0]: f for f in self.file_paths}
        self.reordered_paths = [mp[n] for n in names]
        self.master.quit()

    def get_reordered_paths(self):
        return self.reordered_paths

def rearrange_files(file_paths):
    root = tk.Tk()
    app = FileSelectorGUI(root, file_paths)
    root.mainloop()
    return app.get_reordered_paths()

# -------------------------------
# CLI
# -------------------------------
parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))

parser.add_argument('--mode', choices=['auto','auto_debug','manual'], default='auto',
    help="auto(기본): 창 없이 자동 매칭 등록 / auto_debug: 2초 프리뷰 후 자동 / manual: 클릭 UI")
parser.add_argument('--detector_weights', type=str, default='yolo11m.pt',
    help="YOLOv8 weights or name (auto-download), e.g., yolov8sall.pt")
parser.add_argument('--det_conf', type=float, default=0.75, help='YOLO confidence threshold')

parser.add_argument('--viz_yolo', action='store_true',
    help='Overlay YOLO detection boxes on frames where YOLO ran')
parser.add_argument('--viz_yolo_font', type=float, default=0.6,
    help='YOLO label font scale (OpenCV)')

# 추적 튜닝
parser.add_argument('--track_refine_iter_base', type=int, default=2,
    help='기본 track refine 반복수')
parser.add_argument('--track_refine_iter_max', type=int, default=5,
    help='Δt가 클 때 늘릴 수 있는 최대 반복수')
parser.add_argument('--iter_gain_per_sec', type=float, default=1.5,
    help='Δt(초)당 추가 반복수(예: 1.5 → 0.5초 늘면 +0~1회)')

# YOLO 앵커 & 드리프트 / 재획득
parser.add_argument('--yolo_anchor_period', type=int, default=40,
    help='N프레임마다 YOLO 1회로 드리프트 점검')
parser.add_argument('--drift_px_thresh', type=float, default=40.0,
    help='프로젝션 중심과 YOLO 중심 픽셀 거리 임계치 → 넘으면 재획득 큐잉')
parser.add_argument('--lost_reacquire', action='store_true', default=True,
    help='LOST 시 백그라운드 재획득 실행 (기본 True)')

# FoundationPose iteration
parser.add_argument('--est_refine_iter', type=int, default=4, help='초기 등록 refine 반복수')

args = parser.parse_args()

# -------------------------------
# 유틸: optical frame 회전 행렬 구성
#  - optical:  x=right, y=down, z=forward
#  - env up=(0,0,1) 기준 look-at으로 맞춘 뒤 y를 'down'으로 설정
# -------------------------------
def _optical_cam_R_world(pos_xyz, focus_xy, look_down_deg):
    px, py, pz = pos_xyz
    fx, fy = focus_xy
    r = math.hypot(fx - px, fy - py)
    z_drop = math.tan(math.radians(look_down_deg)) * r
    tgt = np.array([fx, fy, pz - z_drop], dtype=np.float64)

    pos = np.array([px, py, pz], dtype=np.float64)
    fwd = tgt - pos
    n = np.linalg.norm(fwd)
    if n < 1e-8:
        fwd = np.array([0.0, -1.0, 0.0])
    else:
        fwd /= n

    upw = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(fwd, upw)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= rn

    # 중요: 우수계 보장 → right × down = fwd  ⇒  down = fwd × right
    down = np.cross(fwd, right)

    # camera→world 회전행렬 (열벡터: [x_right, y_down, z_forward])
    Rcw = np.column_stack([right, down, fwd])
    return Rcw

# -------------------------------
# ROS2 노드
# -------------------------------
class PoseEstimationNode(Node):
    def __init__(self, mesh_paths):
        super().__init__('pose_estimation_node')

        # 토픽 구독
        # self.image_sub = self.create_subscription(Image, '/top/cam1/color/image_raw', self.image_callback, 10)
        # self.depth_sub = self.create_subscription(Image, '/top/cam1/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        # self.info_sub  = self.create_subscription(CameraInfo, '/top/cam1/color/camera_info', self.camera_info_callback, 10)
        
        self.image_sub = self.create_subscription(Image, '/front/cam2/color/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/front/cam2/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.info_sub  = self.create_subscription(CameraInfo, '/front/cam2/color/camera_info', self.camera_info_callback, 10)

        self.bridge = CvBridge()
        self.cam_K = None

        # 최신 프레임 버퍼 & 워커
        self._frame_lock = threading.Lock()
        self._latest = {'color': None, 'depth': None, 'stamp': None}
        self._proc_busy = False
        self._stop = False
        self._proc_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._proc_thread.start()

        # 프리뷰(자동 디버그) 1회 제한 플래그
        self._preview_done = False

        # 메쉬 로딩
        self.mesh_files = mesh_paths
        self.meshes = [trimesh.load(m) for m in self.mesh_files]
        self.bounds = [trimesh.bounds.oriented_bounds(m) for m in self.meshes]
        self.bboxes = [np.stack([-extents/2, extents/2], axis=0).reshape(2, 3) for (_, extents) in self.bounds]
        self.mesh_labels = [os.path.splitext(os.path.basename(p))[0].lower() for p in self.mesh_files]

        # FoundationPose
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()

        # SAM2 & YOLO
        self.seg_model = SAM("sam2.1_b.pt")
        self.detector = None
        try:
            self.detector = YOLO(args.detector_weights or 'yolov8sall.pt')
            self.get_logger().info(f"[YOLO] loaded: {args.detector_weights}")
        except Exception as e:
            self.get_logger().warn(f"[YOLO] load failed: {e}")

        # 상태
        self.tracked = {}
        self.pose_publishers = {}
        self.frame_count = 0
        self.first_registered = False

        # 재획득 워커
        self._rq = queue.Queue(maxsize=32)
        self._rq_thread = threading.Thread(target=self._reacquire_loop, daemon=True)
        self._rq_thread.start()

        # === TF 브로드캐스터 (base_link -> cam_front_color_optical_frame) 추가 ===
        self.tf_broadcaster = TransformBroadcaster(self)
        self._tf_timer = self.create_timer(1.0/30.0, self._broadcast_camera_tf)

        # === TF 리스너 (cam -> base 조회용) 추가 ===
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
    # ---------- CameraInfo ----------
    def camera_info_callback(self, msg):
        if self.cam_K is None:
            self.cam_K = np.array(msg.k).reshape((3, 3))
            self.get_logger().info(f"Camera intrinsic matrix initialized: {self.cam_K}")

    # ---------- 최신 프레임 저장 ----------
    def image_callback(self, msg):
        try:
            color = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            with self._frame_lock:
                if self._latest['depth'] is None:
                    self._latest['color'] = color
                else:
                    self._latest['color'] = color
        except Exception as e:
            self.get_logger().warn(f"image_callback failed: {e}")

    def depth_callback(self, msg):
        try:
            # RViz echo 기준 encoding=16UC1 → m단위 float32로 변환
            depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if depth_mm.dtype != np.uint16:
                depth_mm = depth_mm.astype(np.uint16)
            depth = depth_mm.astype(np.float32) / 1000.0
            with self._frame_lock:
                self._latest['depth'] = depth
                self._latest['stamp'] = time.time()
        except Exception as e:
            self.get_logger().warn(f"depth_callback failed: {e}")

    # ---------- 워커 루프 ----------
    def _process_loop(self):
        while not self._stop:
            with self._frame_lock:
                ready = (self._latest['color'] is not None and
                         self._latest['depth'] is not None and
                         self.cam_K is not None)
                if ready:
                    color = np.ascontiguousarray(self._latest['color'])
                    depth = np.ascontiguousarray(self._latest['depth'])
                    ts = self._latest['stamp']
            if not ready:
                time.sleep(0.001); continue
            if self._proc_busy:
                time.sleep(0.0005); continue

            self._proc_busy = True
            try:
                self._process_one(color, depth, ts)
            except Exception as e:
                print(f"[process_loop] exception: {e}")
            finally:
                self._proc_busy = False

    # ---------- YOLO ----------
    def _yolo_boxes_bgr(self, bgr_img, conf=None):
        out = []
        if self.detector is None:
            return out
        cc = args.det_conf if conf is None else conf
        try:
            res = self.detector.predict(bgr_img, conf=cc, verbose=False)[0]
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

    # ---------- SAM (풀프레임) ----------
    def _sam_full_masks(self, rgb):
        H, W = rgb.shape[:2]
        masks = []
        try:
            res = self.seg_model.predict(rgb, verbose=False)
            if not res or not res[0] or getattr(res[0], 'masks', None) is None:
                return masks
            md = res[0].masks.data.cpu().numpy()  # (N,h,w)
            for k in range(md.shape[0]):
                m = (cv2.resize(md[k].astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                cnt = max(cnts, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                masks.append({'mask': m, 'contour': cnt.reshape(-1,1,2), 'box': (x, y, x+w, y+h)})
        except Exception as e:
            self.get_logger().warn(f"SAM full-image failed: {e}")
        return masks

    # ---------- 헬퍼 ----------
    def _make_rect_mask(self, rgb_shape, box):
        H, W = rgb_shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        m = np.zeros((H, W), dtype=np.uint8)
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = 255
        return m

    def _sam_mask_from_box(self, rgb, box):
        if self.seg_model is None:
            return None
        try:
            x1, y1, x2, y2 = [int(v) for v in box]
            res = self.seg_model.predict(rgb, bboxes=[(x1, y1, x2, y2)], verbose=False)
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

    # ---------- 초기 자동 등록 ----------
    def _auto_register_on_first_frame(self, color, depth, debug=False):
        H, W = color.shape[:2]
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

        if debug and (not self._preview_done):
            vis = color.copy()
            for (x1,y1,x2,y2,label,conf,cx,cy) in yolo:
                cv2.rectangle(vis, (x1,y1), (x2,y2), (255,0,255), 2)
                cv2.putText(vis, f"{label}:{conf:.2f}", (x1, max(0,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2, cv2.LINE_AA)
                cv2.circle(vis, (cx,cy), 6, (0,0,255), -1)
            cv2.putText(vis, "[AUTO-DEBUG] preview (2s, only once)...", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("AUTO DEBUG PREVIEW", vis[..., ::-1])
            t0 = time.time()
            while (time.time() - t0) < 2.0:
                if cv2.waitKey(1) in [27, ord('q')]:
                    break
            cv2.destroyWindow("AUTO DEBUG PREVIEW")
            self._preview_done = True

        for out_idx, (mi, det) in enumerate(sorted(selected, key=lambda x: x[0])):
            temp_mesh = self.meshes[mi]
            to_origin, extents = self.bounds[mi]
            bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

            x1, y1, x2, y2, _, _, _, _ = det
            ob_mask = self._sam_mask_from_box(color, (x1, y1, x2, y2))
            if ob_mask is None:
                ob_mask = self._make_rect_mask(color.shape, (x1, y1, x2, y2))

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

            self.tracked[out_idx] = {
                'pose_est': pose_est,
                'to_origin': to_origin,
                'bbox': bbox,
                'last_ts': time.time(),
                'label': self.mesh_labels[mi],
                'reacquiring': False,
                'lock': threading.Lock()
            }

        self.first_registered = True
        self.get_logger().info(f"AUTO: registered {len(self.tracked)} object(s).")

    # ---------- 수동 등록 (첫 프레임에만 실행) ----------
    def _manual_register_on_first_frame(self, color, depth):
        """
        SAM 풀-세그먼트로 장면의 마스크 후보를 만들고, 사용자가 클릭한 마스크를
        현재 mesh 큐 순서대로 하나씩 매칭하여 FoundationPose.register 실행.
        's': 현재 메시 스킵, 'r': 전체 초기화/다시, 'c' 또는 Enter/Space: 확정, 'q' 또는 ESC: 취소
        """
        if self.seg_model is None:
            self.get_logger().warn("manual mode: SAM model not loaded.")
            return

        H, W = color.shape[:2]
        masks = self._sam_full_masks(color)  # [{'mask': uint8(H,W), 'contour': (N,1,2), 'box': (x1,y1,x2,y2)}, ...]
        if not masks:
            self.get_logger().warn("manual mode: no masks from SAM.")
            return

        # 준비: 화면 그리기
        win = "MANUAL: Click objects (s=skip, r=redo, c=confirm)"
        overlay = color.copy()

        def _draw_overlay():
            vis = color.copy()
            for m in masks:
                cv2.drawContours(vis, [m['contour']], -1, (0, 255, 0), 2)
            next_idx = self._manual_next_mesh_idx
            next_name = os.path.splitext(os.path.basename(self.mesh_files[next_idx]))[0] if next_idx < len(self.mesh_files) else "None"
            y0, dy = 28, 22
            lines = [
                f"Next mesh: {next_name}",
                "Instructions:",
                "- Left click: select mask for current mesh",
                "- s: skip this mesh   r: redo all   c/Enter/Space: confirm   q/ESC: cancel",
            ]
            for i, line in enumerate(lines):
                y = y0 + i*dy
                cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
            return vis

        # 상태
        self._manual_next_mesh_idx = 0
        selected_entries = []  # list of dicts: { 'mesh_idx', 'mask' }

        # 클릭 핸들러
        def _on_mouse(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            # 해당 픽셀을 포함하는 가장 큰 컨투어/마스크 찾기
            hit = None
            best_area = -1
            for m in masks:
                if m['mask'][max(0,min(H-1,y)), max(0,min(W-1,x))] == 255:
                    area = int(cv2.contourArea(m['contour']))
                    if area > best_area:
                        best_area = area
                        hit = m
            if hit is None:
                self.get_logger().info("manual: no mask under cursor.")
                return
            mi = self._manual_next_mesh_idx
            if mi >= len(self.meshes):
                self.get_logger().warn("manual: no more meshes to assign.")
                return

            # 현재 mesh에 선택된 마스크를 배정
            selected_entries.append({'mesh_idx': mi, 'mask': hit['mask']})
            self._manual_next_mesh_idx += 1
            self.get_logger().info(f"manual: assigned mask to mesh idx={mi} ({self.mesh_labels[mi]})")

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, _on_mouse)

        while True:
            cv2.imshow(win, _draw_overlay()[..., ::-1])  # BGR로 변환해서 표시
            key = cv2.waitKey(20) & 0xFF
            if key in [ord('q'), 27]:  # ESC
                self.get_logger().info("manual: canceled.")
                cv2.destroyWindow(win)
                return
            elif key == ord('s'):
                # 현재 mesh 스킵
                self._manual_next_mesh_idx += 1
                self.get_logger().info("manual: skipped current mesh.")
            elif key == ord('r'):
                # 전부 되돌리기
                self._manual_next_mesh_idx = 0
                selected_entries.clear()
                self.get_logger().info("manual: redo all selections.")
            elif key in [ord('c'), 13, 32]:  # 'c', Enter(13), Space(32) → 확정
                break

        cv2.destroyWindow(win)

        if not selected_entries:
            self.get_logger().warn("manual: no selections were made.")
            return

        # 선택된 것들로 실제 register 수행 → self.tracked에 올리기
        # 기존 tracked 비우고 새로 세팅
        self.tracked.clear()
        out_idx = 0
        for entry in selected_entries:
            mi = entry['mesh_idx']
            if mi >= len(self.meshes):
                continue
            temp_mesh = self.meshes[mi]
            to_origin, extents = self.bounds[mi]
            bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
            ob_mask = entry['mask']

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

                self.tracked[out_idx] = {
                    'pose_est': pose_est,
                    'to_origin': to_origin,
                    'bbox': bbox,
                    'last_ts': time.time(),
                    'label': self.mesh_labels[mi],   # YOLO 앵커링 시 사용
                    'reacquiring': False,
                    'lock': threading.Lock(),
                }
                out_idx += 1
            except Exception as e:
                self.get_logger().warn(f"manual: register failed for mesh idx={mi}. err={e}")

        if out_idx > 0:
            self.first_registered = True
            self.get_logger().info(f"MANUAL: registered {out_idx} object(s).")
        else:
            self.get_logger().warn("manual: no objects registered.")


    # ---------- 메인 처리 ----------
    def _process_one(self, color, depth, ts):
        H, W = color.shape[:2]
        depth = depth.copy()
        depth[(depth < 0.1) | (~np.isfinite(depth))] = 0

        if not self.first_registered:
            if args.mode in ('auto', 'auto_debug'):
                self._auto_register_on_first_frame(color, depth, debug=(args.mode=='auto_debug'))
            else:
                # === 수동 등록 UI 실행 ===
                self._manual_register_on_first_frame(color, depth)
            if not self.first_registered:
                self._viz_only(color)
                return

        need_anchor = (self.frame_count % max(1, args.yolo_anchor_period) == 0)
        yolo_boxes = self._yolo_boxes_bgr(color[..., ::-1]) if need_anchor else None

        vis = color.copy()
        if args.viz_yolo and need_anchor and yolo_boxes:
            vis = self._draw_yolo_boxes(vis, yolo_boxes)
        dead_ids = []
        for oid, data in self.tracked.items():
            lock = data['lock']
            if data['reacquiring']:
                continue

            with lock:
                pose_est = data['pose_est']
                to_origin = data['to_origin']
                bbox = data['bbox']
                last_ts = data['last_ts']

                dt = max(0.0, (ts - (last_ts if last_ts else ts)))
                extra = int(max(0.0, (dt * args.iter_gain_per_sec)))
                iters = min(args.track_refine_iter_max, args.track_refine_iter_base + extra)

                try:
                    pose = pose_est.track_one(rgb=color, depth=depth, K=self.cam_K, iteration=iters)
                    center_pose = pose @ np.linalg.inv(to_origin)
                except Exception as e:
                    self.get_logger().warn(f"Track {oid} exception → queue reacquire: {e}")
                    if args.lost_reacquire:
                        self._queue_reacquire(oid)
                    else:
                        dead_ids.append(oid)
                    continue

                vis = self._visualize_pose(vis, center_pose, bbox)
                self._publish_pose_stamped(center_pose, f"/Current_OBJ_position_{oid+1}")

                if need_anchor and yolo_boxes:
                    det = self._pick_best_det_for_label(yolo_boxes, data['label'])
                    if det is not None:
                        _, _, _, _, _, _, cx, cy = det
                        px = self._project_center(center_pose, self.cam_K)
                        if px is not None:
                            dx = np.hypot(px[0]-cx, px[1]-cy)
                            if dx > args.drift_px_thresh:
                                self.get_logger().info(f"[Drift] oid={oid} {dx:.1f}px > {args.drift_px_thresh} → reacquire")
                                if args.lost_reacquire:
                                    self._queue_reacquire(oid)

                data['last_ts'] = ts

        for oid in dead_ids:
            self.tracked.pop(oid, None)

        cv2.imshow('Pose Estimation & Tracking', vis[..., ::-1])
        cv2.waitKey(1)
        self.frame_count += 1

    # ---------- 보조들 ----------
    def _draw_yolo_boxes(self, img, dets):
        vis = img.copy()
        for (x1, y1, x2, y2, label, conf, cx, cy) in dets:
            # box
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
            # center
            cv2.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            # label:conf
            txt = f"{label}:{conf:.2f}"
            cv2.putText(vis, txt, (int(x1), max(0, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, float(args.viz_yolo_font), (255, 0, 255), 2, cv2.LINE_AA)
        # header mark
        cv2.putText(vis, "[YOLO tick]", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 255), 2, cv2.LINE_AA)
        return vis
        

    def _pick_best_det_for_label(self, dets, label):
        cands = [d for d in dets if d[4] == label]
        if not cands: return None
        cands.sort(key=lambda d: (d[5], (d[2]-d[0])*(d[3]-d[1])), reverse=True)
        return cands[0]

    def _project_center(self, T, K):
        try:
            c = T[:3, 3].reshape(3,1)
            if c[2,0] <= 0: return None
            uv = (K @ (c / c[2,0]))[:2, 0]
            return (float(uv[0]), float(uv[1]))
        except:
            return None

    def _visualize_pose(self, img, center_pose, bbox):
        vis = draw_posed_3d_box(self.cam_K, img=img, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=self.cam_K, thickness=3, transparency=0, is_input_rgb=True)
        return vis

    def _lookup_T(self, parent_frame, child_frame):
        try:
            t = self.tf_buffer.lookup_transform(parent_frame, child_frame, rclpy.time.Time())
        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed ({parent_frame}->{child_frame}): {e}")
            return None
        q_xyzw = [
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w,
        ]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R.from_quat(q_xyzw).as_matrix()
        T[:3, 3] = np.array([
            t.transform.translation.x,
            t.transform.translation.y,
            t.transform.translation.z,
        ], dtype=np.float64)
        return T

    def _publish_pose_stamped(self, center_pose, topic_name):
        if topic_name not in self.pose_publishers:
            self.pose_publishers[topic_name] = self.create_publisher(PoseStamped, topic_name, 10)

        # 1) TF에서 base->cam 가져오기
        T_base_cam = self._lookup_T(BASE_FRAME, CAM_FRAME)
        if T_base_cam is None:
            return

        # 2) cam->obj(center) 는 center_pose
        T_cam_obj = center_pose

        # 3) base->obj = (base->cam) * (cam->obj)
        T_base_obj = T_base_cam @ T_cam_obj

        # 4) 퍼블리시
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

    def _viz_only(self, color):
        cv2.imshow('Pose Estimation & Tracking', color[..., ::-1])
        cv2.waitKey(1)

    # ---------- 재획득 큐잉 & 워커 ----------
    def _queue_reacquire(self, oid):
        data = self.tracked.get(oid)
        if data is None: return
        if data['reacquiring']: return
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
                self.get_logger().warn(f"Reacquire {oid} failed: {e}")
            finally:
                if oid in self.tracked:
                    self.tracked[oid]['reacquiring'] = False
                self._rq.task_done()

    # 클래스 PoseEstimationNode 안에 그대로 교체 붙여넣기

    def _depth_mask_in_box(self, depth_img, box, eps=0.05, min_pix=200, morph=3):
        """YOLO 박스 내부에서 depth 중심값 기반으로 얇은 마스크 생성 (빠름). 실패시 None."""
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
        # 최신 프레임 스냅샷
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

        # 1) YOLO로 대상 박스 선택 (라벨 매칭)
        yolo = self._yolo_boxes_bgr(color[..., ::-1], conf=args.det_conf)
        if not yolo:
            return
        dets = [d for d in yolo if d[4] == label]
        if not dets:
            return
        dets.sort(key=lambda d: (d[5], (d[2]-d[0])*(d[3]-d[1])), reverse=True)
        x1, y1, x2, y2 = dets[0][0], dets[0][1], dets[0][2], dets[0][3]

        # 2) 빠른 경로: depth-클러스터 마스크 → 실패 시 rect
        ob_mask = self._depth_mask_in_box(depth, (x1, y1, x2, y2))
        if ob_mask is None:
            ob_mask = self._make_rect_mask(color.shape, (x1, y1, x2, y2))

        with lock:
            pose_est = data['pose_est']
            try:
                pose_est.register(K=K, rgb=color, depth=depth,
                                ob_mask=ob_mask, iteration=args.est_refine_iter)
                data['last_ts'] = time.time()
                return  # 성공 → SAM 불필요
            except Exception as e:
                self.get_logger().warn(f"Fast reacquire failed (depth/rect). oid={oid} err={e}")

        # 3) 느린 경로: SAM(box) — 쿨다운 이후에만 시도 (최후수단)
        if not hasattr(self, "_last_sam_time"):
            self._last_sam_time = {}
        if not hasattr(self, "slow_sam_cooldown"):
            self.slow_sam_cooldown = 1.0  # 초당 1회 이하

        last = float(self._last_sam_time.get(oid, 0.0))
        now = time.time()
        if (now - last) < float(self.slow_sam_cooldown):
            return
        if self.seg_model is None:
            return

        ob_mask_sam = self._sam_mask_from_box(color, (x1, y1, x2, y2))
        if ob_mask_sam is None:
            return

        with lock:
            try:
                pose_est = data['pose_est']
                pose_est.register(K=K, rgb=color, depth=depth,
                                ob_mask=ob_mask_sam, iteration=args.est_refine_iter)
                data['last_ts'] = time.time()
                self._last_sam_time[oid] = now
                return
            except Exception as e:
                self.get_logger().warn(f"SAM reacquire failed. oid={oid} err={e}")
                return


    # ---------- TF broadcaster: base_link -> cam_front_color_optical_frame ----------
    def _broadcast_camera_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = BASE_FRAME                   # "world"
        t.child_frame_id = CAM_FRAME                     # "camera_color_optical_frame"
        t.transform.translation.x = CALIB_T['x']
        t.transform.translation.y = CALIB_T['y']
        t.transform.translation.z = CALIB_T['z']
        t.transform.rotation.x = CALIB_T['qx']
        t.transform.rotation.y = CALIB_T['qy']
        t.transform.rotation.z = CALIB_T['qz']
        t.transform.rotation.w = CALIB_T['qw']
        self.tf_broadcaster.sendTransform(t)

# -------------------------------
# main
# -------------------------------
def main(args_ros=None):
    source_directory = "demo_data"
    file_paths = glob.glob(os.path.join(source_directory, '**', '*.obj'), recursive=True) + \
                 glob.glob(os.path.join(source_directory, '**', '*.stl'), recursive=True) + \
                 glob.glob(os.path.join(source_directory, '**', '*.STL'), recursive=True)

    if args.mode in ('auto','auto_debug'):
        mesh_paths = file_paths
    else:
        mesh_paths = rearrange_files(file_paths)

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