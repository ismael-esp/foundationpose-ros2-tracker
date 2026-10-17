#!/usr/bin/env python3
# multi_object_switcher.py
# In-process MULTI-OBJECT switcher on top of foundation_pose.py (single-object).
#
# Why this exists:
#   The BODex pick-and-place workflow tracks ONE object at a time, but cycles
#   through ~10 of them sequentially. With plain foundation_pose.py that means:
#   stop the process, edit the --object / --object_aliases on the command line,
#   restart, and pay the full model-reload cost (FoundationPose nets + SAM2 +
#   YOLO-World, ~15-30 s) every single switch.
#
#   This node loads those heavy, OBJECT-INDEPENDENT models exactly ONCE and then
#   swaps only the three small object-SPECIFIC things on a keypress:
#       1. the mesh        -> FoundationPose.reset_object(...)   (nets stay resident)
#       2. the prompt      -> YOLO-World.set_classes([name, *aliases])
#       3. the publisher   -> repoint self.pose_pub to /<slug>_pose
#   ...then drops to SEARCHING (reusing foundation_pose.py's own 'n' re-detect
#   reset) so the new object re-locks. Switch cost goes from a full restart to
#   ~instant.
#
#   foundation_pose.py is intentionally left UNTOUCHED — this file imports it as
#   a module and subclasses SingleObjectPoseNode. Every CLI flag, the BODex
#   Z-up / yaw-lock contract, the orientation_mode policy, precision handling,
#   and the camera presets are inherited verbatim.
#
# Usage (same as foundation_pose.py, but --objects is a CSV; aliases come from
# the OBJECT_CATALOG below, not the command line):
#
#   # all 10 (DEFAULT_OBJECTS order) — --objects may be omitted entirely:
#   CUDA_VISIBLE_DEVICES=1 python3 multi_object_switcher.py \
#       --hand_frame ra_palm_link --viz_local --camera extra --cam_world_source tf
#   # ...or pass an explicit subset/order:
#   CUDA_VISIBLE_DEVICES=1 python3 multi_object_switcher.py \
#       --objects "tiger,red_cube,green_cube,pikachu,whale,milk,snorlax,mustard,green_can,tea_peach" \
#       --hand_frame ra_palm_link --viz_local --camera extra --cam_world_source tf
#
# In the viewer window:
#   ]  or  .   -> next object        [  or  ,   -> previous object
#   1..9 / 0   -> jump to object N (0 = the 10th)
#   n          -> re-detect the CURRENT object (unchanged foundation_pose.py key)
#
# The active object is shown as a banner on the overlay, and the published topic
# always follows it (/<slug>_pose), so the BODex consumer sees the right object.

import sys
import re

# Import the single-object node as a library. This runs foundation_pose.py's
# module-level setup (sys.path.append('./FoundationPose'), `from estimater
# import *`, the _DET_DEVICE probe, class/flag definitions) ONCE, in foundation_
# pose's OWN namespace — so the estimater `Image` wildcard cannot shadow the
# ROS2 Image class in this file. We never `from foundation_pose import *`.
import foundation_pose as fp

import trimesh
import numpy as np
import cv2
import rclpy
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy)
from geometry_msgs.msg import PoseStamped


# ---------------------------------------------------------------------------
# Object catalog (Option C): object_id -> extra YOLO-World text prompts.
#
# The object_id is the demo_data/<id>/ folder name AND the mesh stem AND the
# published topic stem (/<slug>_pose) — it must match the BODex object_id.
# Aliases are open-vocabulary hints fed to YOLO-World alongside the primary
# name; tune them per object. An object missing from this dict simply uses its
# own name as the only prompt. Edit freely — this is the one place to do it.
# ---------------------------------------------------------------------------
# The canonical 10-object demo set, in order, with the aliases proven on the
# command line (2026-06-29). Keep this order — it is what --objects defaults to
# and what the 1-9/0 jump keys map to.
OBJECT_CATALOG = {
    'tiger':      ['stuffed tiger', 'plush toy', 'toy tiger'],
    'red_cube':   ['red cube', 'wooden cube with red tape', 'red taped cube'],
    'green_cube': ['green cube', 'wooden cube with green tape', 'green taped cube'],
    'pikachu':    ['stuffed pikachu pokemon', 'plush toy', 'toy pikachu'],
    'whale':      ['stuffed orca whale', 'plush toy', 'toy whale'],
    'milk':       ['stuffed hello kitty milk', 'plush toy', 'toy yellow milk box'],
    'snorlax':    ['stuffed snorlax pokemon', 'plush toy', 'toy snorlax'],
    'mustard':    ['yellow mustard sauce bottle', 'yellow sauce bottle', 'yellow mustard'],
    'green_can':  ['green metal drink can', 'green can', 'can'],
    'tea_peach':  ['peach tea plastic bottle', 'plastic bottle', 'tea bottle'],
}

# Default object list (and 1-9/0 jump order) when --objects is omitted.
DEFAULT_OBJECTS = list(OBJECT_CATALOG.keys())


def _slugify(name):
    """Same slug rule foundation_pose.py uses for the /<slug>_pose topic."""
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_') or 'object'


def _aliases_for(name):
    return list(OBJECT_CATALOG.get(name, []))


# ---------------------------------------------------------------------------
# Multi-object node
# ---------------------------------------------------------------------------
class MultiObjectPoseNode(fp.SingleObjectPoseNode):
    """Single-object tracker that can hot-swap which object it tracks.

    Subclasses fp.SingleObjectPoseNode. Overrides exactly two seams:
      * __init__         — build the object list, pre-cache meshes/scorers,
                           pre-create one publisher per object.
      * _publish_overlay — draw the active-object banner and dispatch the
                           object-switch keys (the only place a waitKey runs).
    Everything else (detection, hybrid track loop, occlusion hand-attach,
    gravity/yaw lock, precision, telemetry) is inherited unchanged.
    """

    def __init__(self, args, init_idx=0):
        self.objects = list(args.objects_list)
        self.obj_idx = init_idx % len(self.objects)

        # Resolve the chosen initial object before super().__init__, which
        # expects args.object / args.object_aliases_list / mesh_path to be set.
        first = self.objects[self.obj_idx]
        args.object = first
        args.object_aliases_list = _aliases_for(first)
        mesh_path = fp._resolve_mesh(first)
        if mesh_path is None:
            raise FileNotFoundError(
                f"No mesh for '{first}' under demo_data/. "
                f"Expected demo_data/{first}/<name>.obj")

        # Base __init__ loads the heavy nets, registers subscriptions/TF, creates
        # the first object's publisher, and STARTS the processing thread.
        super().__init__(args, mesh_path)

        # ---- Pre-cache every object's mesh data + QualityScorer ----
        # Front-loads the per-switch cost so switching is ~instant: only
        # reset_object() (voxel-downsample of cached verts) runs at switch time.
        self._obj_cache = {}
        for name in self.objects:
            self._obj_cache[name] = self._build_cache_entry(name)

        # ---- Pre-create one /<slug>_pose publisher per object ----
        # Done here on the main thread (thread-safe) so the switch path never
        # has to create/destroy a publisher from the processing thread.
        qos_rel = QoSProfile(depth=1)
        qos_rel.reliability = ReliabilityPolicy.RELIABLE
        qos_rel.history = HistoryPolicy.KEEP_LAST
        qos_rel.durability = DurabilityPolicy.VOLATILE
        # Base already made the first object's publisher as self.pose_pub.
        self._pose_pubs = {self.slug: self.pose_pub}
        for name in self.objects:
            slug = _slugify(name)
            if slug not in self._pose_pubs:
                self._pose_pubs[slug] = self.create_publisher(
                    PoseStamped, f"/{slug}_pose", qos_rel)

        self.get_logger().info(
            f"[multi] {len(self.objects)} objects loaded: {self.objects}. "
            f"Keys in viewer: ']'/'.' next, '['/',' prev, 1-9/0 jump, "
            f"'n' re-detect current. Active -> {first} ({self.pose_topic})")

    # ------------------------------------------------------------------
    def _build_cache_entry(self, name):
        mesh_path = fp._resolve_mesh(name)
        if mesh_path is None:
            raise FileNotFoundError(
                f"No mesh for '{name}' under demo_data/. "
                f"Expected demo_data/{name}/<name>.obj")
        mesh = trimesh.load(mesh_path, force='mesh')
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2.0, extents / 2.0], axis=0).reshape(2, 3)
        entry = {
            'name': name,
            'slug': _slugify(name),
            'mesh': mesh,
            'to_origin': to_origin,
            'bbox': bbox,
            'diameter': float(np.linalg.norm(extents)),
            'qscorer': fp.QualityScorer(mesh, max_verts=384),
            'aliases': _aliases_for(name),
        }
        self.get_logger().info(
            f"[multi] cached '{name}' (verts={len(mesh.vertices)}, "
            f"extents={np.round(extents, 4)}, topic /{entry['slug']}_pose)")
        return entry

    # ------------------------------------------------------------------
    def _request_redetect(self, why):
        """Trigger foundation_pose.py's own drop-to-SEARCHING reset (the same
        path the 'n' key uses), which clears the yaw lock / EMA / cam_pose and
        re-runs detection for whatever object is now active."""
        if not self._manual_redetect:
            self._manual_redetect = True
            self.get_logger().warn(f"[multi] re-detect requested ({why})")

    # ------------------------------------------------------------------
    def _switch_to(self, idx):
        """Hot-swap the tracked object. Heavy models stay resident."""
        idx = idx % len(self.objects)
        if idx == self.obj_idx:
            # Re-selecting the current object == re-detect it.
            self._request_redetect("same object re-selected")
            return

        name = self.objects[idx]
        c = self._obj_cache[name]

        # 1. Swap the object-specific node state used by the overlay / detectors.
        self.mesh = c['mesh']
        self.to_origin = c['to_origin']
        self.bbox = c['bbox']
        self.obj_diameter = c['diameter']
        self.qscorer = c['qscorer']

        # 2. Swap the FoundationPose object (re-voxelizes; nets are untouched).
        self.pose_est.reset_object(
            model_pts=c['mesh'].vertices,
            model_normals=c['mesh'].vertex_normals,
            mesh=c['mesh'])

        # 3. Re-prompt the detector(s): new name + aliases, new quality scorer.
        self.a.object = name
        self.a.object_aliases_list = c['aliases']
        if hasattr(self, 'detectors'):
            for d in self.detectors:
                if isinstance(d, fp.YoloWorldDetector):
                    d._scorer_ref = self.qscorer
                    if getattr(d, 'model', None) is not None:
                        try:
                            d.model.set_classes([name] + list(c['aliases']))
                        except Exception as e:
                            self.get_logger().warn(
                                f"[multi] set_classes failed: {e}")
                else:
                    # DepthClusterDetector uses obj_diameter; keep it current.
                    if hasattr(d, 'obj_diameter'):
                        d.obj_diameter = self.obj_diameter
        if hasattr(self, 'click_detector'):
            # Click->SAM2 is object-agnostic; nothing to swap, but keep the
            # diameter in sync if it tracks one.
            if hasattr(self.click_detector, 'obj_diameter'):
                self.click_detector.obj_diameter = self.obj_diameter

        # 4. Repoint the publisher + TF frame (pre-created in __init__).
        self.slug = c['slug']
        self.obj_frame = f"{self.slug}_frame"
        self.pose_topic = f"/{self.slug}_pose"
        self.pose_pub = self._pose_pubs[self.slug]

        self.obj_idx = idx
        self.get_logger().info(
            f"[multi] switched -> [{idx + 1}/{len(self.objects)}] {name} "
            f"(topic {self.pose_topic}, prompts={[name] + list(c['aliases'])})")
        # 5. Drop the old lock and re-acquire the new object.
        self._request_redetect(f"switched to {name}")

    # ------------------------------------------------------------------
    def _handle_key(self, k):
        if k == 255 or k == 0xFF:          # no key this frame
            return
        if k in (ord(']'), ord('.')):
            self._switch_to(self.obj_idx + 1)
        elif k in (ord('['), ord(',')):
            self._switch_to(self.obj_idx - 1)
        elif k == ord('n'):
            self._request_redetect("'n' pressed")
        elif ord('1') <= k <= ord('9'):
            target = k - ord('1')
            if target < len(self.objects):
                self._switch_to(target)
        elif k == ord('0'):
            if len(self.objects) >= 10:
                self._switch_to(9)

    # ------------------------------------------------------------------
    def _draw_object_legend(self, vis):
        """Small numbered object list in the top-right corner so the jump-key
        order (1-9/0) is always visible. The active object is highlighted."""
        H, W = vis.shape[:2]
        fs, lh = 0.42, 16                      # small font + tight line height
        # widest label decides the box width
        labels = [f"{(i + 1) % 10}  {name}"
                  for i, name in enumerate(self.objects)]
        wmax = max((cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0]
                    for s in labels), default=60)
        x0 = max(10, W - wmax - 18)
        y0 = 16
        for i, label in enumerate(labels):
            y = y0 + i * lh
            active = (i == self.obj_idx)
            color = (255, 255, 0) if active else (220, 220, 220)  # yellow / grey (vis is RGB)
            text = (">" + label) if active else (" " + label)
            cv2.putText(vis, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (0, 0, 0), 3, cv2.LINE_AA)          # outline
            cv2.putText(vis, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def _publish_overlay(self, img_rgb, status):
        """Override: prepend the active-object banner and dispatch switch keys.

        This is the single place a waitKey runs in the node, so all keyboard
        handling lives here. Non-viz_local path is delegated to the base so the
        overlay image topics still publish unchanged."""
        banner = (f"[obj {self.obj_idx + 1}/{len(self.objects)}] "
                  f"{self.objects[self.obj_idx]}   "
                  f"]/. next  [/, prev  1-9/0 jump  n redetect")
        status2 = banner + "\n" + str(status)

        if not self.a.viz_local:
            return super()._publish_overlay(img_rgb, status2)

        vis = img_rgb.copy()
        for i, line in enumerate(status2.split('\n')):
            y = 28 + i * 30
            cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 5, cv2.LINE_AA)          # outline
            cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)
        self._draw_object_legend(vis)
        cv2.imshow('foundation_pose', vis[..., ::-1])
        k = cv2.waitKey(1) & 0xFF
        self._handle_key(k)


# ---------------------------------------------------------------------------
# arg parsing: reuse fp.parse_args verbatim, but accept --objects (CSV).
# ---------------------------------------------------------------------------
def parse_args_multi():
    """Reuse foundation_pose.py's full argparse. We translate --objects (CSV)
    into a --object <first> so fp.parse_args (which requires --object) accepts
    it, then attach the parsed list as args.objects_list. Every other flag is
    passed straight through."""
    argv = list(sys.argv[1:])
    objects = None

    # Accept "--objects a,b,c" and "--objects=a,b,c".
    for i, tok in enumerate(argv):
        if tok == '--objects' and i + 1 < len(argv):
            objects = [s.strip() for s in argv[i + 1].split(',') if s.strip()]
            argv = argv[:i] + ['--object', objects[0]] + argv[i + 2:]
            break
        if tok.startswith('--objects='):
            objects = [s.strip() for s in tok.split('=', 1)[1].split(',')
                       if s.strip()]
            argv = argv[:i] + ['--object', objects[0]] + argv[i + 1:]
            break

    # If neither --objects nor a single --object was given, load the full
    # canonical 10-object set (DEFAULT_OBJECTS), injecting --object so
    # fp.parse_args (which requires --object) is satisfied.
    if objects is None and '--object' not in argv:
        objects = list(DEFAULT_OBJECTS)
        argv = ['--object', objects[0]] + argv

    sys.argv = [sys.argv[0]] + argv
    a = fp.parse_args()
    a.objects_list = objects if objects else [a.object]
    return a


def _prompt_initial_object(objects):
    """Block in the terminal (before any window opens) until the operator picks
    the initial object to detect. Returns a 0-based index. Numbers shown are
    1..N (N = number of objects). Non-interactive / EOF defaults to 1."""
    n = len(objects)
    print("\n" + "=" * 44)
    print("  Select the INITIAL object to detect:")
    for i, name in enumerate(objects):
        print(f"   {i + 1:2d}  {name}")
    print("=" * 44)
    while True:
        try:
            raw = input(f"Enter number [1-{n}] (default 1): ").strip()
        except EOFError:
            print("  (no input -> defaulting to 1)")
            return 0
        if raw == "":
            return 0
        try:
            v = int(raw)
        except ValueError:
            print("  not a number, try again.")
            continue
        if 1 <= v <= n:
            print(f"  -> starting on [{v}] {objects[v - 1]}\n")
            return v - 1
        print(f"  out of range, enter 1-{n}.")


def main():
    args = parse_args_multi()
    init_idx = _prompt_initial_object(args.objects_list)
    rclpy.init()
    node = MultiObjectPoseNode(args, init_idx=init_idx)
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
