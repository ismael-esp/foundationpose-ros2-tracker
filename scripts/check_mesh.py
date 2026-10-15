#!/usr/bin/env python3
"""Check object meshes before tracking: units (metres), size, vertex count, watertightness.

    python scripts/check_mesh.py                 # every mesh under demo_data/
    python scripts/check_mesh.py "red cube"      # one object (same lookup as foundation_pose.py)

FoundationPose needs the mesh in METRES with a sensible origin. A mesh exported in mm/cm
(diameter of metres) registers far off the object and the box explodes on screen.
"""
import glob
import os
import re
import sys

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find(obj):
    for d in (obj, re.sub(r'[^a-z0-9]+', '_', obj.lower()).strip('_')):
        c = [f for e in ('obj', 'stl', 'STL') for f in glob.glob(os.path.join(ROOT, 'demo_data', d, f'*.{e}'))]
        if c:
            return [x for x in c if os.path.splitext(os.path.basename(x))[0].lower() == obj.lower()] or c[:1]
    return []


paths = find(sys.argv[1]) if len(sys.argv) > 1 else sorted(
    p for p in glob.glob(os.path.join(ROOT, 'demo_data', '*', '*.obj')) if '/old_data/' not in p)
if not paths:
    sys.exit(f"no mesh found for {sys.argv[1]!r} under demo_data/")

print(f"{'mesh':42s} {'extents [m]':24s} {'diam':>6s} {'verts':>7s}  verdict")
for p in paths:
    m = trimesh.load(p, force='mesh')
    e, d = m.extents, float(np.linalg.norm(m.extents))
    if d > 0.6:
        v = 'TOO BIG - probably mm/cm units, rescale to metres'
    elif d < 0.02:
        v = 'TOO SMALL - check units'
    elif len(m.vertices) > 100_000:
        v = 'ok, but heavy - decimate for speed'
    else:
        v = 'ok'
    rel = os.path.relpath(p, ROOT)
    print(f"{rel:42s} {str(np.round(e, 3)):24s} {d:6.3f} {len(m.vertices):7d}  {v}")
