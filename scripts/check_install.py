#!/usr/bin/env python3
"""Sanity-check the tracker environment. Run from the repo root:  python scripts/check_install.py"""
import os
import sys
import importlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.append('./FoundationPose')
sys.path.append('./FoundationPose/nvdiffrast')

ok = True


def check(name, fn):
    global ok
    try:
        info = fn()
        print(f"  [ OK ] {name}{(' - ' + info) if info else ''}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  [FAIL] {name} - {type(e).__name__}: {e}")


def _torch():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    return f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}"


def _numpy():
    import numpy
    assert int(numpy.__version__.split('.')[0]) < 2, "NumPy 2.x breaks ROS Humble cv_bridge"
    return numpy.__version__


def _weights():
    w = 'FoundationPose/weights'
    runs = sorted(os.listdir(w))
    for r in ('2023-10-28-18-33-37', '2024-01-11-20-02-45'):
        assert os.path.isfile(os.path.join(w, r, 'model_best.pth')), f"missing {r}/model_best.pth"
    return ', '.join(runs)


print("FoundationPose ROS 2 tracker - environment check")
check("torch + CUDA", _torch)
check("numpy < 2", _numpy)
check("nvdiffrast", lambda: importlib.import_module('nvdiffrast.torch') and None)
check("mycpp (C++ ext)", lambda: importlib.import_module('mycpp') and None)
check("mycuda (CUDA ext)", lambda: importlib.import_module('common') and None)
check("pytorch3d", lambda: importlib.import_module('pytorch3d').__version__)
check("FoundationPose estimater", lambda: importlib.import_module('estimater') and None)
check("FoundationPose weights", _weights)
check("ultralytics (YOLO-World + SAM2)", lambda: importlib.import_module('ultralytics').__version__)
check("rclpy (source /opt/ros/humble/setup.bash)", lambda: importlib.import_module('rclpy') and None)
print("\nAll good." if ok else "\nSome checks failed - see README > Troubleshooting.")
sys.exit(0 if ok else 1)
