#!/usr/bin/env bash
# Live tracking on a RealSense camera. Everything after the object name is passed through
# to foundation_pose.py, e.g.
#
#   bash scripts/run_tracker.sh "red cube" --camera head --hand_frame ra_palm_link
#   bash scripts/run_tracker.sh "mustard" --color_topic /camera/camera/color/image_raw \
#        --depth_topic /camera/camera/aligned_depth_to_color/image_raw \
#        --camera_info_topic /camera/camera/color/camera_info --cam_frame camera_color_optical_frame
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OBJ="${1:?usage: run_tracker.sh <object name> [foundation_pose.py flags]}"; shift

eval "$(conda shell.bash hook)"; conda activate "${ENV_NAME:-fp_ros}"
source /opt/ros/humble/setup.bash
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$(python3 -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"

HAS_HAND=0; for a in "$@"; do [ "$a" = "--hand_frame" ] && HAS_HAND=1; done
EXTRA=(); [ $HAS_HAND = 1 ] || EXTRA=(--hand_frame "${HAND_FRAME:-ra_palm_link}")

cd "$ROOT"
exec python3 foundation_pose.py --object "$OBJ" --viz_local "${EXTRA[@]}" "$@"
