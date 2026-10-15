#!/usr/bin/env bash
# Offline demo: replay a recorded RealSense RGB-D bag through foundation_pose.py — no camera,
# no robot. Runs on an isolated ROS domain and records overlay MP4 + pose CSV under runs/replay/.
#
#   bash scripts/replay_demo.sh                              # green cube on the sample bag
#   bash scripts/replay_demo.sh "red cube"                   # any object with a mesh in demo_data/
#   bash scripts/replay_demo.sh "green cube" path/to/bag 0.5 # custom bag + playback rate
#   bash scripts/replay_demo.sh "green cube" "" 0.5 --viz_local  # watch live (window replaces the recorded overlay)
set -eo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OBJ="${1:-green cube}"
BAG="${2:-}"; BAG="${BAG:-$ROOT/data/front_4obj_move_db3}"
RATE="${3:-0.5}"
BAG_URL="https://github.com/ismael-esp/foundationpose-ros2-tracker/releases/download/v1.0/front_4obj_move_db3.tar.zst"

if [ ! -d "$BAG" ]; then
  echo "==> Sample bag not found, downloading (~1 GB) ..."
  mkdir -p "$ROOT/data"
  curl -L "$BAG_URL" | zstd -d | tar -x -C "$ROOT/data"
fi

eval "$(conda shell.bash hook)"; conda activate "${ENV_NAME:-fp_ros}"
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}" ROS_LOCALHOST_ONLY=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$(python3 -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH:-}"

SLUG="$(echo "$OBJ" | tr 'A-Z' 'a-z' | sed -E 's/[^a-z0-9]+/_/g; s/^_|_$//g')"
OUT="$ROOT/runs/replay/$SLUG"; mkdir -p "$(dirname "$OUT")"
DUR="$(python3 - "$BAG" "$RATE" <<'EOF'
import sys, yaml, glob, os
m = yaml.safe_load(open(os.path.join(sys.argv[1], 'metadata.yaml')))['rosbag2_bagfile_information']
print(int(m['duration']['nanoseconds'] / 1e9 / float(sys.argv[2])) + 25)
EOF
)"

cd "$ROOT"
python3 foundation_pose.py --object "$OBJ" --hand_frame ra_palm_link --detect_mode auto \
  --color_topic /camera/camera/color/image_raw \
  --depth_topic /camera/camera/aligned_depth_to_color/image_raw \
  --camera_info_topic /camera/camera/color/camera_info \
  --cam_frame camera_color_optical_frame "${@:4}" > "$OUT.node.log" 2>&1 &
NODE=$!
trap 'kill -INT $NODE 2>/dev/null || true' EXIT

echo "==> Loading models (log: $OUT.node.log) ..."
until grep -q "Waiting for color+depth" "$OUT.node.log"; do
  kill -0 $NODE 2>/dev/null || { echo "tracker exited, see $OUT.node.log"; exit 1; }
  sleep 1
done

python3 scripts/record_overlay.py --pose_topic "/${SLUG}_pose" --out "$OUT" --duration "$DUR" &
REC=$!
sleep 2
echo "==> Playing $BAG at ${RATE}x"
ros2 bag play "$BAG" --rate "$RATE" > /dev/null 2>&1
wait $REC
grep -E "register OK|YOLO-World\] [0-9]" "$OUT.node.log" | head -3 || true
echo "==> Overlay: $OUT.mp4   Poses: ${OUT}_poses.csv   Log: $OUT.node.log"
