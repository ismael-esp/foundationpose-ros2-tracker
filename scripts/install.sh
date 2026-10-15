#!/usr/bin/env bash
# One-shot installer: conda env + FoundationPose (patched) + CUDA extensions + weights.
#
#   bash scripts/install.sh                # env name "fp_ros", PyTorch cu128 (RTX 50xx / Blackwell OK)
#   ENV_NAME=myenv TORCH_CUDA=cu121 bash scripts/install.sh
#
# Prerequisites (system): NVIDIA driver >= 570, CUDA toolkit 12.8 (nvcc), ROS 2 Humble,
# conda/miniconda, git, cmake, a C++17 compiler, libboost-system/program-options-dev.
set -eo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-fp_ros}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"
FP_COMMIT="e3d597b"   # NVlabs/FoundationPose commit the patch was made against

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"

echo "==> [1/7] PyTorch ($TORCH_CUDA) + Python deps"
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
pip install -r "$ROOT/requirements.txt"
pip install "numpy<2"                                   # system cv_bridge / ROS Humble need NumPy 1.x
pip install "git+https://github.com/ultralytics/CLIP.git"  # YOLO-World text encoder
conda install -y -c conda-forge eigen=3.4 pybind11 boost-cpp
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

echo "==> [2/7] FoundationPose @ $FP_COMMIT + Blackwell/CUDA-12.8 patch"
cd "$ROOT"
if [ ! -d FoundationPose ]; then
  git clone https://github.com/NVlabs/FoundationPose.git
  git -C FoundationPose checkout "$FP_COMMIT"
  git -C FoundationPose apply "$ROOT/patches/foundationpose_blackwell_cu128.patch"
fi

echo "==> [3/7] Pretrained FoundationPose weights"
mkdir -p FoundationPose/weights
[ -d FoundationPose/weights/2023-10-28-18-33-37 ] || \
  gdown --folder https://drive.google.com/drive/folders/1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC -O FoundationPose/weights/2023-10-28-18-33-37
[ -d FoundationPose/weights/2024-01-11-20-02-45 ] || \
  gdown --folder https://drive.google.com/drive/folders/12Te_3TELLes5cim1d7F7EBTwUSe7iRBj -O FoundationPose/weights/2024-01-11-20-02-45

echo "==> [4/7] nvdiffrast"
[ -d FoundationPose/nvdiffrast ] || git clone https://github.com/NVlabs/nvdiffrast FoundationPose/nvdiffrast
pip install --no-build-isolation ./FoundationPose/nvdiffrast

echo "==> [5/7] mycpp (C++ / pybind11)"
cd "$ROOT/FoundationPose/mycpp"
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
make -j"$(nproc)"
cp mycpp*.so "$ROOT/FoundationPose/"                    # Utils.py does a flat `import mycpp`

echo "==> [6/7] mycuda (CUDA extension)"
cd "$ROOT/FoundationPose/bundlesdf/mycuda"
rm -rf build ./*egg* ./*.so
pip install --no-build-isolation -e .

echo "==> [7/7] Activation hook: torch libs on LD_LIBRARY_PATH (nvdiffrast JIT plugin)"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/torch_lib_path.sh" <<'EOF'
#!/bin/bash
TORCH_LIB="$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null)"
if [ -n "$TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
EOF

cd "$ROOT"
echo
echo "Done. YOLO-World (yolov8l-worldv2.pt) and SAM2 (sam2.1_b.pt) weights are downloaded"
echo "automatically by Ultralytics into this folder on the first run."
echo "Verify with:  conda activate $ENV_NAME && python scripts/check_install.py"
