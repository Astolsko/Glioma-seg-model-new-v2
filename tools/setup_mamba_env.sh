#!/usr/bin/env bash
#
# Recreate the `mamba` conda env: a copy of `pytorch2` (same Python and the
# same pinned package versions) with ONLY torch and what torch/mamba need
# swapped, so the SegMamba encoder's CUDA kernels can load.
#
#   tools/setup_mamba_env.sh
#
# Why not newer torch: every mamba_ssm / causal_conv1d wheel after
# v2.2.4 / v1.5.0.post8 was compiled on Ubuntu 22.04 and needs glibc >= 2.32;
# this workstation is Ubuntu 20.04 (glibc 2.31). Those two releases were built
# on 20.04 (their .so files need only GLIBC_2.14) and support torch up to 2.6.
# PyPI's torch 2.6.0 is cu124 with the pre-C++11 ABI -> the cxx11abiFALSE wheels.
#
# Differences from pytorch2 after this script (pip freeze diff):
#   torch 2.11.0+cu128 -> 2.6.0 (cu124), torchvision 0.26.0 -> 0.21.0,
#   triton 3.6.0 -> 3.2.0, sympy 1.14.0 -> 1.13.1, nvidia-*-cu12 12.8 -> 12.4
#   series (exactly torch 2.6's pins), minus torch 2.11-only cuda-bindings,
#   cuda-pathfinder, cuda-toolkit, nvidia-cufile, nvidia-nvshmem;
#   plus mamba_ssm 2.2.4, causal_conv1d 1.5.0.post8, einops, ninja,
#   transformers 4.46.3 (imported by mamba_ssm) and its deps.
set -euo pipefail
export CONDA_NUMBER_CHANNEL_NOTICES=0     # this conda's notices cache is corrupt
CONDA=${CONDA:-/home/user/miniforge3/bin/conda}
SRC=/DATA/conda_envs/pytorch2
DST=/DATA/conda_envs/mamba
UV=$SRC/bin/uv
WORK=$(mktemp -d)

"$CONDA" create -n mamba python=3.10.0 pip -y

# 1. torch 2.6 and mamba_ssm's Python deps, resolved.
"$UV" pip install --python "$DST/bin/python" torch==2.6.0 torchvision==0.21.0 \
    transformers==4.46.3 einops==0.8.1 ninja==1.11.1.4 numpy==1.23.5

# 2. every other pytorch2 pin, exact and unresolved (pytorch2 itself pairs
#    medpy 0.5.2 with numpy 1.23.5, which medpy's metadata forbids).
"$SRC/bin/python" -m pip freeze | grep -v "^-e" \
  | grep -v -E "^(torch|torchvision|torchaudio|triton|sympy|transformers|einops|ninja|cuda-bindings|cuda-pathfinder|cuda-toolkit|nvidia-[a-z0-9-]+)==" \
  > "$WORK/pins.txt"
"$UV" pip install --python "$DST/bin/python" --no-deps -r "$WORK/pins.txt"

# 3. the SegMamba kernels.
BASE=https://github.com/state-spaces/mamba/releases/download/v2.2.4
CC1D=https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8
"$DST/bin/python" -m pip install --no-deps \
    "$CC1D/causal_conv1d-1.5.0.post8+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
    "$BASE/mamba_ssm-2.2.4+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# 4. verify: kernels import and agree with the reference scan.
"$DST/bin/python" - <<'PY'
import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
torch.manual_seed(0)
u = torch.randn(2, 8, 64, device="cuda"); dt = torch.rand(2, 8, 64, device="cuda")
A = -torch.rand(8, 4, device="cuda"); B = torch.randn(2, 4, 64, device="cuda"); C = torch.randn(2, 4, 64, device="cuda")
err = (selective_scan_fn(u, dt, A, B, C, delta_softplus=True)
       - selective_scan_ref(u, dt, A, B, C, delta_softplus=True)).abs().max().item()
print(f"torch {torch.__version__} | selective_scan max|cuda-ref| = {err:.2e}")
assert err < 1e-4
PY
rm -rf "$WORK"
echo "mamba env ready: conda activate mamba"
