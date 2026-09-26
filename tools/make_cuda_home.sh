#!/usr/bin/env bash
# Build a CUDA_HOME for torch.utils.cpp_extension out of the pip-installed CUDA 13 toolkit
# (packages cuda-toolkit[nvcc,cccl]). No sudo, same on the laptop (WSL) and the H100 box.
#
#   .cuda_home/bin      -> nvcc, ptxas, ...
#   .cuda_home/include  -> CUDA headers (CCCL under include/cccl)
#   .cuda_home/nvvm     -> the device compiler back end
#   .cuda_home/lib, lib64 -> runtime libraries; lib64/libcudart.so is the link name torch expects
#
# Usage (inside the activated venv, from the repo root):  bash tools/make_cuda_home.sh
set -euo pipefail

NV=$(python -c "import nvidia, os; print(os.path.join(nvidia.__path__[0], 'cu13'))")
[ -x "$NV/bin/nvcc" ] || { echo "nvcc not found in $NV: pip install 'cuda-toolkit[nvcc,cccl]'"; exit 1; }

H="$(pwd)/.cuda_home"
rm -rf "$H"
mkdir -p "$H/lib64"
ln -s "$NV/bin" "$H/bin"
ln -s "$NV/include" "$H/include"
ln -s "$NV/nvvm" "$H/nvvm"
ln -s "$NV/lib" "$H/lib"
for f in "$NV"/lib/*; do ln -s "$f" "$H/lib64/"; done
ln -sf "$NV/lib/libcudart.so.13" "$H/lib64/libcudart.so"

"$H/bin/nvcc" --version | tail -1
echo "CUDA_HOME ready: $H"
