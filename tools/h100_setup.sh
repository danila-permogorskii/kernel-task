#!/usr/bin/env bash
# One-time setup on a fresh GPU instance (Verda H100). No sudo needed. ~5-10 min.
#
#   git clone git@github.com:danila-permogorskii/kernel-task.git && cd kernel-task
#   bash tools/h100_setup.sh
#
# Steps: check the GPU/driver -> uv + Python 3.12 venv -> torch matching the driver ->
#        pip CUDA compiler (nvcc) -> CUDA_HOME shim -> build the kernel -> pytest -> kernel check
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1. GPU and driver"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
CUDA_DRV=$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')
echo "driver supports CUDA $CUDA_DRV"
case "$CUDA_DRV" in
  13.[2-9]*|1[4-9].*) TORCH_INDEX=cu132 ;;
  13.[01]*)           TORCH_INDEX=cu130 ;;
  *) echo "driver too old for the CUDA 13 wheels (needs CUDA >= 13.0). Pick another image."; exit 1 ;;
esac
echo "using torch wheels: $TORCH_INDEX"

echo "== 2. uv and the venv (Python 3.12, with headers for the extension build)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.12 .venv
source .venv/bin/activate

echo "== 3. packages"
uv pip install "torch==2.14.0" --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
uv pip install -e '.[test]'
CTK=$(python -c "import importlib.metadata as m; print(m.version('cuda-toolkit'))")
uv pip install "cuda-toolkit[nvcc,cccl]==$CTK" ninja
echo "cuda-toolkit $CTK (same version as the torch wheel's runtime)"

echo "== 4. CUDA_HOME shim and the kernel build (first build ~1 min)"
bash tools/make_cuda_home.sh
python -c "from factorized_inference.tr_kernel import load_extension; load_extension(); print('kernel built')"

echo "== 5. tests"
pytest -q
python tools/check_kernel.py

echo
echo "SETUP OK. Next:  bash tools/h100_session.sh"
