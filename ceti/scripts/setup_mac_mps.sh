#!/usr/bin/env bash
# One-time Mac setup: venv, deps, checkpoints, MPS verification (M5 Max 128GB)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "============================================"
echo " Whale-Depth-Anything — Mac M5 Max Setup"
echo "============================================"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "WARNING: This script targets macOS (Apple Silicon). Continuing anyway."
fi

# 128GB hint for logging (override if needed)
export CETI_UNIFIED_MEMORY_GB="${CETI_UNIFIED_MEMORY_GB:-128}"
export CETI_DEVICE="${CETI_DEVICE:-mps}"
export CETI_REQUIRE_MPS="${CETI_REQUIRE_MPS:-1}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
PIP="${REPO_ROOT}/.venv/bin/pip"

if [ ! -x "$PYTHON" ]; then
  echo "Creating venv…"
  python3 -m venv .venv
fi

"$PIP" install -U pip wheel
"$PIP" install -r requirements.txt
"$PIP" install -r ceti/requirements.txt

# PyTorch with Metal (Apple Silicon)
"$PIP" install -U torch torchvision

# Required for download_checkpoints.sh (must be in venv, not system python3)
"$PIP" install -U "huggingface_hub>=0.20.0"

if [ -f ceti/scripts/download_checkpoints.sh ]; then
  bash ceti/scripts/download_checkpoints.sh || true
fi

"$PYTHON" ceti/scripts/verify_mps.py
"$PYTHON" ceti/scripts/smoke_test.py

echo ""
echo "Setup complete. Full training:"
echo "  bash ceti/scripts/train_mac_full.sh"
echo ""
echo "Or phase-1 online bootstrap only:"
echo "  bash ceti/scripts/download_all_online_data.sh"
echo "  CETI_DEVICE=mps python ceti/depth/train_whale_depth.py --config ceti/configs/whale_depth_m5max_128gb.yaml"
