#!/usr/bin/env bash
# FINAL bulletproof M5 Max 128GB: data on disk → teacher cache → student-only train (MPS)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${REPO_ROOT}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3

export CETI_DEVICE=mps
export CETI_REQUIRE_MPS=1
export CETI_UNIFIED_MEMORY_GB=128
export CETI_SKIP_GIT_PULL=1
export CETI_CACHE_TEACHER=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export CETI_TRAIN_CONFIG="${CETI_TRAIN_CONFIG:-ceti/configs/whale_depth_m5max_128gb.yaml}"

echo "============================================"
echo " CETI — M5 Max BULLETPROOF TRAIN"
echo "============================================"
echo "  Config: $CETI_TRAIN_CONFIG"
echo "  Device: MPS (Metal)  workers=0  cache_teacher=ON"
echo ""

"$PYTHON" ceti/scripts/verify_mps.py

bash ceti/scripts/ensure_training_data.sh

N=$("$PYTHON" -c "
from pathlib import Path
import sys
sys.path.insert(0, '$REPO_ROOT')
from ceti.depth.whale_depth_dataset import load_image_paths
print(sum(1 for p in load_image_paths('$REPO_ROOT/ceti/data/whale_depth_train.txt') if p.is_file()))
")
echo "  Train images on disk: $N"
if [ "$N" -lt 500 ]; then
  echo "ERROR: need >= 500 images. ensure_training_data failed."
  exit 1
fi

bash ceti/scripts/train_mac_full.sh

echo ""
echo "============================================"
echo " DONE"
echo "  best.pt → checkpoints/ceti_whale_depth/best.pt"
echo "  proof   → ceti/outputs/proof/ (after train)"
echo "============================================"
