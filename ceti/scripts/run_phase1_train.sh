#!/usr/bin/env bash
# Phase 1: download online data (if needed) + train relative depth on M5 Max (MPS)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${REPO_ROOT}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3

export CETI_DEVICE="${CETI_DEVICE:-mps}"
export CETI_REQUIRE_MPS="${CETI_REQUIRE_MPS:-1}"
export CETI_UNIFIED_MEMORY_GB="${CETI_UNIFIED_MEMORY_GB:-128}"

"$PYTHON" ceti/scripts/verify_mps.py

TRAIN_LIST="${REPO_ROOT}/ceti/data/whale_depth_train.txt"
if [ ! -f "$TRAIN_LIST" ] || [ "$(wc -l < "$TRAIN_LIST" | tr -d ' ')" -lt 500 ]; then
  echo "Training list small or missing — downloading online data first…"
  bash ceti/scripts/download_all_online_data.sh
fi

echo "============================================"
echo " Phase 1 Training — underwater relative depth"
echo "============================================"

"$PYTHON" ceti/depth/train_whale_depth.py \
  --config ceti/configs/whale_depth_phase1.yaml \
  "$@"

echo ""
echo "Inference:"
echo "  python ceti/depth/infer_robot.py \\"
echo "    --encoder vitl \\"
echo "    --depth-checkpoint checkpoints/ceti_phase1_online/best.pt \\"
echo "    --input assets/examples_video/davis_seasnake.mp4 \\"
echo "    --underwater-preprocess"
