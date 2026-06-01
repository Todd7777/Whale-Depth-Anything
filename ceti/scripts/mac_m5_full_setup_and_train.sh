#!/usr/bin/env bash
# Whale-Depth-Anything — M5 Max 128GB: setup → checkpoints → smoke → train
# Run from repo root. Set secrets locally (never commit tokens):
#   export HF_TOKEN='hf_...'              # optional, faster HF downloads
#   export GITHUB_TOKEN='ghp_...'         # only for git pull, not needed here
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export CETI_DEVICE="${CETI_DEVICE:-mps}"
export CETI_REQUIRE_MPS="${CETI_REQUIRE_MPS:-1}"
export CETI_UNIFIED_MEMORY_GB="${CETI_UNIFIED_MEMORY_GB:-128}"

echo "============================================"
echo " Whale-Depth-Anything — M5 Max full pipeline"
echo " Repo: $REPO_ROOT"
echo "============================================"

# --- 1) Pull latest (if git repo) ---
if git rev-parse --is-inside-work-tree &>/dev/null; then
  if git remote get-url whale &>/dev/null; then
    echo "[1/6] git pull from whale remote…"
    git pull whale main || git pull origin main || true
  fi
else
  echo "[1/6] Not a git repo — skip pull"
fi

# --- 2) Venv + deps + MPS ---
echo "[2/6] Mac MPS setup…"
bash ceti/scripts/setup_mac_mps.sh

# --- 3) Checkpoints (uses HF_TOKEN if set) ---
echo "[3/6] Download checkpoints…"
bash ceti/scripts/download_checkpoints.sh

# --- 4) Smoke test ---
echo "[4/6] Smoke test…"
"${REPO_ROOT}/.venv/bin/python" ceti/scripts/smoke_test.py

# --- 5) Training data ---
TRAIN_LIST="${REPO_ROOT}/ceti/data/whale_depth_train.txt"
if [ ! -f "$TRAIN_LIST" ] || [ "$(wc -l < "$TRAIN_LIST" | tr -d ' ')" -lt 500 ]; then
  echo "[5/6] Download phase-1 underwater RGB…"
  bash ceti/scripts/download_all_online_data.sh
else
  echo "[5/6] Train list OK ($(wc -l < "$TRAIN_LIST" | tr -d ' ') lines)"
fi

# --- 6) Full training + proof ---
echo "[6/6] Full training (ViT-L, MPS)…"
bash ceti/scripts/train_mac_full.sh

echo ""
echo "============================================"
echo " DONE"
echo "  Checkpoint: checkpoints/ceti_whale_depth/best.pt"
echo "  Proof:      ceti/outputs/proof/"
echo "============================================"
