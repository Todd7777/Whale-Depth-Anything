#!/usr/bin/env bash
# Phase 1: download maximum practical PUBLIC underwater RGB, then build train/val lists
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${REPO_ROOT}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3

# Override caps via env, e.g. EUVP_MAX=3000 AQUA20_MAX=2000
export EUVP_MAX="${EUVP_MAX:-2500}"
export AQUA20_MAX="${AQUA20_MAX:-1500}"
export DAVIS_RATE="${DAVIS_RATE:-2}"
export DAVIS_MAX="${DAVIS_MAX:-600}"

echo "============================================"
echo " CETI Phase 1 — Online Underwater Data"
echo "============================================"
echo "  EUVP_MAX=$EUVP_MAX  AQUA20_MAX=$AQUA20_MAX"
echo "  DAVIS_RATE=$DAVIS_RATE  DAVIS_MAX=$DAVIS_MAX"
echo ""

"$PYTHON" -c "
import sys
sys.path.insert(0, '$REPO_ROOT')
from ceti.data_curation.download_online import download_all_online, write_source_stats, PHASE1_DEFAULTS
from ceti.data_curation.underwater_real import build_train_val_lists, write_manifest
from ceti.utils.device import configure_compute
from pathlib import Path

configure_compute()
stats = download_all_online(
    Path('$REPO_ROOT/data/underwater_field/rgb'),
    euvp_max=int('$EUVP_MAX'),
    aqua20_max=int('$AQUA20_MAX'),
    davis_sample_rate=int('$DAVIS_RATE'),
    davis_max_per_video=int('$DAVIS_MAX'),
)
by_prefix = write_source_stats(Path('$REPO_ROOT/data/underwater_field/rgb'), stats)
print('By source prefix:', by_prefix)
print('Total RGB:', stats.get('total_rgb', 0))

root = Path('$REPO_ROOT')
write_manifest(root / 'data/underwater_field/rgb', root / 'data/underwater_field/manifest.jsonl', stats)
train_n, val_n = build_train_val_lists(
    root / 'data/underwater_field/rgb',
    root / 'ceti/data/underwater_field_train.txt',
    root / 'ceti/data/underwater_field_val.txt',
)
build_train_val_lists(
    root / 'data/underwater_field/rgb',
    root / 'ceti/data/whale_depth_train.txt',
    root / 'ceti/data/whale_depth_val.txt',
)
print(f'Train/val: {train_n} / {val_n}')
if train_n < 1000:
    print('WARNING: fewer than 1000 train images — check network/HF errors above')
    sys.exit(1)
"

echo ""
echo "Data ready. Next:"
echo "  python ceti/depth/train_whale_depth.py --config ceti/configs/whale_depth_phase1.yaml"
echo "  bash ceti/scripts/run_phase1_train.sh"
