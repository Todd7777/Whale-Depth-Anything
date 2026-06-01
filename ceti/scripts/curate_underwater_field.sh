#!/usr/bin/env bash
# Curate real underwater RGB for CETI (DAVIS + HF datasets + field drops)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${REPO_ROOT}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3

# Phase-1 defaults (override via env). For legacy small run: PHASE1=0 AQUA20_MAX=400
PHASE1="${PHASE1:-1}"
EUVP_MAX="${EUVP_MAX:-2500}"
AQUA20_MAX="${AQUA20_MAX:-1500}"
DAVIS_RATE="${DAVIS_RATE:-2}"
DAVIS_MAX="${DAVIS_MAX:-600}"

echo "============================================"
echo " CETI — Underwater Imagery Curation"
echo "============================================"

if [ "$PHASE1" = "1" ]; then
  export EUVP_MAX AQUA20_MAX DAVIS_RATE DAVIS_MAX
  exec bash "$(dirname "$0")/download_all_online_data.sh"
fi

"$PYTHON" -c "
import sys
sys.path.insert(0, '$REPO_ROOT')
from ceti.data_curation.underwater_real import curate_all
from ceti.utils.device import configure_compute

configure_compute()
stats = curate_all(phase1_online=False, download_hf=True, aqua20_max=int('$AQUA20_MAX'))
for k, v in stats.items():
    print(f'  {k}: {v}')
if stats.get('total_rgb', 0) < 50:
    sys.exit(1)
"

echo "Next: bash ceti/scripts/run_phase1_train.sh"
