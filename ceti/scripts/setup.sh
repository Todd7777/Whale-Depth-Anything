#!/usr/bin/env bash
# CETI project setup script
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "============================================"
echo " CETI Depth & Whale Perception — Setup"
echo "============================================"

# Python venv (optional)
if [ ! -d ".venv" ]; then
    echo "[1/5] Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "[2/5] Installing base dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -r ceti/requirements.txt

echo "[3/5] Creating directory structure..."
mkdir -p checkpoints
mkdir -p data/{flsea,squid,ceti_lab,ceti_field}
mkdir -p data/whale/{yolo,bootstrap,raw_frames,pseudo_labels}
mkdir -p ceti/{outputs,checkpoints,data}
mkdir -p ceti/data

echo "[4/5] Downloading checkpoints..."
bash ceti/scripts/download_checkpoints.sh || echo "  (checkpoint download skipped — run manually if needed)"

echo "[5/5] Creating placeholder file lists..."
for ds in flsea squid ceti_lab; do
    for split in train test; do
        f="ceti/data/${ds}_${split}.txt"
        if [ ! -f "$f" ]; then
            echo "# Placeholder — run prepare_underwater_data.sh after downloading ${ds}" > "$f"
        fi
    done
done

# Bootstrap whale demo data
python ceti/whale/data_curation/download_public.py --demo || true

# Real underwater field imagery (DAVIS + HF datasets)
bash ceti/scripts/curate_underwater_field.sh 2>/dev/null || true

echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. bash ceti/scripts/download_checkpoints.sh   # if not done"
echo "  2. python ceti/scripts/smoke_test.py           # verify installation"
echo "  3. bash ceti/scripts/prove_pipeline.sh --skip-metric-train  # prove UW RGB→depth"
echo "  4. Read ceti/README.md for full workflow"
echo ""
