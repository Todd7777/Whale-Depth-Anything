#!/usr/bin/env bash
# Prepare underwater depth dataset file lists
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

DATASET=""
DATA_ROOT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --root) DATA_ROOT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$DATASET" ] || [ -z "$DATA_ROOT" ]; then
    echo "Usage: $0 --dataset <flsea|squid|ceti_lab> --root <data_directory>"
    echo ""
    echo "Dataset download instructions:"
    echo ""
    echo "  FLSea:  https://github.com/xahidz/revisiting-dm"
    echo "          Request dataset access, extract to --root"
    echo ""
    echo "  SQUID:  https://csms.haifa.ac.il/profiles/tTreibitz/data-sets/underwater/17/"
    echo "          Free registration required"
    echo ""
    echo "  CETI Lab: Place ROV camera RGB-D pairs in:"
    echo "          <root>/rgb/*.jpg  and  <root>/depth/*.png"
    exit 1
fi

echo "Building file lists for $DATASET from $DATA_ROOT..."

python3 -c "
from pathlib import Path
import sys
sys.path.insert(0, '$REPO_ROOT')
from ceti.depth.underwater_dataset import build_file_list_from_directory

root = Path('$DATA_ROOT')
train, test = build_file_list_from_directory(
    root,
    output_path=Path('$REPO_ROOT/ceti/data/${DATASET}'),
    default_focal=500.0,
)
print(f'Train: {len(train)} pairs')
print(f'Test:  {len(test)} pairs')
print(f'File lists written to ceti/data/${DATASET}_train.txt and ceti/data/${DATASET}_test.txt')
"

echo "Done."
