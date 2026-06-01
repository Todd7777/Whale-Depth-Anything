#!/usr/bin/env bash
# Download Depth Anything pretrained checkpoints
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CKPT_DIR="$REPO_ROOT/checkpoints"
mkdir -p "$CKPT_DIR"

PYTHON="${REPO_ROOT}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"

download_hf() {
    local repo="$1"
    local filename="$2"
    local dest_name="${3:-$2}"

    local dest="$CKPT_DIR/$dest_name"

    if [ -f "$dest" ]; then
        echo "  ✓ $dest_name (already exists)"
        return 0
    fi

    echo "  ↓ Downloading $filename from $repo → $dest_name..."
    "$PYTHON" -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(repo_id='$repo', filename='$filename')
shutil.copy(path, '$dest')
print(f'  ✓ Saved to $dest')
" || {
        echo "  ✗ Failed — try: huggingface-cli download $repo $filename --local-dir $CKPT_DIR"
        return 1
    }
}

echo "Downloading Depth Anything checkpoints..."
echo ""

# HF model repos store weights as pytorch_model.bin
download_hf "LiheYoung/depth_anything_vits14" "pytorch_model.bin" "depth_anything_vits14.pth" || true
download_hf "LiheYoung/depth_anything_vitb14" "pytorch_model.bin" "depth_anything_vitb14.pth" || true
download_hf "LiheYoung/depth_anything_vitl14" "pytorch_model.bin" "depth_anything_vitl14.pth" || true

# Metric depth checkpoints are in the HF Space repo
download_hf "LiheYoung/Depth-Anything" "checkpoints/depth_anything_metric_depth_indoor.pt" "depth_anything_metric_depth_indoor.pt" || true
download_hf "LiheYoung/Depth-Anything" "checkpoints/depth_anything_metric_depth_outdoor.pt" "depth_anything_metric_depth_outdoor.pt" || true

echo ""
echo "Checkpoint directory: $CKPT_DIR"
ls -lh "$CKPT_DIR"/ 2>/dev/null || true
echo ""
echo "Note: Models also load automatically via DepthAnything.from_pretrained() at runtime."
