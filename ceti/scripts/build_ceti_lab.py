#!/usr/bin/env python3
"""
Build synthetic underwater RGB-D lab data for pipeline proof (no FLSea download).

Uses Jaffe-McGlamery-style formation from ceti/preprocessing/synthetic_underwater.py
following underwater metric-depth adaptation practice (arxiv:2507.02148).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ceti.depth.underwater_dataset import build_file_list_from_directory
from ceti.preprocessing.synthetic_underwater import underwater_image_formation


def proxy_depth_meters(bgr: np.ndarray, z_near: float = 2.0, z_far: float = 12.0) -> np.ndarray:
    """Smooth proxy geometry (meters) for synthetic underwater rendering."""
    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # Closer where image is brighter / higher in frame (typical open-water prior)
    depth = z_far - (z_far - z_near) * (0.55 * (yy / max(h - 1, 1)) + 0.45 * gray)
    return np.clip(depth, z_near, z_far)


def build_lab(
    source_dir: Path,
    output_root: Path,
    variants: int = 2,
    max_images: int = 10,
    train_ratio: float = 0.8,
) -> tuple[int, int]:
    rgb_out = output_root / "rgb"
    depth_out = output_root / "depth"
    rgb_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    sources = sorted(source_dir.glob("*.jpg")) + sorted(source_dir.glob("*.png"))
    sources = sources[:max_images]
    count = 0

    for src in sources:
        bgr = cv2.imread(str(src))
        if bgr is None:
            continue
        depth_m = proxy_depth_meters(bgr)

        for i in range(variants):
            beta_d = 0.10 + 0.06 * i
            beta_b = 0.05 + 0.04 * i
            uw = underwater_image_formation(bgr, depth_m, beta_d=beta_d, beta_b=beta_b)
            stem = f"{src.stem}_uw{i:02d}"
            cv2.imwrite(str(rgb_out / f"{stem}.jpg"), uw)
            cv2.imwrite(str(depth_out / f"{stem}.png"), (depth_m * 1000).astype(np.uint16))
            count += 1

    list_base = REPO_ROOT / "ceti/data/ceti_lab"
    train_lines, test_lines = build_file_list_from_directory(
        output_root,
        rgb_glob="rgb/*.jpg",
        depth_glob="depth/*.png",
        output_path=list_base,
        train_ratio=train_ratio,
    )
    return len(train_lines), len(test_lines)


def main():
    parser = argparse.ArgumentParser(description="Build CETI synthetic underwater lab RGB-D")
    parser.add_argument("--sources", type=Path, default=REPO_ROOT / "assets/examples")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/ceti_lab")
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=10)
    args = parser.parse_args()

    train_n, test_n = build_lab(args.sources, args.output, args.variants, args.max_images)
    print(f"Synthetic underwater lab: {train_n} train, {test_n} test pairs")
    print(f"Data root: {args.output}")
    print(f"File lists: ceti/data/ceti_lab_train.txt, ceti/data/ceti_lab_test.txt")
    if train_n == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
