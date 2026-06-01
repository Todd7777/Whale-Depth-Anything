"""
Generate synthetic underwater training pairs from in-air RGB-D data.

Applies a simplified underwater image formation model (Jaffe-McGlamery)
to Hypersim or other RGB-D sources, following arxiv:2507.02148 methodology.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def underwater_image_formation(
    image: np.ndarray,
    depth: np.ndarray,
    beta_d: float = 0.15,
    beta_b: float = 0.08,
    ambient: tuple[float, float, float] = (10.0, 35.0, 55.0),
) -> np.ndarray:
    """
    Simplified underwater image formation: I = J * exp(-beta_d * z) + A * (1 - exp(-beta_b * z))

    Args:
        image: RGB float [0,1] or uint8 [0,255]
        depth: Depth map in meters (same HxW as image)
        beta_d: Direct attenuation coefficient
        beta_b: Backscatter coefficient
        ambient: Background water color (RGB)

    Returns:
        Synthetic underwater RGB image (uint8).
    """
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)

    depth = depth.astype(np.float32)
    if depth.ndim == 2:
        depth = depth[:, :, np.newaxis]

    ambient = np.array(ambient, dtype=np.float32) / 255.0

    transmission = np.exp(-beta_d * depth)
    backscatter = ambient * (1.0 - np.exp(-beta_b * depth))

    underwater = img * transmission + backscatter
    return np.clip(underwater * 255, 0, 255).astype(np.uint8)


def process_hypersim_scene(
    rgb_path: Path,
    depth_path: Path,
    output_dir: Path,
    variants: int = 3,
) -> list[tuple[Path, Path]]:
    """
    Generate multiple underwater variants from one Hypersim RGB-D pair.

    Returns list of (rgb_out, depth_out) paths written.
    """
    rgb = cv2.imread(str(rgb_path))
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        return []

    if depth.dtype == np.uint16:
        depth_m = depth.astype(np.float32) / 1000.0
    else:
        depth_m = depth.astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = rgb_path.stem
    written = []

    for i in range(variants):
        beta_d = np.random.uniform(0.08, 0.25)
        beta_b = np.random.uniform(0.04, 0.15)
        uw_rgb = underwater_image_formation(rgb, depth_m, beta_d=beta_d, beta_b=beta_b)

        rgb_out = output_dir / f"{stem}_uw_{i:02d}.jpg"
        depth_out = output_dir / f"{stem}_uw_{i:02d}_depth.png"

        cv2.imwrite(str(rgb_out), uw_rgb)
        cv2.imwrite(str(depth_out), (depth_m * 1000).astype(np.uint16))
        written.append((rgb_out, depth_out))

    return written


def build_file_list(output_dir: Path, list_path: Path, focal_length: float = 886.81) -> int:
    """Build ZoeDepth-format file list from generated pairs."""
    pairs = sorted(output_dir.glob("*_uw_*_depth.png"))
    lines = []
    for depth_path in pairs:
        rgb_path = Path(str(depth_path).replace("_depth.png", ".jpg"))
        if rgb_path.exists():
            rel_rgb = rgb_path.relative_to(output_dir.parent.parent)
            rel_depth = depth_path.relative_to(output_dir.parent.parent)
            lines.append(f"{rel_rgb} {rel_depth} {focal_length}")

    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(lines) + "\n" if lines else "")
    return len(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic underwater RGB-D")
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--file-list", type=Path, default=None)
    args = parser.parse_args()

    count = 0
    for rgb_path in sorted(args.rgb_dir.glob("*.jpg")):
        depth_path = args.depth_dir / f"{rgb_path.stem}_depth.png"
        if not depth_path.exists():
            depth_path = args.depth_dir / f"{rgb_path.stem}.png"
        if depth_path.exists():
            count += len(process_hypersim_scene(rgb_path, depth_path, args.output_dir, args.variants))

    print(f"Generated {count} underwater image pairs in {args.output_dir}")

    if args.file_list:
        n = build_file_list(args.output_dir, args.file_list)
        print(f"Wrote file list with {n} entries to {args.file_list}")
