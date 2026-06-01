"""
Underwater metric depth dataset loaders for FLSea, SQUID, and CETI lab data.

Integrates with ZoeDepth training pipeline via file-list format:
    <rgb_path> <depth_path> <focal_length>
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_depth(path: str, dataset: str) -> np.ndarray:
    """Load depth map in meters."""
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth: {path}")

    if dataset in ("flsea", "squid", "ceti_lab"):
        # FLSea/SQUID: depth stored as mm uint16 or float meters
        if depth.dtype == np.uint16:
            return depth.astype(np.float32) / 1000.0
        return depth.astype(np.float32)

    return depth.astype(np.float32)


class UnderwaterDepthDataset(Dataset):
    """
    Generic underwater RGB-D dataset driven by a file list.

    File list format (one per line):
        rgb_relative_path depth_relative_path focal_length
    Paths are relative to data_root.
    """

    def __init__(
        self,
        data_root: str,
        filenames_file: str,
        dataset_name: str = "flsea",
        min_depth: float = 0.5,
        max_depth: float = 25.0,
        preprocess_method: str = "combined",
        augment: bool = False,
    ):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.preprocess_method = preprocess_method
        self.augment = augment

        self.samples = []
        with open(filenames_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    self.samples.append((parts[0], parts[1], float(parts[2])))

        if augment:
            from ceti.preprocessing.underwater import UnderwaterAugmentation
            self._aug = UnderwaterAugmentation(p=0.5)
        else:
            self._aug = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        from ceti.preprocessing.underwater import preprocess_underwater

        rgb_rel, depth_rel, focal = self.samples[idx]
        rgb_path = os.path.join(self.data_root, rgb_rel)
        depth_path = os.path.join(self.data_root, depth_rel)

        image = cv2.imread(rgb_path)
        if image is None:
            raise FileNotFoundError(rgb_path)

        if self.preprocess_method != "none":
            image = preprocess_underwater(image, method=self.preprocess_method)

        if self._aug is not None:
            image = self._aug(image)

        depth = _read_depth(depth_path, self.dataset_name)

        # Valid depth mask
        mask = (depth >= self.min_depth) & (depth <= self.max_depth)
        depth = np.clip(depth, self.min_depth, self.max_depth)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_t = torch.from_numpy(image_rgb.transpose(2, 0, 1))
        depth_t = torch.from_numpy(depth).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        return {
            "image": image_t,
            "depth": depth_t,
            "mask": mask_t,
            "focal": focal,
            "dataset": self.dataset_name,
        }


def build_file_list_from_directory(
    data_root: Path,
    rgb_glob: str = "**/rgb/*.jpg",
    depth_glob: str = "**/depth/*.png",
    output_path: Path | None = None,
    train_ratio: float = 0.8,
    default_focal: float = 500.0,
) -> tuple[list[str], list[str]]:
    """
    Scan a dataset directory and build train/test file lists.

    Expects paired rgb/ and depth/ subdirectories with matching filenames.
    """
    rgb_files = sorted(data_root.glob(rgb_glob))
    pairs = []

    for rgb_path in rgb_files:
        stem = rgb_path.stem
        # Try common depth path patterns
        candidates = [
            rgb_path.parent.parent / "depth" / f"{stem}.png",
            rgb_path.parent.parent / "depth" / f"{stem}.npy",
            rgb_path.with_name(f"{stem}_depth.png"),
        ]
        for depth_path in candidates:
            if depth_path.exists():
                rel_rgb = rgb_path.relative_to(data_root)
                rel_depth = depth_path.relative_to(data_root)
                pairs.append(f"{rel_rgb} {rel_depth} {default_focal}")
                break

    split = int(len(pairs) * train_ratio)
    train_lines = pairs[:split]
    test_lines = pairs[split:]

    if output_path:
        train_path = output_path.parent / f"{output_path.stem}_train.txt"
        test_path = output_path.parent / f"{output_path.stem}_test.txt"
        train_path.parent.mkdir(parents=True, exist_ok=True)
        train_path.write_text("\n".join(train_lines) + "\n" if train_lines else "")
        test_path.write_text("\n".join(test_lines) + "\n" if test_lines else "")

    return train_lines, test_lines
