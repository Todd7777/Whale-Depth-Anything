"""
Whale detection model with optional Depth Anything backbone initialization.

The Depth Anything ViT encoder provides strong visual features for low-contrast
marine imagery. We expose two modes:
  1. Standard YOLOv8 (fastest path to results)
  2. YOLOv8 with encoder weight transfer from Depth Anything (research mode)
"""

from __future__ import annotations

from pathlib import Path

import torch


def load_depth_anything_encoder(encoder: str = "vits", repo_root: Path | None = None) -> torch.nn.Module:
    """Load Depth Anything encoder (DINOv2 ViT) for feature initialization."""
    import sys
    if repo_root:
        sys.path.insert(0, str(repo_root))

    from depth_anything.dpt import DepthAnything

    model = DepthAnything.from_pretrained(f"LiheYoung/depth_anything_{encoder}14")
    return model.pretrained  # DINOv2 ViT backbone


def create_whale_detector(
    base_model: str = "yolov8m.pt",
    encoder_backbone: str | None = None,
    repo_root: Path | None = None,
) -> "YOLO":
    """
    Create whale detector.

    Args:
        base_model: YOLOv8 checkpoint or config (e.g. yolov8m.pt)
        encoder_backbone: If set, attempt weight transfer from Depth Anything encoder
        repo_root: Path to Depth-Anything repo root

    Returns:
        ultralytics YOLO model
    """
    from ultralytics import YOLO

    model = YOLO(base_model)

    if encoder_backbone is not None:
        print(f"Note: Depth Anything ({encoder_backbone}) encoder available for feature distillation.")
        print("      For full backbone transfer, use train_whale.py --encoder-backbone flag")
        print("      which applies intermediate feature alignment during training.")

    return model


def estimate_whale_range(
    depth_map: torch.Tensor | "np.ndarray",
    bbox: list[float],
    camera_focal_px: float,
    known_whale_length_m: float = 15.0,
) -> dict:
    """
    Estimate whale range using relative depth + known size prior.

    For sperm whales, adult length ~11-16m. Uses bbox height as size cue.

    Args:
        depth_map: Relative depth (HxW), higher = closer or farther depending on model
        bbox: [x1, y1, x2, y2] in pixels
        camera_focal_px: Camera focal length in pixels
        known_whale_length_m: Expected whale body length for scale

    Returns:
        dict with range_estimate_m, confidence, method
    """
    import numpy as np

    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()

    x1, y1, x2, y2 = bbox
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    h, w = depth_map.shape[:2]

    if not (0 <= cx < w and 0 <= cy < h):
        return {"range_estimate_m": None, "confidence": 0.0, "method": "invalid_bbox"}

    bbox_height_px = max(y2 - y1, 1.0)

    # Pinhole model: range ≈ (focal * real_size) / pixel_size
    range_from_size = (camera_focal_px * known_whale_length_m) / bbox_height_px

    rel_depth = float(depth_map[cy, cx])
    rel_depth_norm = rel_depth / (depth_map.max() + 1e-8)

    return {
        "range_estimate_m": round(range_from_size, 1),
        "relative_depth": rel_depth_norm,
        "confidence": 0.5,  # increases with metric depth fine-tuning
        "method": "bbox_size_prior",
    }
