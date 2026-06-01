"""CETI whale detection dataset utilities (YOLO format)."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import yaml


CLASS_NAMES = {
    0: "sperm_whale",
    1: "whale_surface",
    2: "whale_partial",
}


def load_yolo_dataset(data_yaml: str | Path) -> dict:
    with open(data_yaml) as f:
        return yaml.safe_load(f)


def yolo_label_to_bbox(
    label_line: str,
    img_w: int,
    img_h: int,
) -> tuple[int, list[float]]:
    """Convert YOLO normalized label to pixel bbox [x1,y1,x2,y2]."""
    parts = label_line.strip().split()
    cls_id = int(parts[0])
    cx, cy, w, h = map(float, parts[1:5])
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return cls_id, [x1, y1, x2, y2]


def bbox_to_yolo_label(
    cls_id: int,
    bbox: list[float],
    img_w: int,
    img_h: int,
) -> str:
    """Convert pixel bbox to YOLO normalized format."""
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def create_yolo_dataset_yaml(
    output_path: Path,
    train_dir: Path,
    val_dir: Path,
    test_dir: Path | None = None,
    class_names: dict | None = None,
) -> Path:
    """Generate dataset.yaml for ultralytics training."""
    names = class_names or CLASS_NAMES
    data = {
        "path": str(output_path.parent.resolve()),
        "train": str(train_dir.resolve()),
        "val": str(val_dir.resolve()),
        "names": names,
    }
    if test_dir:
        data["test"] = str(test_dir.resolve())

    output_path.write_text(yaml.dump(data, default_flow_style=False))
    return output_path


def validate_yolo_split(split_dir: Path) -> dict:
    """Validate a YOLO split directory and return statistics."""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    stats = {"images": 0, "labels": 0, "boxes": 0, "missing_labels": [], "empty_labels": []}

    if not images_dir.exists():
        return stats

    for img_path in sorted(images_dir.glob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        stats["images"] += 1
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            stats["missing_labels"].append(img_path.name)
            continue
        stats["labels"] += 1
        lines = label_path.read_text().strip().split("\n")
        lines = [l for l in lines if l.strip()]
        if not lines:
            stats["empty_labels"].append(img_path.name)
        stats["boxes"] += len(lines)

    return stats


def merge_bootstrap_datasets(sources: list[Path], output_dir: Path, split_ratio: float = 0.85) -> None:
    """
    Merge multiple bootstrap datasets (each in YOLO format) into train/val splits.
    Remaps class IDs to CETI taxonomy where possible.
    """
    import shutil
    import random

    all_images = []
    for src in sources:
        img_dir = src / "images" if (src / "images").exists() else src
        for img in sorted(img_dir.glob("*")):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                label = src / "labels" / f"{img.stem}.txt" if (src / "labels").exists() else None
                all_images.append((img, label))

    random.shuffle(all_images)
    split = int(len(all_images) * split_ratio)

    for split_name, subset in [("train", all_images[:split]), ("val", all_images[split:])]:
        out_img = output_dir / split_name / "images"
        out_lbl = output_dir / split_name / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in subset:
            shutil.copy2(img_path, out_img / img_path.name)
            if label_path and label_path.exists():
                shutil.copy2(label_path, out_lbl / f"{img_path.stem}.txt")

    print(f"Merged {len(all_images)} images → train={split}, val={len(all_images)-split}")
