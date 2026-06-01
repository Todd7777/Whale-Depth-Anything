#!/usr/bin/env python3
"""
Convert annotations from various formats to YOLO training format.

Supported inputs:
  - CVAT YOLO 1.1 export (obj_train_data/ + train.txt)
  - Beluga-5k COCO annotations
  - Whales from Space shapefiles (via pre-exported CSV)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2


def convert_cvat_yolo_export(export_dir: Path, output_dir: Path) -> int:
    """Convert CVAT YOLO 1.1 export to standard train/images + train/labels layout."""
    obj_dir = export_dir / "obj_train_data"
    if not obj_dir.exists():
        # Try flat export
        obj_dir = export_dir

    out_img = output_dir / "images"
    out_lbl = output_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(obj_dir.glob("*.jpg")) + sorted(obj_dir.glob("*.png")):
        label_path = obj_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        shutil.copy2(img_path, out_img / img_path.name)
        shutil.copy2(label_path, out_lbl / f"{img_path.stem}.txt")
        count += 1

    return count


def convert_coco_beluga(annotations_json: Path, images_dir: Path, output_dir: Path) -> int:
    """Convert Beluga-5k COCO format to YOLO."""
    with open(annotations_json) as f:
        coco = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    id_to_size = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

    # Build category mapping to CETI classes
    cat_map = {}
    for cat in coco["categories"]:
        name = cat["name"].lower()
        if "beluga" in name or "whale" in name:
            cat_map[cat["id"]] = 1  # whale_surface
        else:
            cat_map[cat["id"]] = 1

    out_img = output_dir / "images"
    out_lbl = output_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    # Group annotations by image
    from collections import defaultdict
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    count = 0
    for img_id, anns in anns_by_img.items():
        fname = id_to_file[img_id]
        img_w, img_h = id_to_size[img_id]
        src = images_dir / fname
        if not src.exists():
            continue

        lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            cls_id = cat_map.get(ann["category_id"], 1)
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if lines:
            shutil.copy2(src, out_img / fname)
            (out_lbl / f"{Path(fname).stem}.txt").write_text("\n".join(lines) + "\n")
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Convert annotations to YOLO format")
    parser.add_argument("--cvat-export", type=str, default=None)
    parser.add_argument("--beluga5k", type=str, default=None, help="Beluga-5k root dir")
    parser.add_argument("--whales-from-space", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    output = Path(args.output)
    total = 0

    if args.cvat_export:
        n = convert_cvat_yolo_export(Path(args.cvat_export), output)
        print(f"CVAT export: {n} images converted")
        total += n

    if args.beluga5k:
        root = Path(args.beluga5k)
        ann_file = root / "annotations" / "instances.json"
        if not ann_file.exists():
            ann_file = root / "annotations.json"
        if ann_file.exists():
            n = convert_coco_beluga(ann_file, root / "images", output)
            print(f"Beluga-5k: {n} images converted")
            total += n
        else:
            print(f"Beluga-5k annotations not found in {root}")

    print(f"\nTotal: {total} images → {output}")
    if total > 0:
        print("\nValidate with:")
        print(f"  python -c \"from ceti.whale.dataset import validate_yolo_split; print(validate_yolo_split(Path('{output}')))\"")
        print(f"  from pathlib import Path")


if __name__ == "__main__":
    main()
