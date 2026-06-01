#!/usr/bin/env python3
"""
Semi-supervised pseudo-labeling for whale detection.

Uses a trained teacher model to label unlabeled CETI footage.
All pseudo-labels require human review before training (CETI policy).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def pseudo_label_directory(
    teacher_weights: Path,
    unlabeled_dir: Path,
    output_dir: Path,
    confidence: float = 0.85,
    min_box_area: int = 400,
) -> dict:
    """
    Run teacher model on unlabeled images and save pseudo-labels for review.

    Returns statistics dict.
    """
    from ultralytics import YOLO

    model = YOLO(str(teacher_weights))
    out_img = output_dir / "images"
    out_lbl = output_dir / "labels"
    review = output_dir / "review_queue.json"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "labeled": 0, "skipped_low_conf": 0, "skipped_small": 0}
    review_items = []

    extensions = {".jpg", ".jpeg", ".png"}
    images = [f for f in sorted(unlabeled_dir.rglob("*")) if f.suffix.lower() in extensions]

    for img_path in images:
        stats["processed"] += 1
        results = model(str(img_path), conf=confidence, verbose=False)

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        lines = []
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                if area < min_box_area:
                    stats["skipped_small"] += 1
                    continue

                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                detections.append({"bbox": [float(x1), float(y1), float(x2), float(y2)], "conf": conf, "cls": cls_id})

        if lines:
            out_name = img_path.name
            cv2.imwrite(str(out_img / out_name), img)
            (out_lbl / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n")
            review_items.append({
                "image": out_name,
                "source": str(img_path),
                "detections": detections,
                "status": "pending_review",
            })
            stats["labeled"] += 1
        else:
            stats["skipped_low_conf"] += 1

    review.write_text(json.dumps(review_items, indent=2))
    return stats


def main():
    parser = argparse.ArgumentParser(description="Semi-supervised whale pseudo-labeling")
    parser.add_argument("--teacher", type=str, required=True, help="Teacher model weights (.pt)")
    parser.add_argument("--unlabeled", type=str, required=True, help="Directory of unlabeled images")
    parser.add_argument("--output", type=str, default="./data/whale/pseudo_labels")
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--min-box-area", type=int, default=400)
    args = parser.parse_args()

    stats = pseudo_label_directory(
        Path(args.teacher),
        Path(args.unlabeled),
        Path(args.output),
        confidence=args.confidence,
        min_box_area=args.min_box_area,
    )

    print("=" * 60)
    print("Pseudo-Labeling Results")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    print("\nIMPORTANT: All pseudo-labels require human review before training.")
    print(f"Review queue: {Path(args.output) / 'review_queue.json'}")
    print("After review, move approved labels to data/whale/yolo/train/")


if __name__ == "__main__":
    main()
