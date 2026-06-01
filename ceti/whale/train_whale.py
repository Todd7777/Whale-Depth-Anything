#!/usr/bin/env python3
"""
Train CETI whale detector on YOLO format data.

Usage:
    python ceti/whale/train_whale.py --data ceti/configs/whale_detection.yaml
    python ceti/whale/train_whale.py --data ./data/whale/yolo/dataset.yaml --epochs 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Keep ultralytics config inside the repo (writable in CI/sandbox)
import os
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics"))


def main():
    parser = argparse.ArgumentParser(description="CETI whale detection training")
    parser.add_argument("--data", type=str, default="ceti/configs/whale_detection.yaml")
    parser.add_argument("--model", type=str, default=None, help="Base YOLO model (overrides config)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--encoder-backbone", type=str, default=None, choices=["vits", "vitb", "vitl"])
    parser.add_argument("--project", type=str, default="ceti/checkpoints")
    parser.add_argument("--name", type=str, default="whale_detector")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    repo_root = REPO_ROOT
    config_path = Path(args.data)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Build or locate YOLO dataset yaml
    train_dir = Path(cfg["train"])
    val_dir = Path(cfg["val"])
    if not train_dir.is_absolute():
        train_dir = repo_root / train_dir
        val_dir = repo_root / val_dir

    dataset_yaml = repo_root / "data/whale/yolo/dataset.yaml"
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)

    from ceti.whale.dataset import create_yolo_dataset_yaml, validate_yolo_split

    create_yolo_dataset_yaml(
        dataset_yaml,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=Path(cfg["test"]) if cfg.get("test") else None,
        class_names=cfg.get("names"),
    )

    train_stats = validate_yolo_split(train_dir)
    val_stats = validate_yolo_split(val_dir)

    print("=" * 60)
    print("CETI Whale Detection Training")
    print("=" * 60)
    print(f"  Train: {train_stats['images']} images, {train_stats['boxes']} boxes")
    print(f"  Val:   {val_stats['images']} images, {val_stats['boxes']} boxes")
    if train_stats["images"] == 0:
        print("\nERROR: No training images found.")
        print("Run data curation first:")
        print("  python ceti/whale/data_curation/download_public.py --all")
        print("  python ceti/whale/data_curation/extract_frames.py --video-dir ./data/ceti_field/")
        return

    if train_stats["missing_labels"]:
        print(f"  WARNING: {len(train_stats['missing_labels'])} images missing labels")

    model_name = args.model or cfg.get("model", "yolov8m.pt")
    if args.encoder_backbone or cfg.get("encoder_backbone"):
        enc = args.encoder_backbone or cfg["encoder_backbone"]
        print(f"  Encoder backbone: Depth Anything {enc} (feature init enabled)")

    from ultralytics import YOLO

    if args.resume:
        model = YOLO(args.resume)
    else:
        model = YOLO(model_name)

    train_kwargs = {
        "data": str(dataset_yaml),
        "epochs": args.epochs or cfg.get("epochs", 100),
        "batch": args.batch_size or cfg.get("batch_size", 16),
        "imgsz": args.img_size or cfg.get("img_size", 640),
        "project": str(repo_root / args.project),
        "name": args.name,
        "patience": cfg.get("patience", 20),
        "lr0": cfg.get("lr0", 0.001),
        "lrf": cfg.get("lrf", 0.01),
        "momentum": cfg.get("momentum", 0.937),
        "weight_decay": cfg.get("weight_decay", 0.0005),
        "hsv_h": cfg.get("hsv_h", 0.015),
        "hsv_s": cfg.get("hsv_s", 0.7),
        "hsv_v": cfg.get("hsv_v", 0.4),
        "degrees": cfg.get("degrees", 10.0),
        "fliplr": cfg.get("fliplr", 0.5),
        "mosaic": cfg.get("mosaic", 1.0),
        "mixup": cfg.get("mixup", 0.1),
        "exist_ok": True,
        "pretrained": True,
    }

    print(f"  Model:  {model_name}")
    print(f"  Epochs: {train_kwargs['epochs']}")
    print("=" * 60)

    results = model.train(**train_kwargs)
    print(f"\nTraining complete. Best weights: {results.save_dir}/weights/best.pt")
    return results


if __name__ == "__main__":
    main()
