#!/usr/bin/env python3
"""
Fine-tune Depth Anything metric depth head for underwater environments.

Wraps the ZoeDepth training pipeline with CETI-specific dataset configs
and underwater preprocessing.

Usage:
    cd Depth-Anything/metric_depth
    python ../ceti/depth/train_underwater.py --dataset flsea --epochs 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

# Add metric_depth to path for ZoeDepth imports
REPO_ROOT = Path(__file__).resolve().parents[2]
METRIC_DEPTH = REPO_ROOT / "metric_depth"
sys.path.insert(0, str(METRIC_DEPTH))
sys.path.insert(0, str(REPO_ROOT))


def load_ceti_config(config_path: Path | None = None) -> dict:
    path = config_path or REPO_ROOT / "ceti/configs/underwater_metric.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def register_underwater_dataset(ceti_cfg: dict, dataset: str) -> None:
    """Patch ZoeDepth DATASETS_CONFIG with underwater dataset entry."""
    from zoedepth.utils import config as zoe_config

    _orig_check = zoe_config.check_choices

    def _check_choices(name, value, choices):
        if name == "Dataset" and value in zoe_config.DATASETS_CONFIG:
            return
        return _orig_check(name, value, choices)

    zoe_config.check_choices = _check_choices

    ds_cfg = ceti_cfg["datasets"][dataset]
    zoe_config.DATASETS_CONFIG[dataset] = {
        "dataset": "nyu",
        "min_depth": ds_cfg.get("min_depth", ceti_cfg["min_depth"]),
        "max_depth": ds_cfg.get("max_depth", ceti_cfg["max_depth"]),
        "data_path": ds_cfg["data_path"],
        "gt_path": ds_cfg["data_path"],
        "filenames_file": str(REPO_ROOT / ds_cfg["filenames_file"].lstrip("./")),
        "input_height": ceti_cfg["input_height"],
        "input_width": ceti_cfg["input_width"],
        "data_path_eval": ds_cfg["data_path"],
        "gt_path_eval": ds_cfg["data_path"],
        "filenames_file_eval": str(REPO_ROOT / ds_cfg["filenames_file_eval"].lstrip("./")),
        "min_depth_eval": ds_cfg.get("min_depth", ceti_cfg["min_depth"]),
        "max_depth_eval": ds_cfg.get("max_depth", ceti_cfg["max_depth"]),
        "do_random_rotate": ceti_cfg.get("do_random_rotate", True),
        "degree": ceti_cfg.get("degree", 5.0),
        "do_kb_crop": False,
        "garg_crop": False,
        "eigen_crop": False,
        "avoid_boundary": False,
    }


def main():
    parser = argparse.ArgumentParser(description="CETI underwater metric depth fine-tuning")
    parser.add_argument("--dataset", type=str, default="flsea", choices=["flsea", "squid", "ceti_lab"])
    parser.add_argument("--config", type=str, default=None, help="Path to underwater_metric.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--pretrained-resource", type=str, default="")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training")
    args = parser.parse_args()

    ceti_cfg = load_ceti_config(Path(args.config) if args.config else None)
    register_underwater_dataset(ceti_cfg, args.dataset)

    from ceti.depth.ceti_metric import prepare_metric_depth_env

    prepare_metric_depth_env()

    # Build training command via ZoeDepth API
    from zoedepth.utils.config import get_config
    from zoedepth.utils.misc import count_parameters, parallelize
    from zoedepth.trainers.builder import get_trainer
    from zoedepth.models.builder import build_model
    from zoedepth.data.data_mono import DepthDataLoader
    import torch

    overwrite = {
        "config_version": "zoedepth",
        "save_dir": args.save_dir or ceti_cfg["save_dir"],
        "epochs": args.epochs or ceti_cfg["epochs"],
        "batch_size": args.batch_size or ceti_cfg["batch_size"],
        "pretrained_resource": args.pretrained_resource or f"local::{ceti_cfg['pretrained_metric']}",
        "tags": f"ceti,underwater,{args.dataset}",
        "notes": f"CETI underwater fine-tune on {args.dataset}",
    }

    config = get_config("zoedepth", "train", dataset=args.dataset, **overwrite)

    print("=" * 60)
    print("CETI Underwater Depth Fine-Tuning")
    print("=" * 60)
    print(f"  Dataset:     {args.dataset}")
    print(f"  Depth range: {config.min_depth}–{config.max_depth} m")
    print(f"  Epochs:      {config.epochs}")
    print(f"  Batch size:  {config.batch_size}")
    print(f"  Save dir:    {config.save_dir}")
    print(f"  File list:   {config.filenames_file}")
    print("=" * 60)

    if not Path(config.filenames_file).exists():
        print(f"\nERROR: Training file list not found: {config.filenames_file}")
        print("Run: bash ceti/scripts/prepare_underwater_data.sh --dataset", args.dataset)
        sys.exit(1)

    if args.dry_run:
        print("\nDry run complete — config validated.")
        return

    model = build_model(config)
    model = parallelize(model, config)
    print(f"Model parameters: {count_parameters(model)/1e6:.2f}M")

    train_loader = DepthDataLoader(config, "train").data
    trainer = get_trainer(config)(config, model=model, train_loader=train_loader)
    trainer.train()


if __name__ == "__main__":
    main()
