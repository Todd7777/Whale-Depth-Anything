#!/usr/bin/env python3
"""
Download and prepare public whale detection datasets for bootstrap training.

Since CETI has no pre-existing whale visual annotations, we bootstrap from:
  - Beluga-5k (surface cetacean photos, bounding boxes)
  - Whales from Space (satellite imagery, bounding boxes)

These provide general cetacean detection priors before CETI-specific fine-tuning.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parents[3]
BOOTSTRAP_DIR = REPO_ROOT / "data/whale/bootstrap"


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  Already exists: {dest}")
        return True

    print(f"  Downloading {desc or url}...")
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Saved to {dest}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def setup_beluga5k(output_dir: Path) -> None:
    """
    Beluga-5k dataset setup instructions.

    The dataset must be downloaded manually from:
    https://github.com/parhamap/SSS_Beluga_Whales_Dataset

    Expected structure after download:
        beluga5k/
          images/
          annotations/  (COCO or custom format)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    readme = output_dir / "DOWNLOAD_INSTRUCTIONS.md"
    readme.write_text("""# Beluga-5k Dataset

## Download
1. Visit: https://github.com/parhamap/SSS_Beluga_Whales_Dataset
2. Download images and annotations
3. Place in this directory:
   ```
   beluga5k/
     images/
     annotations/
   ```

## Convert to YOLO
After download, run:
```bash
python ceti/whale/data_curation/convert_annotations.py \\
  --beluga5k ./data/whale/bootstrap/beluga5k/ \\
  --output ./data/whale/bootstrap/beluga5k_yolo/
```

## Citation
Nouri et al., "Machine-Learning Approach for Automatic Detection of Wild Beluga Whales
from Hand-Held Camera Pictures", Remote Sensing, 2022.
""")
    print(f"  Beluga-5k instructions written to {readme}")


def setup_whales_from_space(output_dir: Path) -> None:
    """
    Whales from Space dataset setup instructions.

    Download from NERC UK Polar Data Centre:
    https://doi.org/10.5285/C1AFE32C-493C-4DC7-AF9F-649593B97B2C
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    readme = output_dir / "DOWNLOAD_INSTRUCTIONS.md"
    readme.write_text("""# Whales from Space Dataset

## Download
1. Visit: https://data.bas.ac.uk/full-record.php?id=GB/NERC/BAS/PDC/01482
2. Download "Whales from space dataset: Box and point shapefiles"
3. Download "Whales from space dataset: Image chips"
4. Place shapefiles and image chips in this directory

## Convert to YOLO
```bash
python ceti/whale/data_curation/convert_annotations.py \\
  --whales-from-space ./data/whale/bootstrap/whales_from_space/ \\
  --output ./data/whale/bootstrap/whales_from_space_yolo/
```

## Notes
- Satellite imagery (top-down) differs from CETI aerial/drone perspective
- Use for general whale-like object detection pre-training only
- Remap class IDs: all whale species → class 1 (whale_surface)

## Citation
Cubaynes & Fretwell, "Whales from space dataset", Scientific Data, 2022.
""")
    print(f"  Whales-from-Space instructions written to {readme}")


def create_demo_dataset(output_dir: Path) -> None:
    """
    Create a minimal demo YOLO dataset from Depth Anything example assets
    so the training pipeline can be validated end-to-end.
    """
    examples = REPO_ROOT / "assets/examples"
    if not examples.exists():
        print("  No example assets found for demo dataset")
        return

    train_img = output_dir / "train/images"
    train_lbl = output_dir / "train/labels"
    val_img = output_dir / "val/images"
    val_lbl = output_dir / "val/labels"

    for d in [train_img, train_lbl, val_img, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    images = sorted(examples.glob("*.jpg")) + sorted(examples.glob("*.png"))
    for i, img_path in enumerate(images):
        split_img = train_img if i % 5 != 0 else val_img
        split_lbl = train_lbl if i % 5 != 0 else val_lbl
        shutil.copy2(img_path, split_img / img_path.name)
        # Placeholder label (full-frame, class 1) — replace with real annotations
        (split_lbl / f"{img_path.stem}.txt").write_text("1 0.5 0.5 0.8 0.6\n")

    print(f"  Demo dataset created: {len(images)} images in {output_dir}")
    print("  WARNING: Demo labels are placeholders — not for production training")


def main():
    parser = argparse.ArgumentParser(description="Download public whale datasets")
    parser.add_argument("--beluga", action="store_true", help="Setup Beluga-5k")
    parser.add_argument("--whales-from-space", action="store_true", help="Setup Whales from Space")
    parser.add_argument("--demo", action="store_true", help="Create demo dataset from example assets")
    parser.add_argument("--all", action="store_true", help="Setup all sources")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output = Path(args.output) if args.output else BOOTSTRAP_DIR

    if args.all or args.beluga:
        print("\n[Beluga-5k]")
        setup_beluga5k(output / "beluga5k")

    if args.all or getattr(args, "whales_from_space", False) or args.whales_from_space:
        print("\n[Whales from Space]")
        setup_whales_from_space(output / "whales_from_space")

    if args.all or args.demo:
        print("\n[Demo Dataset]")
        create_demo_dataset(REPO_ROOT / "data/whale/yolo")

    if not any([args.all, args.beluga, args.whales_from_space, args.demo]):
        parser.print_help()


if __name__ == "__main__":
    main()
