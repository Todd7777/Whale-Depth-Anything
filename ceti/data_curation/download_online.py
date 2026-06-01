"""
Download publicly available underwater RGB from Hugging Face, GitHub, and local video.

Phase-1 bootstrap for CETI depth domain adaptation before CETI field data exists.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# Defaults tuned for a strong first training run without one source dominating
PHASE1_DEFAULTS = {
    "euvp_max": 2500,
    "aqua20_max": 1500,
    "uieb_hf_max": 890,
    "uieb_raw_max": 700,
    "submerged3d_max": 80,
    "davis_sample_rate": 2,
    "davis_max_per_video": 600,
    "funiegan_test": True,
}


def _pil_to_bgr(img) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _save_bgr(bgr: np.ndarray, path: Path, quality: int = 92) -> bool:
    if bgr is None or bgr.size == 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])


def _extract_image_from_row(row: dict, keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key not in row or row[key] is None:
            continue
        val = row[key]
        if hasattr(val, "convert"):
            return _pil_to_bgr(val)
        if isinstance(val, np.ndarray):
            if val.ndim == 3 and val.shape[2] == 3:
                return cv2.cvtColor(val, cv2.COLOR_RGB2BGR) if val.dtype != np.uint8 else val
    return None


def download_hf_split(
    dataset_id: str,
    split: str,
    rgb_dir: Path,
    prefix: str,
    image_keys: tuple[str, ...],
    max_images: int | None = None,
    start_index: int = 0,
) -> int:
    from datasets import load_dataset

    rgb_dir.mkdir(parents=True, exist_ok=True)
    print(f"  HF {dataset_id} ({split}) → max {max_images or 'all'}")
    use_stream = max_images is not None and max_images > 1500
    if use_stream:
        ds = load_dataset(dataset_id, split=split, streaming=True)
        iterator = iter(ds)
    else:
        ds = load_dataset(dataset_id, split=split)
        if max_images is not None and len(ds) > max_images:
            pick = np.linspace(0, len(ds) - 1, max_images, dtype=int)
            iterator = (ds[int(i)] for i in pick)
        else:
            iterator = iter(ds)

    saved = 0
    for row in iterator:
        if max_images is not None and saved >= max_images:
            break
        bgr = _extract_image_from_row(row, image_keys)
        if bgr is None:
            continue
        out = rgb_dir / f"{prefix}_{start_index + saved:06d}.jpg"
        if _save_bgr(bgr, out):
            saved += 1
        if saved and saved % 500 == 0:
            print(f"    … {saved} images")

    print(f"    ✓ {saved} from {dataset_id}")
    return saved


def download_euvp(rgb_dir: Path, max_images: int = 2500) -> int:
    return download_hf_split(
        "Ken1053/EUVP",
        "train",
        rgb_dir,
        "euvp",
        ("input_image", "image", "raw"),
        max_images=max_images,
    )


def download_aqua20(rgb_dir: Path, max_images: int = 1500) -> int:
    from datasets import load_dataset

    rgb_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("taufiktrf/AQUA20", split="train")
    n = min(max_images, len(ds)) if max_images else len(ds)
    indices = np.linspace(0, len(ds) - 1, n, dtype=int) if n < len(ds) else np.arange(len(ds))

    saved = 0
    for j, idx in enumerate(indices):
        row = ds[int(idx)]
        img = row.get("image")
        if img is None:
            continue
        bgr = _pil_to_bgr(img)
        if _save_bgr(bgr, rgb_dir / f"aqua20_{j:06d}.jpg"):
            saved += 1
    print(f"    ✓ {saved} from AQUA20")
    return saved


def download_uieb_hikari(rgb_dir: Path, max_images: int | None = 890) -> int:
    total = 0
    for split, pfx in [("train", "uieb_tr"), ("val", "uieb_va")]:
        cap = None
        if max_images:
            cap = max(0, max_images - total)
            if cap == 0:
                break
        n = download_hf_split(
            "Hikari0608/UIEB",
            split,
            rgb_dir,
            pfx,
            ("raw", "image", "input_image"),
            max_images=cap,
            start_index=total,
        )
        total += n
    return total


def download_uieb_raw_hf(rgb_dir: Path, max_images: int = 700) -> int:
    return download_hf_split(
        "yxnd150150/uieb_raw",
        "train",
        rgb_dir,
        "uieb_raw",
        ("input_image", "image"),
        max_images=max_images,
    )


def download_submerged3d(rgb_dir: Path, max_images: int | None = 80) -> int:
    return download_hf_split(
        "theflash987/Submerged3D",
        "train",
        rgb_dir,
        "submerged3d",
        ("image", "rgb", "img"),
        max_images=max_images,
    )


def download_funiegan_test_samples(rgb_dir: Path) -> int:
    """Small EUVP-style test set shipped in FUnIE-GAN repo (23 images)."""
    api = "https://api.github.com/repos/xahidbuffon/FUnIE-GAN/git/trees/bf61260d55dfb6954c651a4ca549fb8ddd9d59d6?recursive=1"
    try:
        with urllib.request.urlopen(api, timeout=60) as resp:
            tree = json.loads(resp.read().decode())
    except Exception as e:
        print(f"    FUnIE-GAN list failed: {e}")
        return 0

    saved = 0
    for node in tree.get("tree", []):
        path = node.get("path", "")
        if not path.startswith("data/test/A/") or not path.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        url = f"https://raw.githubusercontent.com/xahidbuffon/FUnIE-GAN/master/{path}"
        name = Path(path).name
        dest = rgb_dir / f"funiegan_{name}"
        if dest.exists():
            saved += 1
            continue
        try:
            urllib.request.urlretrieve(url, dest)
            saved += 1
        except Exception:
            pass
    print(f"    ✓ {saved} from FUnIE-GAN test/A")
    return saved


def download_all_online(
    rgb_dir: Path | None = None,
    *,
    euvp_max: int = PHASE1_DEFAULTS["euvp_max"],
    aqua20_max: int = PHASE1_DEFAULTS["aqua20_max"],
    uieb_hf_max: int = PHASE1_DEFAULTS["uieb_hf_max"],
    uieb_raw_max: int = PHASE1_DEFAULTS["uieb_raw_max"],
    submerged3d_max: int = PHASE1_DEFAULTS["submerged3d_max"],
    davis_sample_rate: int = PHASE1_DEFAULTS["davis_sample_rate"],
    davis_max_per_video: int = PHASE1_DEFAULTS["davis_max_per_video"],
    funiegan_test: bool = PHASE1_DEFAULTS["funiegan_test"],
    skip_hf: bool = False,
) -> dict[str, int]:
    from ceti.data_curation.underwater_real import curate_davis_videos, ingest_static_dir

    rgb_dir = rgb_dir or REPO_ROOT / "data/underwater_field/rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}

    print("Downloading / curating online underwater RGB…")

    video_dir = REPO_ROOT / "assets/examples_video"
    stats["davis"] = curate_davis_videos(
        video_dir,
        rgb_dir,
        sample_rate=davis_sample_rate,
        max_per_video=davis_max_per_video,
    )

    if not skip_hf:
        for name, fn, kwargs in [
            ("euvp", download_euvp, {"max_images": euvp_max}),
            ("uieb_hikari", download_uieb_hikari, {"max_images": uieb_hf_max}),
            ("uieb_raw", download_uieb_raw_hf, {"max_images": uieb_raw_max}),
            ("aqua20", download_aqua20, {"max_images": aqua20_max}),
            ("submerged3d", download_submerged3d, {"max_images": submerged3d_max}),
        ]:
            try:
                stats[name] = fn(rgb_dir, **kwargs)
            except Exception as e:
                stats[name] = 0
                stats[f"{name}_error"] = str(e)
                print(f"    ✗ {name}: {e}")

    if funiegan_test:
        try:
            stats["funiegan"] = download_funiegan_test_samples(rgb_dir)
        except Exception as e:
            stats["funiegan"] = 0
            stats["funiegan_error"] = str(e)

    stats["ceti_field"] = ingest_static_dir(REPO_ROOT / "data/ceti_field", rgb_dir, "ceti_field")
    stats["whale_raw"] = ingest_static_dir(REPO_ROOT / "data/whale/raw_frames", rgb_dir, "whale_raw")
    stats["total_rgb"] = len(list(rgb_dir.glob("*.jpg")))

    return stats


def write_source_stats(rgb_dir: Path, stats: dict) -> dict[str, int]:
    from collections import Counter

    c = Counter()
    for p in rgb_dir.glob("*.jpg"):
        name = p.stem.split("_")[0]
        if name.startswith("davis"):
            c["davis"] += 1
        elif name.startswith("euvp"):
            c["euvp"] += 1
        elif name.startswith("uieb"):
            c["uieb"] += 1
        elif name.startswith("aqua20"):
            c["aqua20"] += 1
        elif name.startswith("submerged3d"):
            c["submerged3d"] += 1
        elif name.startswith("funiegan"):
            c["funiegan"] += 1
        elif name.startswith("ceti"):
            c["ceti_field"] += 1
        else:
            c["other"] += 1
    return dict(c)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download online underwater RGB for CETI")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/underwater_field/rgb")
    parser.add_argument("--euvp-max", type=int, default=PHASE1_DEFAULTS["euvp_max"])
    parser.add_argument("--aqua20-max", type=int, default=PHASE1_DEFAULTS["aqua20_max"])
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--davis-only", action="store_true")
    args = parser.parse_args()

    if args.davis_only:
        from ceti.data_curation.underwater_real import curate_davis_videos

        n = curate_davis_videos(REPO_ROOT / "assets/examples_video", args.output, sample_rate=2, max_per_video=600)
        print(f"DAVIS frames: {n}")
    else:
        s = download_all_online(
            args.output,
            euvp_max=args.euvp_max,
            aqua20_max=args.aqua20_max,
            skip_hf=args.skip_hf,
        )
        print(json.dumps(s, indent=2))
        print("By prefix:", write_source_stats(args.output, s))
