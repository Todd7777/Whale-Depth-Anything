"""
Curate real underwater RGB imagery for CETI depth / whale domain adaptation.

Sources (research-aligned, no synthetic color cast):
  - DAVIS underwater benchmarks (dolphins, sea snake) — ROV-like open water
  - Submerged3D (Hugging Face) — deep-sea turbidity / wreck scenes
  - AQUA20 subset (Hugging Face) — species in situ, challenging visibility
  - CETI field drops under data/ceti_field/
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# DAVIS clips that are underwater (exclude terrestrial rollercoaster)
DAVIS_UNDERWATER = (
    "davis_dolphins.mp4",
    "davis_seasnake.mp4",
)


def sharpness_score(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    prefix: str,
    sample_rate: int = 8,
    min_sharpness: float = 28.0,
    max_frames: int = 350,
    min_motion: float = 2.0,
) -> int:
    """Extract sharp, motion-rich frames from underwater video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    frame_idx = 0
    prev_gray = None

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            motion = float(cv2.absdiff(prev_gray, gray).mean())
            if motion < min_motion:
                frame_idx += 1
                prev_gray = gray
                continue

        if sharpness_score(frame) < min_sharpness:
            frame_idx += 1
            prev_gray = gray
            continue

        out = output_dir / f"{prefix}_f{frame_idx:06d}.jpg"
        cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
        prev_gray = gray
        frame_idx += 1

    cap.release()
    return saved


def curate_davis_videos(
    video_dir: Path,
    rgb_dir: Path,
    sample_rate: int = 8,
    max_per_video: int = 200,
) -> int:
    total = 0
    for name in DAVIS_UNDERWATER:
        path = video_dir / name
        if not path.exists():
            continue
        stem = path.stem
        n = extract_video_frames(path, rgb_dir, stem, sample_rate=sample_rate, max_frames=max_per_video)
        total += n
    return total


def download_submerged3d(rgb_dir: Path, max_images: int | None = None) -> int:
    """Download real deep-sea RGB from Hugging Face Submerged3D (~80 images)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("pip install datasets") from e

    rgb_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("theflash987/Submerged3D", split="train")
    saved = 0

    for i, row in enumerate(ds):
        if max_images and saved >= max_images:
            break
        img = None
        for key in ("image", "rgb", "img", "picture"):
            if key in row and row[key] is not None:
                img = row[key]
                break
        if img is None:
            continue
        if hasattr(img, "convert"):
            arr = np.array(img.convert("RGB"))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif isinstance(img, np.ndarray):
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
        else:
            continue
        scene = row.get("scene_id", row.get("scene", row.get("id", i)))
        out = rgb_dir / f"submerged3d_{scene}_{i:04d}.jpg"
        cv2.imwrite(str(out), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

    return saved


def download_aqua20_subset(rgb_dir: Path, max_images: int = 500, split: str = "train") -> int:
    """Sample real in-situ marine images from AQUA20 (classification benchmark)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("pip install datasets") from e

    rgb_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("taufiktrf/AQUA20", split=split)
    n = min(max_images, len(ds))
    indices = np.linspace(0, len(ds) - 1, n, dtype=int)

    saved = 0
    for j, idx in enumerate(indices):
        row = ds[int(idx)]
        img = row.get("image")
        if img is None:
            continue
        arr = np.array(img.convert("RGB"))
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        label = row.get("label", row.get("labels", "unk"))
        out = rgb_dir / f"aqua20_{label}_{j:05d}.jpg"
        cv2.imwrite(str(out), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved += 1

    return saved


def ingest_static_dir(src: Path, rgb_dir: Path, prefix: str) -> int:
    """Copy images from an existing directory (ceti_field, user drops)."""
    if not src.exists():
        return 0
    rgb_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
        for path in sorted(src.rglob(ext)):
            dest = rgb_dir / f"{prefix}_{path.stem}{path.suffix.lower()}"
            if dest.suffix != ".jpg":
                dest = dest.with_suffix(".jpg")
                bgr = cv2.imread(str(path))
                if bgr is None:
                    continue
                cv2.imwrite(str(dest), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                shutil.copy2(path, dest)
            count += 1
    return count


def write_manifest(rgb_dir: Path, manifest_path: Path, sources: dict[str, int]) -> int:
    """Write JSONL manifest for provenance tracking."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for path in sorted(rgb_dir.glob("*.jpg")):
        name = path.stem
        if name.startswith("davis_"):
            source = "davis_underwater"
        elif name.startswith("euvp_"):
            source = "euvp_hf"
        elif name.startswith(("uieb_", "uieb_tr_", "uieb_va_", "uieb_raw_")):
            source = "uieb_hf"
        elif name.startswith("submerged3d_"):
            source = "submerged3d_hf"
        elif name.startswith("aqua20_"):
            source = "aqua20_hf"
        elif name.startswith("funiegan_"):
            source = "funiegan_github"
        elif name.startswith("ceti_field_"):
            source = "ceti_field"
        elif name.startswith("whale_raw_"):
            source = "whale_raw"
        else:
            source = "unknown"
        lines.append(
            json.dumps(
                {
                    "path": str(path.resolve()),
                    "file": path.name,
                    "source": source,
                }
            )
        )

    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def build_train_val_lists(
    rgb_dir: Path,
    train_list: Path,
    val_list: Path,
    val_ratio: float = 0.12,
    seed: int = 42,
) -> tuple[int, int]:
    import random

    images = sorted(rgb_dir.glob("*.jpg"))
    if not images:
        train_list.write_text("# No images — run curate_underwater_field.sh\n")
        val_list.write_text("# No validation images\n")
        return 0, 0

    rng = random.Random(seed)
    repo = REPO_ROOT.resolve()

    def _list_path(p: Path) -> str:
        try:
            rel = p.resolve().relative_to(repo)
            return str(rel).replace("\\", "/")
        except ValueError:
            return str(p.resolve())

    paths = [_list_path(p) for p in images]
    rng.shuffle(paths)
    split = max(1, int(len(paths) * (1 - val_ratio)))
    train_lines = paths[:split]
    val_lines = paths[split:] if split < len(paths) else paths[-max(1, len(paths) // 10) :]

    train_list.parent.mkdir(parents=True, exist_ok=True)
    train_list.write_text("\n".join(train_lines) + "\n")
    val_list.write_text("\n".join(val_lines) + "\n")
    return len(train_lines), len(val_lines)


def curate_all(
    output_root: Path | None = None,
    *,
    phase1_online: bool = True,
    download_hf: bool = True,
    aqua20_max: int = 1500,
    euvp_max: int = 2500,
    davis_sample_rate: int = 2,
    davis_max_per_video: int = 600,
) -> dict:
    """
    Full curation into data/underwater_field/.

    phase1_online: use ceti.data_curation.download_online for large public datasets.
    """
    root = output_root or REPO_ROOT / "data/underwater_field"
    rgb_dir = root / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    if phase1_online and download_hf:
        from ceti.data_curation.download_online import download_all_online

        stats = download_all_online(
            rgb_dir,
            euvp_max=euvp_max,
            aqua20_max=aqua20_max,
            davis_sample_rate=davis_sample_rate,
            davis_max_per_video=davis_max_per_video,
        )
    else:
        stats: dict[str, int] = {}
        video_dir = REPO_ROOT / "assets/examples_video"
        stats["davis"] = curate_davis_videos(
            video_dir, rgb_dir, sample_rate=davis_sample_rate, max_per_video=davis_max_per_video
        )
        if download_hf:
            try:
                stats["submerged3d"] = download_submerged3d(rgb_dir)
            except Exception as e:
                stats["submerged3d"] = 0
                stats["submerged3d_error"] = str(e)
            try:
                stats["aqua20"] = download_aqua20_subset(rgb_dir, max_images=aqua20_max)
            except Exception as e:
                stats["aqua20"] = 0
                stats["aqua20_error"] = str(e)
        stats["ceti_field"] = ingest_static_dir(REPO_ROOT / "data/ceti_field", rgb_dir, "ceti_field")
        stats["whale_raw"] = ingest_static_dir(REPO_ROOT / "data/whale/raw_frames", rgb_dir, "whale_raw")
        stats["total_rgb"] = len(list(rgb_dir.glob("*.jpg")))

    manifest = root / "manifest.jsonl"
    write_manifest(rgb_dir, manifest, stats)

    train_n, val_n = build_train_val_lists(
        rgb_dir,
        REPO_ROOT / "ceti/data/underwater_field_train.txt",
        REPO_ROOT / "ceti/data/underwater_field_val.txt",
    )
    stats["train"] = train_n
    stats["val"] = val_n

    # Mirror into ceti_field for backward-compatible paths
    field_rgb = REPO_ROOT / "data/ceti_field/rgb"
    field_rgb.mkdir(parents=True, exist_ok=True)
    for p in rgb_dir.glob("*.jpg"):
        link = field_rgb / p.name
        if not link.exists():
            try:
                link.symlink_to(p.resolve())
            except OSError:
                shutil.copy2(p, link)

    # Whale depth file lists (primary training lists for domain adaptation)
    build_train_val_lists(
        rgb_dir,
        REPO_ROOT / "ceti/data/whale_depth_train.txt",
        REPO_ROOT / "ceti/data/whale_depth_val.txt",
        val_ratio=0.12,
    )

    return stats
