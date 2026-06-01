"""Disk cache for frozen teacher pseudo-depth (speeds up MPS training ~2x)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ceti.depth.whale_depth_dataset import WhaleDepthDataset, load_image_paths


def cache_file_for_image(cache_dir: Path, image_path: Path) -> Path:
    key = hashlib.sha1(str(image_path.resolve()).encode()).hexdigest()[:16]
    name = f"{image_path.stem}_{key}.npy"
    return cache_dir / name


def count_cached(cache_dir: Path, paths: list[Path]) -> int:
    if not cache_dir.exists():
        return 0
    return sum(1 for p in paths if cache_file_for_image(cache_dir, p).exists())


def precompute_teacher_cache(
    teacher: torch.nn.Module,
    paths: list[Path],
    cache_dir: Path,
    device: torch.device,
    *,
    image_size: int = 518,
    preprocess_method: str = "combined",
    batch_size: int = 12,
    use_amp: bool = True,
    hub_encoder: str = "vitl",
) -> int:
    """
    Run frozen teacher once; save float16 depth maps for student-only training.
    """
    from ceti.utils.device import autocast_device_type, empty_cache

    def predict_depth(model, images: torch.Tensor) -> torch.Tensor:
        return model(images)

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = cache_dir / "manifest.txt"
    needed = [p for p in paths if not cache_file_for_image(cache_dir, p).exists()]
    if not needed:
        print(f"  Teacher cache complete ({len(paths)} maps in {cache_dir})")
        return len(paths)

    print(f"  Precomputing teacher depth: {len(needed)} images (batch {batch_size})…")
    list_file = cache_dir / "_precompute_list.txt"
    list_file.write_text("\n".join(str(p) for p in needed) + "\n")

    ds = WhaleDepthDataset(
        list_file,
        image_size=image_size,
        preprocess_method=preprocess_method,
        augment=False,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    amp_dtype = autocast_device_type(device)
    saved = count_cached(cache_dir, paths)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Teacher cache", leave=False):
            images = batch["image"].to(device)
            with torch.autocast(device_type=amp_dtype, enabled=use_amp):
                depth = predict_depth(teacher, images)
            depth = depth.detach().float().cpu().numpy()
            for i, path_str in enumerate(batch["path"]):
                d = depth[i]
                if d.ndim == 3:
                    d = d.squeeze(0)
                out = cache_file_for_image(cache_dir, Path(path_str))
                np.save(out, d.astype(np.float16))

    empty_cache(device)
    saved = count_cached(cache_dir, paths)
    manifest.write_text(f"encoder={hub_encoder}\nsize={image_size}\npreprocess={preprocess_method}\ncount={saved}\n")
    print(f"  Cached {saved} teacher depth maps → {cache_dir}")
    return saved
