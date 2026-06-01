#!/usr/bin/env python3
"""Verify PyTorch MPS (Metal) for CETI training on Apple Silicon."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    import torch

    from ceti.utils.device import configure_compute, device_name, get_device, mps_available

    print("=" * 60)
    print("CETI — MPS / Metal verification")
    print("=" * 60)
    print(f"  Platform:  {platform.system()} {platform.machine()}")
    print(f"  Processor: {platform.processor() or 'n/a'}")
    print(f"  PyTorch:   {torch.__version__}")
    print(f"  MPS built: {getattr(torch.backends, 'mps', None) is not None}")
    print(f"  MPS avail: {mps_available()}")

    n = configure_compute()
    print(f"  CPU threads: {n}")

    device = get_device()
    print(f"  CETI_DEVICE: {__import__('os').environ.get('CETI_DEVICE', 'auto')}")
    print(f"  Selected:    {device_name(device)}")

    if device.type != "mps":
        print("\nFAIL: Expected MPS on Apple Silicon for full training.")
        print("  Run: bash ceti/scripts/setup_mac_mps.sh")
        return 1

    x = torch.randn(2, 3, 64, 64, device=device)
    y = x * 2.0 + 1.0
    with torch.autocast(device_type="mps", enabled=True):
        z = torch.nn.functional.conv2d(
            x.float(), torch.randn(8, 3, 3, 3, device=device), padding=1
        )
    del x, y, z
    if hasattr(torch, "mps"):
        torch.mps.empty_cache()

    print("\nOK — MPS forward + autocast succeeded. Ready for train_mac_full.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
