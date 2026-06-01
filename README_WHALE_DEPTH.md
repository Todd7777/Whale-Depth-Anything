# Whale-Depth-Anything

**Harvard SEAS · Project CETI · Professor Stephanie Gil (AVATARS)**

Monocular **underwater depth** and **whale perception** built on [Depth Anything](https://github.com/LiheYoung/Depth-Anything), extended for lab robots and field deployment toward [AVATARS](https://www.seas.harvard.edu/) autonomous whale tracking and rendezvous.

Repository: [https://github.com/Todd7777/Whale-Depth-Anything](https://github.com/Todd7777/Whale-Depth-Anything)

---

## What Professor Gil’s stack needs

| Track | Deliverable | This repo |
|-------|-------------|-----------|
| **A** | Metric depth (meters) on ROV/AUV | ZoeDepth path + FLSea/SQUID when RGB-D available |
| **B** | Sperm whale detection for rendezvous | YOLO pipeline (`ceti/whale/`) — needs CETI labels |
| **B2** | Relative depth in marine / underwater scenes | **Primary focus** — teacher–student fine-tune |
| **C** | Robot / ROS2 perception | `ceti/robot/ros2_perception_node.py`, `infer_robot.py` |
| **Fusion** | Depth + bbox + tag AOA | `ceti/robot/avatars_pipeline.py` |

See [ceti/docs/AVATARS_ALIGNMENT.md](ceti/docs/AVATARS_ALIGNMENT.md) for research alignment and milestones.

---

## Mac M5 Max (128GB) — full training on Metal (MPS)

**Do not train on CPU** on Apple Silicon; use the integrated GPU via MPS.

```bash
git clone https://github.com/Todd7777/Whale-Depth-Anything.git
cd Whale-Depth-Anything

export CETI_DEVICE=mps CETI_REQUIRE_MPS=1 CETI_UNIFIED_MEMORY_GB=128
export CETI_SKIP_GIT_PULL=1   # avoids broken optional git remote
chmod +x ceti/scripts/*.sh

# BULLETPROOF train (after setup once):
bash ceti/scripts/mac_train_bulletproof.sh
```

**Guide:** [ceti/docs/MAC_M5_SETUP.md](ceti/docs/MAC_M5_SETUP.md)

Full install + train:

```bash
bash ceti/scripts/setup_mac_mps.sh
bash ceti/scripts/mac_train_bulletproof.sh
```

Checkpoint: `checkpoints/ceti_whale_depth/best.pt`

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `CETI_DEVICE` | `mps` | `mps` \| `cuda` \| `cpu` \| `auto` |
| `CETI_REQUIRE_MPS` | `1` on Mac train scripts | Exit if MPS unavailable |
| `CETI_UNIFIED_MEMORY_GB` | `128` | Log hint only |
| `CETI_TRAIN_CONFIG` | `whale_depth_m5max_128gb.yaml` | Override config in `train_mac_full.sh` |

Verify Metal before a long run:

```bash
python ceti/scripts/verify_mps.py
```

### If you run out of GPU memory

Default `batch_size` is **20** for 128GB. If OOM, lower to `16` or `12` in `ceti/configs/whale_depth_m5max_128gb.yaml`. Keep `device: mps`.

---

## Inference (after training)

```bash
python ceti/depth/infer_robot.py \
  --encoder vitl \
  --depth-checkpoint checkpoints/ceti_whale_depth/best.pt \
  --input assets/examples_video/davis_dolphins.mp4 \
  --underwater-preprocess \
  --outdir ceti/outputs/depth_demo
```

---

## Data (not in git)

Training images live under `data/underwater_field/rgb/` after `download_all_online_data.sh`. Add CETI field footage under `data/ceti_field/` when available.

Train lists (portable relative paths): `ceti/data/whale_depth_train.txt`

---

## Documentation

- [ceti/README.md](ceti/README.md) — full CETI stack
- [ceti/docs/PHASE1_ONLINE_DATA.md](ceti/docs/PHASE1_ONLINE_DATA.md) — public bootstrap sources
- [ceti/docs/AVATARS_ALIGNMENT.md](ceti/docs/AVATARS_ALIGNMENT.md) — professor / lab milestones

Upstream Depth Anything paper: [arXiv:2401.10891](https://arxiv.org/abs/2401.10891)
