# AVATARS / Professor Gil alignment

This document maps **Whale-Depth-Anything** work to what Professor **Stephanie Gil** and **Project CETI** need for the **AVATARS** autonomy framework (Autonomous Vehicles for whAle Tracking And Rendezvous by remote Sensing).

## Research objective

Fuse **monocular depth**, **visual whale detections**, and **acoustic tag bearing** so an autonomous platform can approach sperm whales safely and repeatably in open ocean conditions—without relying on depth or whale appearance from Depth Anything’s original pretraining (neither includes sperm whales or CETI field optics).

## Track mapping

### Track A — Metric depth (meters)

- **Need:** Absolute range on ROV/AUV for navigation and 3D reasoning.
- **Method:** ZoeDepth fine-tune on FLSea / SQUID RGB-D (`ceti/depth/train_underwater.py`, `ceti_metric.py`).
- **Status:** Pipeline ready; **requires metric datasets** (not in phase-1 online scrape).
- **MPS:** Metric train/eval uses `get_device()` — runs on Mac GPU when CUDA absent.

### Track B — Whale detection

- **Need:** Bounding boxes on surface / partial submergence for rendezvous geometry.
- **Method:** YOLOv8 + optional Depth Anything backbone init (`ceti/whale/`).
- **Status:** Scaffold ready; **CETI field labels are the gating item**.

### Track B2 — Underwater / marine relative depth (primary deliverable now)

- **Need:** Stable relative depth on turbid, color-shifted underwater RGB (and later aerial whale video).
- **Method:** Frozen Depth Anything teacher → student distillation (scale-shift + gradient loss), `preprocess_method: combined`.
- **Data phase 1:** ~6k public underwater RGB (EUVP, AQUA20, UIEB, DAVIS, Submerged3D) — see [PHASE1_ONLINE_DATA.md](PHASE1_ONLINE_DATA.md).
- **Data phase 2:** `data/ceti_field/` expedition and lab footage (target domain).
- **Compute:** **Apple M5 Max MPS**, ViT-L, AMP — `whale_depth_m5max_128gb.yaml`.

### Track C — Robot integration

- **Need:** Low-latency depth (+ detections) on lab stack.
- **Artifacts:** `infer_robot.py`, `ros2_perception_node.py` (MPS-aware).

### AVATARS fusion

- **Artifact:** `ceti/robot/avatars_pipeline.py` — depth map + detections + tag AOA.
- **Blocked until:** Track B labels + acceptable Track B2 (and ideally Track A on ROV).

## Milestones (suggested for lab reviews)

| Milestone | Evidence | Command |
|-----------|----------|---------|
| M0 | Repo + MPS verified on Mac | `bash ceti/scripts/setup_mac_mps.sh` |
| M1 | Real UW RGB → depth panels | `bash ceti/scripts/prove_pipeline.sh --skip-metric-train` |
| M2 | Phase-1 domain-adapted checkpoint | `bash ceti/scripts/train_mac_full.sh` |
| M3 | CETI field frames in train mix | Add `data/ceti_field/`, re-curate, re-train |
| M4 | Whale detector v1 | Label + `ceti/whale/train_whale.py` |
| M5 | AVATARS fusion demo | `avatars_pipeline.py` on field clip |

## What to report to the professor

1. **Pipeline proof** — `ceti/outputs/proof/report.json` + side-by-side depth panels on DAVIS underwater video.
2. **Training config** — ViT-L, pseudo-depth, real public marine RGB (not synthetic-only).
3. **Honest limits** — no sperm-whale GT in pretrain; metric meters need FLSea/SQUID; detection needs CETI annotations.
4. **Next data ask** — CETI field RGB/video, Dominica expedition frames, ROV calibration if metric depth is required.

## References

- Gil et al., AVATARS — Harvard SEAS.
- [Project CETI](https://www.projectceti.org/)
- Depth Anything — [LiheYoung/Depth-Anything](https://github.com/LiheYoung/Depth-Anything)
- Underwater adaptation context — [arxiv:2507.02148](https://arxiv.org/abs/2507.02148)
