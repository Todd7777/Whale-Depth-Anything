# Phase 1 — Online underwater RGB bootstrap

Public sources used before CETI field bulk ingest. Goal: **5k–8k+ diverse underwater RGB** for relative-depth domain adaptation (teacher–student, no metric GT).

## Sources

| Source | HF / origin | Default cap | Role |
|--------|-------------|-------------|------|
| EUVP | `Ken1053/EUVP` | 2500 | Large paired underwater scenes |
| AQUA20 | `taufiktrf/AQUA20` | 1500 | Marine life / reef (stratified sample) |
| UIEB | `Hikari0608/UIEB` | ~890 | Raw underwater |
| UIEB raw | `yxnd150150/uieb_raw` | 700 | Additional raw |
| Submerged3D | `theflash987/Submerged3D` | 80 | Synthetic submerged |
| DAVIS | `assets/examples_video/` | 600 frames/video | Dolphins + seasnake motion |
| FUnIE-GAN | GitHub test/A | 23 | Extra test stills |

## Commands

```bash
# Download + train/val lists (requires network)
bash ceti/scripts/download_all_online_data.sh

# Train (MPS on Mac, CUDA on Linux)
bash ceti/scripts/run_phase1_train.sh

# Smaller smoke test
EUVP_MAX=200 AQUA20_MAX=100 bash ceti/scripts/download_all_online_data.sh
```

## Later (not phase 1)

- `data/ceti_field/` — CETI expedition video
- FLSea / SQUID — metric RGB-D for ZoeDepth track
- Whale detection labels — sperm whale YOLO on field frames

Checkpoint: `checkpoints/ceti_phase1_online/best.pt`
