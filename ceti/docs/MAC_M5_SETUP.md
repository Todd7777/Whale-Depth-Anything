# M5 Max (128GB) — complete setup

Repository: https://github.com/Todd7777/Whale-Depth-Anything

## Prerequisites

- macOS on Apple Silicon (M5 Max)
- Xcode Command Line Tools: `xcode-select --install`
- Git, stable Wi‑Fi (large downloads)
- **~15 GB free disk** (venv + checkpoints + training images)

Optional (faster Hugging Face):

```bash
export HF_TOKEN='hf_...'   # paste locally — never commit
```

---

## Fresh clone (recommended)

```bash
cd ~
git clone https://github.com/Todd7777/Whale-Depth-Anything.git
cd Whale-Depth-Anything

export CETI_DEVICE=mps
export CETI_REQUIRE_MPS=1
export CETI_UNIFIED_MEMORY_GB=128
export CETI_SKIP_GIT_PULL=1

chmod +x ceti/scripts/*.sh

bash ceti/scripts/mac_m5_full_setup_and_train.sh
```

This runs: MPS setup → checkpoints → smoke test → **download ~6k images** → train 40 epochs → proof.

**Time:** setup ~30 min + data download ~30–60 min + training **many hours** (ViT-L, 5911 images).

---

## Already cloned / setup passed once?

Skip re-install; only data + train:

```bash
cd ~/Whale-Depth-Anything   # your path

export CETI_DEVICE=mps
export CETI_REQUIRE_MPS=1
export CETI_UNIFIED_MEMORY_GB=128
export HF_TOKEN='hf_...'    # optional

bash ceti/scripts/mac_train_only.sh
```

---

## Step-by-step (manual)

```bash
cd ~/Whale-Depth-Anything
export CETI_DEVICE=mps CETI_REQUIRE_MPS=1 CETI_UNIFIED_MEMORY_GB=128

# 1) Venv + PyTorch Metal + MPS check
bash ceti/scripts/setup_mac_mps.sh

# 2) Relative-depth checkpoints (vits/vitb/vitl) — required
bash ceti/scripts/download_checkpoints.sh

# 3) MUST have image files on disk (not just train list lines)
bash ceti/scripts/ensure_training_data.sh

# 4) Verify count
ls data/underwater_field/rgb/*.jpg | wc -l    # expect 5000+

# 5) Train + proof
bash ceti/scripts/train_mac_full.sh
```

---

## Success checks

| Step | Expected |
|------|----------|
| `verify_mps.py` | `OK — MPS forward + autocast succeeded` |
| `smoke_test.py` | `All tests passed!` |
| `data/underwater_field/rgb/` | Thousands of `.jpg` files |
| Training log | `Device: MPS / Metal` |
| Output | `checkpoints/ceti_whale_depth/best.pt` |
| Demo | `ceti/outputs/proof/` panels |

---

## Inference after training

```bash
cd ~/Whale-Depth-Anything
source .venv/bin/activate

python ceti/depth/infer_robot.py \
  --encoder vitl \
  --depth-checkpoint checkpoints/ceti_whale_depth/best.pt \
  --input assets/examples_video/davis_dolphins.mp4 \
  --underwater-preprocess \
  --outdir ceti/outputs/depth_demo
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `FileNotFoundError` for `data/underwater_field/rgb/...` | Run `bash ceti/scripts/ensure_training_data.sh` |
| `whale does not appear to be a git repository` | `export CETI_SKIP_GIT_PULL=1` or ignore — not fatal |
| Metric checkpoint 404 | OK — optional; relative training uses ViT-L `.pth` only |
| `Killed: 9` during epoch 1 | **Fixed:** `workers: 0`, `batch_size: 10`. If still killed: `--batch-size 6` or `export CETI_TEACHER_ON_CPU=1` |
| OOM on GPU | Lower `batch_size` to 8 or 6; keep `workers: 0` on Mac |
| Training on CPU | Fix PyTorch Metal; must see `MPS / Metal` in log |

---

## Config (M5 Max 128GB)

File: `ceti/configs/whale_depth_m5max_128gb.yaml`

- ViT-L, **cache_teacher: true** (~2× faster epochs), batch 12, workers 0, val every 5 epochs

### Speed options

| Setting | Effect |
|---------|--------|
| `cache_teacher: true` | One-time teacher pass, then student-only training (**default in m5max yaml**) |
| `batch_size: 12` | Raise to 14 if stable; lower to 8 if `Killed: 9` |
| `val_every: 5` | Less validation overhead |
| `CETI_TORCH_COMPILE=1` | Optional compile (experimental on MPS) |
| `freeze_encoder: true` | Faster but weaker — only depth head trains |
- Override: `export CETI_TRAIN_CONFIG=ceti/configs/whale_depth_phase1.yaml`
