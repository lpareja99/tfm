# Road-defect semantic segmentation — Master's thesis (roadai)

Semantic segmentation of road-surface defects with **Mask2Former** and **five swappable
backbones** — Swin-T, HRNet-W32, BEiT v2, InterImage (DCNv3) and FlashInternImage (DCNv4).
Models are trained on **Azure ML** and run locally via **Docker**. Beyond the in-domain
test set (§4.1), the thesis also studies **robustness to adverse weather** (§4.2) by running
inference on a dry/wet/half-wet dataset.

> **Quick start:** `cp .env.example .env` → `make build-base` → `make weather MODEL=swin MODE=smoke`.
> Run `make help` to see every command.

---

## Repository structure

```
tfm/
├── README.md                 # this file
├── Makefile                  # task runner — `make help`
├── Dockerfile                # road_defect_base image (FROM pytorch 2.1.2 / CUDA 11.8)
├── docker-compose.yml        # base + per-experiment (Azure child) build services
├── requirements.txt
├── .env.example              # Azure/ACR coordinates template → copy to .env (gitignored)
├── experiments/              # the final method — one self-contained dir per backbone
│   ├── swin-T-512x512/  HRNet-T-512x512/  BeiT2-T/
│   ├── InterImage-T-512x512/  flashInternImage-T-512x512/
│   │     config.py            # training config (+ variants: focal_loss, change_weights, v2/v3, …)
│   │     config_test.py       # test/inference config
│   │     Dockerfile           # Azure "child" image (+ Dockerfile.local for GPU sm_50, + .monolith)
│   │     custom_modules/      # backbone + CUDA ops (DCNv3 / DCNv4) for interimage/flash
│   │     azure/train_job.yml  # Azure ML job spec
├── runs/                     # per-experiment presets for `make train-local EXP=1a..5a`
├── scripts/
│   ├── run/                  # the entry-point runners (weather, flowity test, azure download)
│   ├── azure/                # download_job.py (pull Azure pipeline outputs)
│   ├── data_prep/            # dataset creation / resizing / CVAT conversion
│   ├── logs/                 # parse training/test logs → xlsx, curve plots
│   └── analysis/             # metric & qualitative analysis (+ understand_mask2former/)
├── notebooks/                # EDA + results_analysis/ (§4.1 training/test/qualitative)
├── docs/                     # SETUP.md · RESULTS.md · RECOVERY.md
├── exploratory/              # earlier abandoned approaches (monolith + HuggingFace), documented
└── data/                     # ALL gitignored — see "Data & checkpoints" below
```

`data/` is **not** in git (datasets, weights and outputs are re-downloadable / backed up
externally). See [docs/RECOVERY.md](docs/RECOVERY.md) for how to obtain everything.

---

## The five experiments

| Backbone | `experiments/` dir | Docker image | Custom CUDA op | Runs on |
|---|---|---|---|---|
| Swin-T | `swin-T-512x512` | `road_defect_base` | — | CPU or GPU |
| HRNet-W32 | `HRNet-T-512x512` | `road_defect_base` | — | CPU or GPU |
| BEiT v2 | `BeiT2-T` | `road_defect_base` | — | CPU or GPU |
| InterImage-T | `InterImage-T-512x512` | `road_defect_intern` | **DCNv3** | **GPU only** |
| FlashInternImage-T | `flashInternImage-T-512x512` | `road_defect_flash` | **DCNv4** | **GPU only** |

All share Mask2Former with **9 classes** (see [Classes](#classes)). InterImage and
FlashInternImage need their custom CUDA op compiled → they only run on GPU, in their own image.

---

## Setup

1. **Coordinates:** `cp .env.example .env` and fill in your Azure/ACR values (only needed for
   the Azure targets; `.env` is gitignored). See [docs/SETUP.md](docs/SETUP.md) for the full
   environment (Docker, NVIDIA toolkit, and the local GPU sm_50 builds).
2. **Build the base image:** `make build-base` (→ `road_defect_base`).
3. **(GPU backbones)** `make build-flash` and/or `make build-intern` to compile DCNv4/DCNv3 for
   your GPU. On the thesis machine (Quadro M1200) they are compiled for `sm_50`.

---

## How to run each thing

Everything is a `make` target (run `make help` for the list). Each target wraps the underlying
runner/script shown below.

### Train
```bash
make train-local EXP=1c          # local training with mim; EXP selects the backbone/preset
make train-azure EXP=1c          # submit an Azure ML job (needs .env + `az login`)
```
`EXP` presets live in `runs/*.env`: `1a` swin (cracks, 4-cls) · `1b` swin (all defects, 17-cls) ·
`1c` swin (relabel, 9-cls) · `2a` hrnet · `3a` interimage · `4a` flash · `5a` beit.
*(FlashInternImage/InterImage local training needs the GPU image + PYTHONPATH — the thesis trained them on Azure.)*

### Test — in-domain (§4.1)
```bash
make test-flowity MODEL=swin MODE=smoke     # 3 images; MODE=full = 723 images
make test-flowity MODEL=flash MODE=full     # any of swin|flash|hrnet|beit|interimage
```
Wraps `scripts/run/run_flowity_test.sh`. Produces per-run metric logs + raw masks under
`experiments/<exp>/outputx5/seed_<S>/test/`. Regenerate the metric tables with `make parse-logs`.

### Test — adverse-weather robustness (§4.2)
```bash
make weather MODEL=swin  MODE=smoke DEVICE=cpu    # smoke = 3 imgs
make weather MODEL=flash MODE=full                # dry + wet + half conditions
```
Wraps `scripts/run/run_weather.sh`. Outputs go to `data/output/weather/<model>/<cond>/`
(`pred_masks/` + `vis/` + logs). `flash`/`interimage` require GPU.

### Download trained checkpoints from Azure
```bash
make download-jobs                       # the thesis training jobs → data/checkpoints/ (bajar_jobs.sh)
make download-job-azure JOB_ID=<parent>  # one pipeline parent job (download_job.py)
```

### Analysis & dev
```bash
make parse-logs                 # (re)build notebooks/results_analysis/*_analytics_*.xlsx
make num-params EXP=1c          # params + backbone GFLOPs for a config
make jupyter                    # Jupyter Lab on http://localhost:8888
```
The §4.1 analysis notebooks live in [notebooks/results_analysis/](notebooks/results_analysis/)
(training curves, test metrics, qualitative panels). Standalone analysis tools are in
`scripts/analysis/`.

---

## Data & checkpoints

Everything under `data/` is **gitignored** — clone the repo, then obtain the data separately:

| Path | What | How to get it |
|---|---|---|
| `data/2026-01-19-defect_dataset/` | in-domain defect dataset (§4.1) | see [docs/RECOVERY.md](docs/RECOVERY.md) |
| `data/final_dataset/` | adverse-weather dataset (§4.2) | Kaggle (see RECOVERY) |
| `data/checkpoints/<model>/seed_<S>/` | trained checkpoints (~100 GB) | `make download-jobs` (Azure) |
| `data/pretrained/` | pretrained backbone weights | public downloads (catalog in that dir) |
| `data/output/` | inference outputs (weather, flowity test, analysis figures) | produced by the runs above |

---

## Results

Full tables (params/GFLOPs/FPS + per-class test IoU for all 5 backbones) are in
[docs/RESULTS.md](docs/RESULTS.md). Headline in-domain test mIoU (Flowity, 9 classes):
FlashInternImage 41.4 · Swin-T 40.0 · BEiT v2 37.5 · HRNet 35.2 · InterImage 30.2.

---

## Classes

9 classes (id · name · RGB), matching `experiments/*/config.py`:

| 0 bg `(0,0,0)` | 1 cracks `(250,50,83)` | 2 cracks_alligator `(36,179,83)` |
|---|---|---|
| **3** cracks_severe `(102,204,255)` | **4** edge_cracks `(255,165,0)` | **5** fretting `(128,128,128)` |
| **6** pothole `(255,255,0)` | **7** manhole `(0,255,255)` | **8** pole_shadow `(255,0,255)` |

---

## More

- **[docs/SETUP.md](docs/SETUP.md)** — full environment + Docker image builds (incl. local GPU sm_50).
- **[docs/RESULTS.md](docs/RESULTS.md)** — complexity + per-class test metrics + checkpoint index.
- **[docs/RECOVERY.md](docs/RECOVERY.md)** — how to reobtain datasets and checkpoints.
- **[exploratory/](exploratory/)** — earlier approaches (monolithic project, HuggingFace trial), documented for context.
