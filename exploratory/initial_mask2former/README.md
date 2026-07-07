# initial_mask2former — the first (monolithic) implementation

> **Status: exploratory / historical. Not part of the final reproducible pipeline
> and not maintained.** The final method lives per-backbone under `experiments/`.

## What it is

The first end-to-end implementation of the thesis: a **single, monolithic project**
that held *everything* in one place — every backbone config, the Azure ML job specs,
the CVAT data-preparation scripts, and the results-analysis scripts. It trained
Mask2Former on the road-defect dataset both locally and on Azure ML.

## How it was organized

```
initial_mask2former/
├── Dockerfile              # FROM pytorch:1.13.1-cuda11.6; installs mmengine/mmcv 2.1.0/mmdet/mmseg via mim
├── docker-compose.yml      # 3 services: mmseg-dev (GPU shell), tensorboard (:6007), jupyter (:8888)
├── requirements.txt        # mmseg deps + Azure ML SDK
├── configs/
│   ├── local/              # configs used for local runs (Swin, HRNet, InternImage, FlashInternImage,
│   │                       #   plus dataset variants: no-bg, subset-500, combined-cracks, augmentation)
│   └── azure/              # configs used inside Azure jobs (HRNet, Tversky-loss augmentation)
├── custom_modules/
│   └── intern_image.py     # InternImage backbone registration
└── scripts/
    ├── azure/              # Azure ML job wrappers + job YAMLs
    │   ├── Swin/, HRNet/   #   train_mask2former.py (builds a `mim train` command) + train_job.yml
    │   ├── test_mask2former.py, test_job.yml
    │   ├── download_job.py # download job outputs/checkpoints from Azure
    │   ├── azure_ml_hook.py, cracks_augmentation.py, db_creation.yml
    ├── mask_to_cvat.py, save_cvat_output.py, build_test_dataset_cvat.py  # CVAT annotation pipeline
    ├── create_balanced_subset.py, create_val_split.py                    # dataset splitting
    ├── mim_test_executer.py                                              # auto-find config+ckpt in a work_dir and `mim test`
    ├── hyperparam_tersorboard_upload.py, movie_flipping.py, image_demo.sh
    └── results_analysis/   # boundary_IoU, confusion_mtx, dilation/erosion tests, heat maps, t-SNE
```

## How it was run

Local (Docker Compose):
```bash
cd initial_mask2former
docker compose build
docker compose run --rm mmseg-dev bash        # GPU dev shell at /app
# then, inside the container, train with mim or via the wrapper:
mim train mmseg configs/local/HRNet.py --work-dir work_dirs/hrnet
# TensorBoard: docker compose up tensorboard   -> http://localhost:6007
# Jupyter:     docker compose up jupyter        -> http://localhost:8888
```

Azure ML (submitted with the `az ml job create` targets, later folded into the root Makefile):
```bash
az ml job create --file scripts/azure/Swin/train_job.yml \
  --subscription $AZ_SUBSCRIPTION --resource-group $AZ_RESOURCE_GROUP --workspace-name $AZ_WORKSPACE
```

## Why it was replaced

Mixing all backbones, data-prep, and analysis in one project made experiments hard
to isolate, reproduce, and containerize independently. The final design gives each
backbone its **own self-contained experiment** under `experiments/<backbone>/`
(config + Dockerfile + Azure job), with a single parametrized `Makefile` and
per-experiment images.

## Things to keep in mind

- **Superseded, do not extend.** Kept only as a record of the initial approach.
- Azure/ACR identifiers were parametrized to `${AZ_*}` / `${ACR_REGISTRY}` (see the
  root `.env.example`); the originals live in the pre-cleanup backup branches/tags.
- Many scripts use **hardcoded relative paths** (`data/...`, `work_dirs/...`,
  `output/...`) from the monolith layout and will not run as-is against the current
  repo layout without editing.
- Docker base here is **pytorch 1.13.1 / CUDA 11.6** — older than the final
  `road_defect_base` (pytorch 2.1.2 / CUDA 11.8).

## What graduated into the final pipeline

- **Results analysis** → promoted to `scripts/analysis/` and reworked in `notebooks/results_analysis/`.
- **Azure job download** → simplified into the root `bajar_jobs.sh` / `scripts/azure/download_job.py`.
- **Test-set / dataset preparation** → `scripts/data_prep/new_dataset_flowity_test_preparation.py`.
