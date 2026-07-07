# Recovery & reproduction

Nothing under `data/` is tracked in git (datasets, pretrained weights, trained checkpoints and
outputs are large and re-downloadable / externally backed up). This page explains how to
reobtain each piece so the pipeline runs end-to-end from a fresh clone.

## Azure coordinates

The Azure targets read their coordinates from a gitignored `.env` (template: `.env.example`):

```
ACR_REGISTRY=<your-registry>.azurecr.io
AZ_SUBSCRIPTION=<subscription-id>
AZ_RESOURCE_GROUP=<resource-group>
AZ_WORKSPACE=<workspace>
AZ_COMPUTE=<compute-target>
```

`docker-compose.yml` and the `Makefile` interpolate these; the Azure job specs
(`experiments/*/azure/*.yml`) use the same `${...}` / `azureml:${AZ_COMPUTE}` placeholders —
substitute your own values (the Azure ML CLI does not expand env vars in YAML). Then
`az login` before any `az ml ...` command.

## Datasets

| Dataset | Path | How to obtain |
|---|---|---|
| Adverse-weather (§4.2) | `data/final_dataset/` | Kaggle — Laura Pareja "weather-adverse" set (≈335 imgs: dry / wet / half). Masks are resized to `labels_720/` (1280×720, NEAREST) with `scripts/data_prep/resize_weather_labels.py`. |
| In-domain defect (§4.1) | `data/2026-01-19-defect_dataset/` | The "Flowity" defect set (images + `labels_basic_defects_relabel` for the 9-class task + `splits/{train,val,test}.txt`). Prepared with `scripts/data_prep/`. |

Expected layout (both share the mmseg convention):
```
data/<dataset>/
├── images/
├── labels_720/  (weather)  |  labels_basic_defects_relabel/  (defect, 9-class)
└── splits/{train,val,test}.txt   |  {dry,wet,half}.txt  (weather)
```

## Pretrained backbone weights

`data/pretrained/` holds the public ImageNet/ADE20K backbone weights the configs load at
**training** time. See `data/pretrained/README.md` for the catalog + download sources
(OpenGVLab for InterImage/FlashInternImage, official BEiT v2, OpenMMLab for Swin/HRNet). They
are also copied to the per-experiment path each `config.py` expects, so the configs need no edit.

## Trained checkpoints

Trained on Azure ML (5 seeds per backbone: `42, 91, 777, 1337, 2026`). Pull them back with:

```bash
make download-jobs                       # bajar_jobs.sh → data/checkpoints/<model>/seed_<S>/
make download-job-azure JOB_ID=<parent>  # a single pipeline parent job (download_job.py)
```

Checkpoints land at `data/checkpoints/<model>/seed_<S>/best_mIoU_iter_*.pth`. The winners used
downstream are Swin **seed 91** (`best_mIoU_iter_34000.pth`) and FlashInternImage **seed 777**
(`best_mIoU_iter_35000.pth`). ~100 GB total → kept local + on an external backup disk.

> The mapping from Azure job names to `(model, seed)` and the external-backup index are kept in
> the author's private notes (not published), since job names are workspace-internal. The public
> checkpoint index (seed / iter / mIoU) is in [RESULTS.md](RESULTS.md).

## Reproduce from scratch

1. `cp .env.example .env` and fill in coordinates → `make build-base` (+ `build-flash`/`build-intern` for GPU backbones).
2. Obtain the datasets (above) into `data/`.
3. Either **train** (`make train-azure EXP=…` or `make train-local EXP=…`) or **download** the
   existing checkpoints (`make download-jobs`).
4. **Test** in-domain: `make test-flowity MODEL=… MODE=full` → `make parse-logs`.
5. **Weather** robustness: `make weather MODEL=… MODE=full`.
