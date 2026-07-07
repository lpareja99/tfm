# hugging_faces_trial — the HuggingFace Transformers attempt

> **Status: exploratory / historical. Abandoned in favour of MMSegmentation.
> Not part of the final reproducible pipeline and not maintained.**

## What it is

A short-lived attempt to train and evaluate Mask2Former through the **HuggingFace
Transformers** stack (`Mask2FormerForUniversalSegmentation` +
`Mask2FormerImageProcessor` + `Trainer`) instead of MMSegmentation, fine-tuning
from `facebook/mask2former-swin-tiny-ade-semantic`.

Unlike the final method (9 defect classes), this trial targeted only the **crack
classes** — 4 labels: `bg`, `cracks`, `cracks_alligator`, `cracks_severe`
(labels from `labels_cracks`).

## How it was organized

```
hugging_faces_trial/
├── Dockerfile          # FROM pytorch:2.6.0-cuda12.4-cudnn9; HF stack, numpy<2 pin
├── requirements.txt    # transformers, datasets, evaluate, accelerate, albumentations, wandb, ...
├── config.yml          # data paths, ID2LABEL (4 classes), BASE_MODEL, IMAGE_SIZE 512x512, WORK_DIR_BASE
├── configs/config.py   # the config actually imported by the scripts
├── dataset.py          # BasicRoadDataset: reads splits/{train,val,test}.txt, image + label PNG
├── train.py            # HF Trainer loop; albumentations aug; EarlyStopping; wandb logging
├── test.py             # evaluation on the test split
└── metrics.py          # compute_metrics using evaluate.load("mean_iou")
```

## How it was run

```bash
cd hugging_faces_trial
docker build -t hf-trial .
docker run --rm --gpus all -v $(pwd):/app -v ../data:/app/data hf-trial \
  python train.py           # trains from configs/config.py; logs to Weights & Biases
docker run --rm --gpus all -v $(pwd):/app -v ../data:/app/data hf-trial \
  python test.py            # evaluates a checkpoint on the test split
```

Training/eval settings (base model, data paths, label map, image size, work dir)
are edited in `configs/config.py` (see also the annotated `config.yml`).

## Why it was replaced

MMSegmentation gave the setup the thesis actually needed: the Mask2Former +
**swappable-backbone** design, the `_base_` config-inheritance system, `mim`
train/test tooling, and easy Azure ML containerization. The HuggingFace route was
dropped in its favour early on.

## Things to keep in mind

- **Superseded, do not extend.** Kept only as a record of the alternative explored.
- Scope was **crack classes only (4 labels)**, not the full 9-class defect set of
  the final work — numbers here are not comparable to the final results.
- Requires a **Weights & Biases** account/login for logging (`wandb`), and a
  different, newer Docker base (pytorch 2.6 / CUDA 12.4) than `road_defect_base`.
- `config.yml` and `configs/config.py` overlap; the **scripts import
  `configs/config.py`**, so that is the source of truth (`config.yml` is annotated notes).
- Paths assume the dataset is mounted at `data/2026-01-19-defect_dataset`.
