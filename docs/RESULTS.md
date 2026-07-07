# Results

Model complexity and in-domain test performance for the five backbones (Mask2Former, 9 classes).

## Model complexity & speed

FPS measured single-image at 512×512; parameters full-model; GFLOPs for the backbone only
(the Mask2Former head is too dynamic for the tracer).

| Backbone | Params (M) | Backbone GFLOPs | FPS (img/s) |
|---|---:|---:|---:|
| Swin-T | 47.28 | 25.61 | 15.35 |
| HRNet-W32 | 48.94 | 41.53 | 6.42 |
| InterImage-T | 48.53 | 25.10 | 7.16 |
| FlashInternImage-T | 50.54 | 26.53 | 14.64 |
| BEiT v2 (base) | 109.02 | 107.12 | 10.64 |

## In-domain test (Flowity test set, 723 images)

Per-class **IoU (%)** and overall mIoU:

| Class | Swin-T | FlashInternImage | BEiT v2 | HRNet | InterImage |
|---|---:|---:|---:|---:|---:|
| bg | 93.57 | 95.75 | 96.07 | 95.53 | 95.98 |
| cracks | 24.08 | 25.52 | 19.62 | 22.25 | 23.05 |
| cracks_alligator | 50.60 | 51.00 | 49.17 | 45.65 | 45.63 |
| cracks_severe | 30.97 | 33.54 | 30.50 | 29.26 | 28.29 |
| edge_cracks | 3.92 | 0.08 | 0.00 | 0.08 | 0.00 |
| fretting | 18.27 | 21.10 | 22.73 | 16.81 | 8.02 |
| pothole | 28.67 | 30.30 | 25.91 | 17.91 | 10.17 |
| manhole | 62.58 | 66.60 | 41.54 | 47.95 | 42.90 |
| pole_shadow | 47.34 | 48.73 | 51.48 | 41.61 | 17.94 |
| **mIoU** | **40.00** | **41.40** | **37.45** | **35.23** | **30.22** |
| aAcc | 92.98 | 95.29 | 95.56 | 95.03 | 95.42 |
| mAcc | 58.98 | 57.45 | 51.15 | 48.66 | 41.73 |

Full per-class Acc/Dice/Fscore/Precision/Recall and **per-seed** breakdowns are in
[`notebooks/results_analysis/`](../notebooks/results_analysis/): `iou_flowity_table.md` and the
`testing_analytics_<model>.xlsx` / `training_analytics_<model>.xlsx` files (regenerate with
`make parse-logs`).

## Checkpoint index

Each backbone was trained with **5 seeds** — `42, 91, 777, 1337, 2026`. Checkpoints are not in
git (see [RECOVERY.md](RECOVERY.md)); they live under `data/checkpoints/<model>/seed_<S>/`.

| Model | "Winner" seed | Example checkpoint |
|---|---|---|
| Swin-T | 91 | `best_mIoU_iter_34000.pth` |
| FlashInternImage | 777 | `best_mIoU_iter_35000.pth` |
| InterImage | 91 | `best_mIoU_iter_*.pth` |
| HRNet | 1337 | `best_mIoU_iter_*.pth` |
| BEiT v2 | 42 | `best_mIoU_iter_*.pth` |

Swin seed 91 and FlashInternImage seed 777 are the checkpoints used for the §4.2
adverse-weather study. Per-seed iteration/mIoU detail is in the `*_analytics_*.xlsx` files.

## Adverse-weather robustness (§4.2)

Running the in-domain checkpoints on the dry/wet/half weather set shows a large **domain-shift
collapse** (in-domain mIoU ~30–41 → weather mIoU single digits), quantified per condition by
`make weather` (outputs under `data/output/weather/`). The fine-grained analysis
(present-class mIoU + confusion matrices) is produced by the weather-analysis tooling.
