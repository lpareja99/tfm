#!/usr/bin/env bash
# ===========================================================================
# §4.1 Flowity test-set inference. Produces, in ONE run per (model, seed):
#   (a) the metric log  -> later parsed into testing_analytics_<model>.xlsx
#   (b) raw index masks -> experiments/<exp>/outputx5/seed_<S>/test/raw_masks/
#       (exactly the path notebooks/results_analysis/qualitative_reorganized.ipynb expects)
#
# Reuses config_test.py of each model (backbone matches the checkpoint) and just
# repoints the data to Flowity via --cfg-options.
#
# Usage:
#   bash scripts/run/run_flowity_test.sh <model> [mode] [seed] [device]
#     model : swin | flash | hrnet | beit | interimage
#     mode  : smoke (3 imgs) | full (default, 723 imgs)
#     seed  : checkpoint seed (default = the "winner" seed used by the qualitative notebook)
#     device: auto (default) | gpu | cpu   (flash/interimage are GPU-only)
# ===========================================================================
set -uo pipefail
# repo root (this script lives in scripts/run/) — works from any CWD
TFM="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FLOW="$TFM/data/2026-01-19-defect_dataset"

MODEL="${1:?model required: swin|flash|hrnet|beit|interimage}"
MODE="${2:-full}"
SEED_OVERRIDE="${3:-}"
DEVICE="${4:-auto}"

# per-model: dir | image | needs_gpu | custom_import | qualitative winner seed
case "$MODEL" in
  swin)       EXPDIR=swin-T-512x512;              IMAGE=road_defect_base;   NEEDS_GPU=0; CUSTOM_IMPORT=0; DEF_SEED=91   ;;
  flash)      EXPDIR=flashInternImage-T-512x512;  IMAGE=road_defect_flash;  NEEDS_GPU=1; CUSTOM_IMPORT=1; DEF_SEED=777  ;;
  interimage) EXPDIR=InterImage-T-512x512;        IMAGE=road_defect_intern; NEEDS_GPU=1; CUSTOM_IMPORT=1; DEF_SEED=91   ;;
  hrnet)      EXPDIR=HRNet-T-512x512;              IMAGE=road_defect_base;   NEEDS_GPU=0; CUSTOM_IMPORT=0; DEF_SEED=1337 ;;
  beit)       EXPDIR=BeiT2-T;                      IMAGE=road_defect_base;   NEEDS_GPU=0; CUSTOM_IMPORT=0; DEF_SEED=42   ;;
  *) echo "ERROR: unknown model '$MODEL'"; exit 2 ;;
esac
SEED="${SEED_OVERRIDE:-$DEF_SEED}"

if [ "$NEEDS_GPU" = "1" ]; then
  [ "$DEVICE" = "cpu" ] && { echo "ERROR: $MODEL is GPU-only"; exit 2; }
  DEVICE=gpu
elif [ "$DEVICE" = "auto" ]; then DEVICE=gpu; fi
if [ "$DEVICE" = "gpu" ]; then GPU_ARGS="--gpus all"; CVD=0; else GPU_ARGS=""; CVD=""; fi

docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "ERROR: missing image '$IMAGE'"; exit 2; }

# test split (smoke = first 3)
ANN="splits/test.txt"
if [ "$MODE" = "smoke" ]; then
  head -n 3 "$FLOW/splits/test.txt" > "$FLOW/splits/test_smoke.txt"
  ANN="splits/test_smoke.txt"; echo ">>> $MODEL | SMOKE (3 imgs) | seed=$SEED | device=$DEVICE"
else
  echo ">>> $MODEL | FULL Flowity test (723 imgs) | seed=$SEED | device=$DEVICE"
fi

OUT="experiments/$EXPDIR/outputx5/seed_$SEED/test"   # raw_masks land here (qualitative path)
mkdir -p "$TFM/$OUT"

docker run --rm $GPU_ARGS --shm-size=8g \
  -e CUDA_VISIBLE_DEVICES="$CVD" \
  -e MODEL="$MODEL" -e SEED="$SEED" -e ANN="$ANN" -e OUT="$OUT" -e CUSTOM_IMPORT="$CUSTOM_IMPORT" \
  -v "$TFM":/app -w /app/experiments/$EXPDIR \
  "$IMAGE" \
  bash -lc '
    set -e
    CKPT=$(find /app/data/checkpoints/$MODEL/seed_$SEED -name "best_mIoU_iter_*.pth" 2>/dev/null | sort -t_ -k4 -n | tail -1)
    [ -z "$CKPT" ] && { echo "ERROR: no checkpoint under data/checkpoints/$MODEL/seed_$SEED"; exit 3; }
    echo "checkpoint: $CKPT"
    [ "$CUSTOM_IMPORT" = "1" ] && export PYTHONPATH=$(pwd):$PYTHONPATH
    mim test mmseg config_test.py --checkpoint "$CKPT" \
      --work-dir /app/$OUT \
      --out /app/$OUT/raw_masks \
      --cfg-options \
        test_dataloader.dataset.data_root=/app/data/2026-01-19-defect_dataset \
        test_dataloader.dataset.ann_file=$ANN \
        test_dataloader.dataset.data_prefix.seg_map_path=labels_basic_defects_relabel
  '
echo "=> masks in $OUT/raw_masks/ ; metrics log in $OUT/"
