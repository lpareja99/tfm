#!/usr/bin/env bash
# ===========================================================================
# Adverse-weather robustness inference (§4.2) for ANY of the 5 models.
# Single parametrized runner -> supersedes run_swin_weather_cpu.sh / run_flash_weather_gpu.sh.
#
# Usage:
#   bash scripts/run/run_weather.sh <model> [mode] [device] [seed]
#     model  : swin | flash | hrnet | beit | interimage
#     mode   : smoke (3 imgs, default) | full (dry wet half)
#     device : auto (default) | gpu | cpu   (flash/interimage IGNORE this: GPU-only)
#     seed   : checkpoint seed (defaults to each model's known best seed)
#
# Examples:
#   bash scripts/run/run_weather.sh swin smoke
#   bash scripts/run/run_weather.sh flash full
#   nohup bash scripts/run/run_weather.sh interimage full > out_weather/interimage/full.log 2>&1 &
#
# Outputs: out_weather/<model>/<cond>/  (pred_masks/ + vis/ + logs)
# ===========================================================================
set -uo pipefail
# repo root (this script lives in scripts/run/) — works from any CWD
TFM="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="$TFM/data/final_dataset"

MODEL="${1:?model required: swin|flash|hrnet|beit|interimage}"
MODE="${2:-smoke}"
DEVICE="${3:-auto}"
SEED_OVERRIDE="${4:-}"

# ---- per-model table: dir | image | needs_gpu | custom_import | best_seed ----
case "$MODEL" in
  swin)       EXPDIR=swin-T-512x512;              IMAGE=road_defect_base;  NEEDS_GPU=0; CUSTOM_IMPORT=0; DEF_SEED=91  ;;
  flash)      EXPDIR=flashInternImage-T-512x512;  IMAGE=road_defect_flash; NEEDS_GPU=1; CUSTOM_IMPORT=1; DEF_SEED=777 ;;
  interimage) EXPDIR=InterImage-T-512x512;        IMAGE=road_defect_intern;NEEDS_GPU=1; CUSTOM_IMPORT=1; DEF_SEED=777 ;;
  hrnet)      EXPDIR=HRNet-T-512x512;             IMAGE=road_defect_base;  NEEDS_GPU=0; CUSTOM_IMPORT=0; DEF_SEED=91  ;;
  beit)       EXPDIR=BeiT2-T;                     IMAGE=road_defect_base;  NEEDS_GPU=0; CUSTOM_IMPORT=0; DEF_SEED=91  ;;
  *) echo "ERROR: unknown model '$MODEL' (swin|flash|hrnet|beit|interimage)"; exit 2 ;;
esac
SEED="${SEED_OVERRIDE:-$DEF_SEED}"

# ---- resolve device ----
if [ "$NEEDS_GPU" = "1" ]; then
  if [ "$DEVICE" = "cpu" ]; then echo "ERROR: $MODEL requires GPU (custom CUDA op); cannot run on cpu."; exit 2; fi
  DEVICE=gpu
elif [ "$DEVICE" = "auto" ]; then
  DEVICE=gpu   # a local GPU (M1200) is present; pass cpu explicitly if you run out of VRAM
fi
if [ "$DEVICE" = "gpu" ]; then GPU_ARGS="--gpus all"; CVD=0; else GPU_ARGS=""; CVD=""; fi

# ---- check docker image ----
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "ERROR: missing docker image '$IMAGE'."
  [ "$MODEL" = "interimage" ] && echo "  build: docker build -t road_defect_intern -f experiments/$EXPDIR/Dockerfile.local ."
  [ "$MODEL" = "flash" ]      && echo "  build: docker build -t road_defect_flash -f experiments/$EXPDIR/Dockerfile.local experiments/$EXPDIR/custom_modules/ops_dcnv4"
  exit 2
fi

# ---- conditions ----
if [ "$MODE" = "smoke" ]; then
  head -n 3 "$DATA/splits/dry.txt" > "$DATA/splits/smoke.txt"
  CONDS="smoke"; echo ">>> $MODEL | SMOKE (3 imgs) | device=$DEVICE | seed=$SEED"
else
  CONDS="dry wet half"; echo ">>> $MODEL | FULL ($CONDS) | device=$DEVICE | seed=$SEED"
fi
mkdir -p "$TFM/out_weather/$MODEL"

docker run --rm $GPU_ARGS --shm-size=8g \
  -e CUDA_VISIBLE_DEVICES="$CVD" \
  -e MODEL="$MODEL" -e SEED="$SEED" -e CONDS="$CONDS" -e CUSTOM_IMPORT="$CUSTOM_IMPORT" \
  -v "$TFM":/app \
  -w /app/experiments/$EXPDIR \
  "$IMAGE" \
  bash -lc '
    set -e
    # winner checkpoint = latest best_mIoU (highest iter) for the chosen seed
    CKPT=$(find /app/descargas_azure/$MODEL/seed_$SEED -name "best_mIoU_iter_*.pth" 2>/dev/null | sort -t_ -k4 -n | tail -1)
    if [ -z "$CKPT" ]; then echo "ERROR: no checkpoint under descargas_azure/$MODEL/seed_$SEED"; exit 3; fi
    echo "checkpoint: $CKPT"
    [ "$CUSTOM_IMPORT" = "1" ] && export PYTHONPATH=$(pwd):$PYTHONPATH   # for config custom_imports
    for c in $CONDS; do
      echo "======== $MODEL | condition: $c ========"
      mim test mmseg config_test.py --checkpoint "$CKPT" \
        --work-dir /app/out_weather/$MODEL/$c \
        --out /app/out_weather/$MODEL/$c/pred_masks \
        --show-dir vis \
        --cfg-options \
          test_dataloader.dataset.data_root=/app/data/final_dataset \
          test_dataloader.dataset.ann_file=splits/$c.txt \
          test_dataloader.dataset.data_prefix.seg_map_path=labels_720 \
        || echo ">>> FAILED on $c"
    done
  '

echo
echo "=== mIoU SUMMARY ($MODEL) ==="
grep -rhiE "mIoU" "$TFM"/out_weather/$MODEL/*/ 2>/dev/null | grep -iE "aAcc|mIoU" | tail -10
echo "(outputs in out_weather/$MODEL/<cond>/ ; masks in pred_masks/)"
