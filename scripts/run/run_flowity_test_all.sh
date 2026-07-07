#!/usr/bin/env bash
# ===========================================================================
# Queue ALL Flowity test runs (5 models x 5 seeds) sequentially, one GPU
# container at a time. IDEMPOTENT: skips any (model, seed) whose raw_masks/
# folder is already complete (723 masks). Safe to stop and re-launch.
#
#   Run in background:  nohup bash scripts/run/run_flowity_test_all.sh > out_flowity_queue.log 2>&1 &
#
# Produces raw_masks/ (for the qualitative notebook, winner seeds) and per-run
# metric logs (parsed later into testing_analytics_<model>.xlsx).
# ===========================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
TOT=723

declare -A EXP=(
  [swin]=swin-T-512x512
  [beit]=BeiT2-T
  [hrnet]=HRNet-T-512x512
  [flash]=flashInternImage-T-512x512
  [interimage]=InterImage-T-512x512
)
MODELS="swin beit hrnet flash interimage"
SEEDS="42 91 777 1337 2026"

echo "===== FLOWITY TEST QUEUE START $(date '+%Y-%m-%d %H:%M') ====="
for m in $MODELS; do
  for s in $SEEDS; do
    d="experiments/${EXP[$m]}/outputx5/seed_$s/test/raw_masks"
    n=$(ls "$d" 2>/dev/null | wc -l)
    if [ "$n" -ge "$TOT" ]; then
      echo "SKIP  $m seed_$s  (already complete: $n masks)"
      continue
    fi
    echo "----- RUN  $m seed_$s  ($(date '+%H:%M')) -----"
    bash scripts/run/run_flowity_test.sh "$m" full "$s" 2>&1 \
      | grep -aE "checkpoint:|Iter\(test\) \[723|mIoU:|ERROR|masks in" | tail -4 \
      || echo ">>> FAILED $m seed_$s"
  done
done
echo "===== FLOWITY TEST QUEUE DONE $(date '+%Y-%m-%d %H:%M') ====="
