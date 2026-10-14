#!/usr/bin/env bash
# Train ensemble members sequentially on the local GPU with the round-1b recipe on cache v1.
# Each job is "<name> <labels file> <seed> [extra train.py flags]". Checkpoints go to data/models/<name>/, the Kaggle
# download layout, so tools/eval_ensemble.py scores local and cloud members together.
# Usage: bash tools/train_local_queue.sh "L1 report_labels_v2s.jsonl 2" "L4 report_labels_v2s.jsonl 0 --pool conv"
# Cache v3: CACHE="data/work/cache_v3/train_*" RSNA_SLOTS=5 bash tools/train_local_queue.sh ...  (train.py checks the two agree)
set -uo pipefail
# bash reads a script as it runs, so run a copy, or editing this file would change a queue that is already running.
if [ -z "${QUEUE_COPY:-}" ]; then
  tmp=$(mktemp); cp "$0" "$tmp"; cd "$(dirname "$0")/.."; QUEUE_COPY=1 exec bash "$tmp" "$@"
fi
# numpy's Intel Fortran runtime aborts on a console close event ("forrtl error 200"), which killed one run after epoch 1.
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
CACHE=${CACHE:-data/work/cache/train_s0of1}
for job in "$@"; do
  read -r name labels seed extra <<< "$job"
  echo "$(date +%H:%M) start $name labels=$labels seed=$seed cache=$CACHE slots=${RSNA_SLOTS:-3} extra=${extra:-}"
  # shellcheck disable=SC2086  # extra must word-split into flags
  conda run --no-capture-output -n rsna_knee python -u src/train.py --cache "$CACHE" \
    --labels "data/work/$labels" --epochs 12 --bs 4 --workers 4 --seed "$seed" \
    --lr 3e-4 --backbone-lr 3e-4 --wd 1e-4 --dropout 0 --out "data/models/$name/kneenet.pt" ${extra:-} \
    > "data/logs/train_local_$name.log" 2>&1
  echo "$(date +%H:%M) end $name exit=$? $(grep -E '^averaged' "data/logs/train_local_$name.log" | cut -c1-60)"
done
