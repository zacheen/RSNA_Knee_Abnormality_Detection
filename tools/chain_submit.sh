#!/usr/bin/env bash
# Wait for the train kernel, run the submit kernel, then submit it to the competition if its runtime projection fits.
# Usage: bash tools/chain_submit.sh "<submission message>" [train kernel slug] [submit kernel folder]
# Defaults are rsna-knee-train and submit. Round 3 onward uses rsna-knee-train-v2 and submit_v2.
set -uo pipefail
# bash reads a script as it runs, so editing this file would change a chain that is already waiting. Run a copy.
if [ -z "${CHAIN_COPY:-}" ]; then
  tmp=$(mktemp); cp "$0" "$tmp"; cd "$(dirname "$0")/.."; CHAIN_COPY=1 exec bash "$tmp" "$@"
fi
K() { conda run -n rsna_knee kaggle "$@" 2>&1; }
status() { K kernels status "zacheenoao/$1" | tail -1; }
wait_k() { until status "$1" | grep -qvE "RUNNING|QUEUED"; do sleep 60; done; echo "$(date +%H:%M) $(status "$1")"; }
TK=${2:-rsna-knee-train}
SF=${3:-submit}
SK=$(grep -oE 'rsna-knee-[a-z0-9-]+' "kaggle_kernels/$SF/kernel-metadata.json" | head -1)
L=data/kernel_logs/$TK; mkdir -p "$L"

wait_k "$TK"
status "$TK" | grep -q COMPLETE || { echo "train did not complete, stopping"; exit 1; }
K kernels output "zacheenoao/$TK" -p "$L/train" --file-pattern ".*\.log" > /dev/null
grep -o '"data":"[^"]*"' "$L"/train/*.log | sed 's/"data"://' | grep -E "device=|^\"epoch|budget|averaged"

# Kaggle runs at most 2 GPU sessions, so the push can be refused until another kernel ends. Without a version the
# steps below would read the previous run's log and submit nothing valid, so retry, and give up after 6 h.
ver=""
for _ in $(seq 180); do
  out=$(bash tools/push_kernel.sh "$SF" 2>&1)
  ver=$(echo "$out" | grep -oE "version [0-9]+" | grep -oE "[0-9]+" | head -1)
  [ -n "$ver" ] && break
  echo "$(date +%H:%M) push refused: $(echo "$out" | tail -1)"; sleep 120
done
[ -n "$ver" ] || { echo "submit kernel never pushed, stopping"; exit 1; }
echo "submit kernel $SK version $ver"
rm -rf "$L/submit"  # a stale log from an earlier version must not stand in for this one
sleep 60; wait_k "$SK"
status "$SK" | grep -q COMPLETE || { echo "submit kernel did not complete, stopping"; exit 1; }
K kernels output "zacheenoao/$SK" -p "$L/submit" > /dev/null
# Matches the "projected 1300 studies <x>h" text infer.py prints, which is the upper bound. Keep the two in step.
proj=$(grep -oE "projected 1300 studies [0-9.]+h" "$L"/submit/*.log | grep -oE "[0-9.]+" | tail -1)
echo "projected hours $proj"
# The commit run sees 3 studies, so the real 1300-study rerun is only known from this projection. 8 h leaves margin under 9.
awk -v p="$proj" 'BEGIN{exit !(p != "" && p < 8)}' || { echo "projection missing or too slow, not submitting"; exit 1; }

conda run --no-capture-output -n rsna_knee python tools/submit_code.py "$ver" "${1:-baseline}" "zacheenoao/$SK"
