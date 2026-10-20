#!/usr/bin/env bash
# Push kaggle_kernels/submit_two as it stands, wait for the commit run, and submit only when group A loaded 30
# checkpoints, group B loaded the expected count and both parts merged. A push refused for a busy GPU retries every 10 min.
# Usage: bash tools/submit_two.sh <expected B checkpoints> "<description>"
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
nb=$1; msg=$2; k=zacheenoao/rsna-knee-submit-two
while :; do
  out=$(bash tools/push_kernel.sh submit_two 2>&1 | tail -1); echo "$(date +%H:%M) submit_two $out"
  grep -q "session count" <<< "$out" || break; sleep 600
done
ver=$(grep -oE "version [0-9]+" <<< "$out" | grep -oE "[0-9]+"); [ -n "$ver" ] || exit 1
sleep 120
until conda run -n rsna_knee kaggle kernels status $k 2>&1 | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 120; done
O=data/logs/submit_two_v$ver; mkdir -p "$O"
conda run -n rsna_knee kaggle kernels output $k -p "$O" >/dev/null 2>&1
grep -ahoE "a [0-9]+ b [0-9]+|lower bound[^\"]{0,30}|merged[^\"]{0,40}|Traceback|[A-Za-z]*Error[^\"]{0,120}" "$O"/*.log | sort -u
if grep -aq "a 30 b $nb" "$O"/*.log && grep -aq "merged 2 parts" "$O"/*.log && [ -s "$O/submission.csv" ]; then
  conda run --no-capture-output -n rsna_knee python tools/submit_code.py "$ver" "$msg" $k 2>&1 | tail -1
else echo "NOT SUBMITTED"; fi
