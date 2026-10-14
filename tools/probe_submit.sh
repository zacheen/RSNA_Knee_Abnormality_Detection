#!/usr/bin/env bash
# Submit one leaderboard probe. Sets MEMBERS in kaggle_kernels/submit_probe/body.py, commits it, pushes the kernel,
# waits for the commit run, and submits only when every expected checkpoint loaded and the 1300-study projection
# (its lower bound, see below) is under 6 h.
# Usage: bash tools/probe_submit.sh "<description>" W8 [W2 ...]     VERSION=<n> skips the push and submits version n
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
msg=$1; shift
k=zacheenoao/rsna-knee-submit-probe
list=$(printf '"%s", ' "$@"); list="[${list%, }]"
ver=${VERSION:-}
if [ -z "$ver" ]; then
  sed -i "s/^MEMBERS = .*/MEMBERS = $list/" kaggle_kernels/submit_probe/body.py
  git add kaggle_kernels/submit_probe/body.py
  git commit -q -m "Probe the leaderboard with $*" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || true
  out=$(bash tools/push_kernel.sh submit_probe 2>&1 | tail -1); echo "$out"
  ver=$(grep -oE "version [0-9]+" <<< "$out" | grep -oE "[0-9]+")
  [ -n "$ver" ] || { echo "push failed"; exit 1; }
  sleep 90
fi
while :; do
  s=$(conda run -n rsna_knee kaggle kernels status $k 2>&1 | tail -1)
  case "$s" in *COMPLETE*|*ERROR*|*CANCEL*) break;; esac
  sleep 60
done
o=data/logs/probe_$(echo "$*" | tr ' ' '_')
mkdir -p "$o"
conda run -n rsna_knee kaggle kernels output $k -p "$o" >/dev/null 2>&1
n=$(grep -ahoE "loaded [0-9]+" "$o"/*.log | grep -oE "[0-9]+")
# Gate on the lower bound. At 3 studies the upper bound charges model loading to every study, so it overshoots: one
# 15-checkpoint run projected 6.02 h by the upper bound and finished scoring in under 1.3 h.
h=$(grep -ahoE "lower bound [0-9.]+" "$o"/*.log | grep -oE "[0-9.]+$" | tail -1)
echo "$(date +%H:%M) version $ver $s members=$* loaded=$n projected=$h"
# Every member ships 3 checkpoints, the last epochs train.py keeps by default (--avg-last 3).
if [ "$n" = $(( 3 * $# )) ] && [ -s "$o/submission.csv" ] && awk "BEGIN{exit !(${h:-99} < 6)}"; then
  conda run --no-capture-output -n rsna_knee python tools/submit_code.py "$ver" "$msg" $k 2>&1 | tail -1
else
  echo "NOT SUBMITTED"
fi
