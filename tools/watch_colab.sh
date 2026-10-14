#!/usr/bin/env bash
# Poll the Colab runs on Drive and exit as soon as one of the named jobs finishes or stalls, so a background caller
# wakes up for it. A stall (train.log untouched for STALL_MIN minutes and no DONE) also raises a Windows toast, because
# restarting a disconnected runtime is something only the user can do. It is filed under the RSNA Knee category.
# Exit codes 0 done, 2 stalled, 3 failed (last log line is an error).
# Usage: bash tools/watch_colab.sh W4 W5 [...]   env STALL_MIN (default 40), POLL_SEC (default 300)
set -uo pipefail
runs=${RUNS:-/h/我的雲端硬碟/rsna_knee/runs}
stall=${STALL_MIN:-40}
notify() {
  powershell.exe -NoProfile -File "D:\dont_move\git_save\Daily_Task\shared\notify.ps1" -Message "$1" -Title "RSNA Knee 警告" -Category "RSNA Knee" >/dev/null
}
t0=$(date +%s)
while :; do
  for job in "$@"; do
    d="$runs/$job"
    if [ -f "$d/DONE" ]; then
      echo "$(date +%H:%M) done $job $(grep -E '^averaged' "$d/train.log" | cut -c1-60)"
      exit 0
    fi
    if [ -f "$d/train.log" ] && tail -1 "$d/train.log" | grep -qE '^[A-Za-z]*(Error|Exception)\b'; then
      # The runner stops at a failed job and its last cell releases the runtime, so the user has nothing to restart.
      echo "$(date +%H:%M) failed $job, last line $(tail -1 "$d/train.log" | cut -c1-80)"
      notify "Colab 的 $job 訓練失敗，runtime 已經自動釋放，不需要你處理"
      exit 3
    fi
    if [ -f "$d/train.log" ]; then
      m=$(stat -c %Y "$d/train.log"); [ "$m" -gt "$t0" ] || m=$t0  # a log already stale when watching starts gets a fresh grace period
      age=$(( ($(date +%s) - m) / 60 ))
      if [ "$age" -ge "$stall" ]; then
        echo "$(date +%H:%M) stalled $job, train.log untouched for $age min, last line $(tail -1 "$d/train.log" | cut -c1-80)"
        notify "Colab 的 $job 已經 $age 分鐘沒有進度，runtime 可能斷線了，請打開 Colab 看一下"
        exit 2
      fi
    fi
  done
  sleep "${POLL_SEC:-300}"
done
