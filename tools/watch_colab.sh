#!/usr/bin/env bash
# Poll the Colab runs on Drive and exit as soon as one of the named jobs finishes or stalls, so a background caller
# wakes up for it. A stall (train.log untouched for STALL_MIN minutes and no DONE) also raises a Windows toast, because
# restarting a disconnected runtime is something only the user can do. tools/notify.ps1 files it under the alert header.
# Usage: bash tools/watch_colab.sh W4 W5 [...]   env STALL_MIN (default 40), POLL_SEC (default 300)
set -uo pipefail
runs=${RUNS:-/h/我的雲端硬碟/rsna_knee/runs}
stall=${STALL_MIN:-40}
notify() {
  powershell.exe -NoProfile -File "D:\dont_move\git_save\RSNA_Knee_Abnormality_Detection\tools\notify.ps1" -Message "$1" -Category alert >/dev/null
}
while :; do
  for job in "$@"; do
    d="$runs/$job"
    if [ -f "$d/DONE" ]; then
      echo "$(date +%H:%M) done $job $(grep -E '^averaged' "$d/train.log" | cut -c1-60)"
      exit 0
    fi
    if [ -f "$d/train.log" ]; then
      age=$(( ($(date +%s) - $(stat -c %Y "$d/train.log")) / 60 ))
      if [ "$age" -ge "$stall" ]; then
        echo "$(date +%H:%M) stalled $job, train.log untouched for $age min, last line $(tail -1 "$d/train.log" | cut -c1-80)"
        notify "Colab 的 $job 已經 $age 分鐘沒有進度，runtime 可能斷線了，請打開 Colab 看一下"
        exit 2
      fi
    fi
  done
  sleep "${POLL_SEC:-300}"
done
