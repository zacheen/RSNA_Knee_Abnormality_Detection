#!/usr/bin/env bash
# Copy code, labels, train.csv and the Colab runner into the Google Drive folder that colab/runner.ipynb reads.
# Drive for desktop streams files, so this costs no local disk beyond its cache.
# jobs.txt and cache_kernels.txt are seeded from the examples only when absent, so edits made on Drive survive a re-sync.
# Usage: bash tools/sync_colab.sh [drive folder]   default /h/我的雲端硬碟/rsna_knee
set -euo pipefail
cd "$(dirname "$0")/.."
dst=${1:-/h/我的雲端硬碟/rsna_knee}
mkdir -p "$dst/code" "$dst/labels" "$dst/raw" "$dst/runs"
cp src/*.py tools/kernel_output_urls.py "$dst/code/"
bash tools/code_version.sh > "$dst/code/VERSION"
cp data/work/report_labels*.jsonl "$dst/labels/"
cp data/raw/train.csv "$dst/raw/"
# KEEP_RUNNER=1 leaves runner.ipynb alone, because a copy written while it is open in Colab gets overwritten by its autosave.
[ "${KEEP_RUNNER:-0}" = 1 ] || cp colab/runner.ipynb "$dst/"
# runner_A and runner_B differ only in their jobs file, so two Colab sessions can train at once without taking the same job.
for v in A B; do
  sed "s/JOBS_FILE = 'jobs.txt'/JOBS_FILE = 'jobs_$v.txt'/" colab/runner.ipynb > "$dst/runner_$v.ipynb"
  grep -q "jobs_$v.txt" "$dst/runner_$v.ipynb"
  [ -f "$dst/jobs_$v.txt" ] || cp colab/jobs.example.txt "$dst/jobs_$v.txt"
done
[ -f "$dst/jobs.txt" ] || cp colab/jobs.example.txt "$dst/jobs.txt"
[ -f "$dst/cache_kernels.txt" ] || cp colab/cache_kernels.example.txt "$dst/cache_kernels.txt"
ls "$dst" "$dst/code"
