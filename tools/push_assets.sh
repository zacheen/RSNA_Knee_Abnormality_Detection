#!/usr/bin/env bash
# Mirror src/*.py, report labels and offline wheels into the private Kaggle dataset that every cloud kernel mounts.
# Everything sits flat at the dataset root, because the CLI skips subdirectories by default.
# First run:  bash tools/push_assets.sh create
# Afterwards: bash tools/push_assets.sh version "message"
set -euo pipefail
cd "$(dirname "$0")/.."
D=kaggle_datasets/rsna-knee-assets
find "$D" -maxdepth 1 -type f ! -name dataset-metadata.json -delete
cp src/*.py "$D/"
bash tools/code_version.sh > "$D/VERSION"
LABELS=${LABELS:-data/work/report_labels.jsonl}  # the cloud smoke test points this at data/work_smoke/
if [ -f "$LABELS" ]; then cp "$LABELS" "$D/report_labels.jsonl"; fi  # the default file train kernels read
# Every label version also ships under its own name, so a kernel can pick one explicitly. v1 lives in
# report_labels.jsonl locally, which is the alias above, so it is copied as report_labels_v1.jsonl.
if [ -f data/work/report_labels.jsonl ]; then cp data/work/report_labels.jsonl "$D/report_labels_v1.jsonl"; fi
for f in data/work/report_labels_v*.jsonl; do [ -f "$f" ] && cp "$f" "$D/"; done
if [ -d data/wheels ]; then cp data/wheels/*.whl "$D/"; fi
# The CLI builds its upload cache name from -p, and a path with a slash in it breaks that on Windows, hence -p .
case "${1:-version}" in
  create)  (cd "$D" && conda run -n rsna_knee kaggle datasets create -p .) ;;
  version) (cd "$D" && conda run -n rsna_knee kaggle datasets version -p . -m "${2:-update}") ;;
esac
