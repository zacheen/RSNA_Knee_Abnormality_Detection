#!/usr/bin/env bash
# Build kaggle_kernels/<stage>/run.py from common/bootstrap.py, <stage>/includes and body.py, then push. run.py is generated, edit body.py.
# Usage: bash tools/push_kernel.sh <stage folder>
set -euo pipefail
cd "$(dirname "$0")/../kaggle_kernels"
inc=()
if [ -f "$1/includes" ]; then mapfile -t inc < <(tr -d '\r' < "$1/includes"); fi  # a CRLF includes file would leave \r in each path
cat common/bootstrap.py "${inc[@]}" "$1/body.py" > "$1/run.py"
conda run -n rsna_knee kaggle kernels push -p "$1"
