#!/usr/bin/env bash
# Print the commit hash of HEAD, suffixed -dirty when src/ has uncommitted changes, for the VERSION file that
# push_assets.sh and sync_colab.sh ship next to the code.
set -euo pipefail
cd "$(dirname "$0")/.."
v=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
[ -z "$(git status --porcelain -- src 2>/dev/null)" ] || v="$v-dirty"
echo "$v"
