#!/usr/bin/env bash
# Download cache zips from their build kernels, extract into $DEST (default data/work/cache), delete each zip once extracted.
# The server drops long transfers (schannel "server closed abruptly"), and curl's own retry budget runs out on a
# 10 GB file. So the outer loop keeps resuming with a freshly signed URL until the size on disk equals the size the
# server reports. Rerunning the script also resumes.
# Usage: [DEST=data/work/cache_v4] bash tools/fetch_cache.sh <kernel slug>...   e.g. rsna-knee-build-cache
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${DEST:-data/work/cache}  # one folder per cache version, so train globs never mix layouts
mkdir -p "$DEST"
urls() { conda run -n rsna_knee python tools/kernel_output_urls.py "zacheenoao/$1" 'cache_train.*\.zip' | tr -d '\r'; }
for k in "$@"; do
  d=data/cloud_cache/$k  # one folder per kernel, so a loop never touches another kernel's zip
  mkdir -p "$d"
  names=$(urls "$k" | awk '{print $1}')
  [ -n "$names" ] || { echo "no cache zip in $k" >&2; exit 1; }
  for name in $names; do
    f=$d/$name
    total=$(curl -sS -L -r 0-0 -o /dev/null -D - "$(urls "$k" | awk -v n="$name" '$1==n{print $2}')" \
            | tr -d '\r' | awk -F/ 'tolower($0) ~ /^content-range/ {print $2}')
    [ -n "$total" ] || { echo "could not read the size of $name" >&2; exit 1; }
    while [ "$(stat -c %s "$f" 2>/dev/null || echo 0)" -lt "$total" ]; do
      echo "$(date +%H:%M) $name $(( $(stat -c %s "$f" 2>/dev/null || echo 0) / 1048576 )) of $(( total / 1048576 )) MB"
      url=$(urls "$k" | awk -v n="$name" '$1==n{print $2}')
      curl -sS -L --fail --retry 5 --retry-delay 5 --retry-all-errors -C - -o "$f" "$url" || sleep 10
    done
    echo "$(date +%H:%M) extracting $name ($(du -h "$f" | cut -f1))"
    unzip -q -o "$f" -d "$DEST" || { echo "extract failed, kept $f" >&2; exit 1; }
    rm -f "$f"
  done
done
echo "$(date +%H:%M) done"
du -sh "$DEST"/* 2>/dev/null
