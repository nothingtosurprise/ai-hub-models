#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

set +e
umask 022

LOG=/data/local/tmp/QDC_logs
OUT=$LOG/results
MM_CACHE=/data/local/tmp/geniex-cache
TC=/data/local/tmp/TestContent

mkdir -p "$LOG" "$OUT" "$MM_CACHE"
# Keep stdout off the SSH channel: `exec > >(tee ...) 2>&1` lets the process
# substitution stall the channel until its 4 KiB stdout buffer fills, then
# QDC's pyshell-bash-ssh runner trips sshd's ClientAliveInterval=15/CountMax=4
# (~73s) and tears the session down mid-cell. File-only redirect avoids it.
exec > "$LOG/script.log" 2>&1
date -u
uname -a

TGZ=$TC/geniex-bench.tar.gz
URL="{LINUX_BENCH_URL}"
curl -fSL --retry 3 --retry-delay 5 -o "$TGZ" "$URL"
tar -xzf "$TGZ" -C "$TC"
rm -f "$TGZ"

BUNDLE=$(ls -d "$TC"/geniex-bench-linux-arm64-* 2>/dev/null | head -n1)
[ -n "$BUNDLE" ] || { echo "FATAL: extracted bundle dir missing under $TC"; exit 1; }

[ -x "$BUNDLE/bin/geniex-bench" ] || chmod +x "$BUNDLE/bin/geniex-bench" 2>/dev/null
[ -x "$BUNDLE/bin/geniex-bench" ] || { echo "FATAL: $BUNDLE/bin/geniex-bench missing"; exit 1; }

cd "$BUNDLE"
export LD_LIBRARY_PATH="$BUNDLE/lib:$BUNDLE/lib/llama_cpp:$BUNDLE/lib/qairt:$LD_LIBRARY_PATH"
export GENIEX_PLUGIN_PATH="$BUNDLE/lib"

# geniex-bench fails randomly on QDC devices; give each invocation one
# retry before letting the failure propagate.
geniex_retry() {
    "$@" || {
        echo "geniex_retry: command failed, retrying once: $*" >&2
        "$@"
    }
}

IMG=$TC/test.png

declare -A TSV
# shellcheck disable=SC2043  # {CTX_LIST} is a python-substituted placeholder expanding to "512 1024 4096"
for ctx in {CTX_LIST}; do
  TSV[$ctx]=/data/local/tmp/matrix-$ctx.tsv
  : > "${TSV[$ctx]}"
done

while IFS='|' read -r name plugin devs model_id vlm image; do
  [ -z "$name" ] && continue
  echo "=== plan $name id=$model_id ==="
  imgpath=""
  [ "$image" = "1" ] && imgpath="$IMG"
  IFS=','
  for d in $devs; do
    # shellcheck disable=SC2043
    for ctx in {CTX_LIST}; do
      printf '%s-%s-%s-c%s\t%s\t%s\t%s\t\t\t%s\t%s\n' \
        "$name" "$plugin" "$d" "$ctx" "$plugin" "$d" "$model_id" "$imgpath" "$vlm" \
        >> "${TSV[$ctx]}"
    done
  done
  IFS='|'
done <<'EOF'
{MODELS}
EOF

failed_ctxs=""
# shellcheck disable=SC2043
for ctx in {CTX_LIST}; do
  tsv="${TSV[$ctx]}"
  echo "=== matrix ctx=$ctx ==="
  cat "$tsv"
  geniex_retry ./bin/geniex-bench --matrix-file "$tsv" --output-json-dir "$OUT" -r 3 \
    {BENCH_SIZE_FLAGS} \
    --mm-data-dir "$MM_CACHE" --chipset "{CHIPSET}"
  bench_rc=$?
  echo "rc=$bench_rc  ($(ls "$OUT" | wc -l) cell json files so far)"
  [ "$bench_rc" -ne 0 ] && failed_ctxs="$failed_ctxs $ctx"
done

echo "=== done ==="
if [ -n "$failed_ctxs" ]; then
  echo "FATAL: geniex-bench failed for context lengths:$failed_ctxs"
  exit 1
fi
exit 0
