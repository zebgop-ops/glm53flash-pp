#!/bin/bash
# Build the bind-mount overlay tree from the official image + this repo's patches.
#   ./serve/make-patched-tree.sh [image] [outdir]
# Extracts the pristine files with `docker create` + `docker cp` (the image is never executed),
# applies patches/*.patch, drops in the three new files from ported-files/, and verifies the
# result against ported-files/ byte-for-byte. Then point the launcher at it: GLM53_PATCH=<outdir>.
set -euo pipefail
IMG=${1:-vllm/vllm-openai:glm53-flash}
OUT=${2:-$(pwd)/patch}
HERE=$(cd "$(dirname "$0")/.." && pwd)
V=/usr/local/lib/python3.12/dist-packages/vllm
mkdir -p "$OUT"
C=$(docker create "$IMG"); trap 'docker rm "$C" >/dev/null' EXIT
for p in "$HERE"/patches/*.patch; do
  rel=$(grep -m1 "^+++ b/" "$p" | cut -f1 | sed "s|^+++ b/||")
  mkdir -p "$OUT/$(dirname "$rel")"
  docker cp "$C:$V/${rel#vllm/}" "$OUT/$rel"
  (cd "$OUT" && patch -p1 --forward -N < "$p" >/dev/null) || { echo "FAILED: $p" >&2; exit 1; }
done
for rel in vllm/v1/attention/backends/mla/triton_mla_sparse.py vllm/v1/attention/ops/mqa_logits_triton.py vllm/v1/attention/ops/triton_mla_sparse_kernel.py; do
  mkdir -p "$OUT/$(dirname "$rel")"; cp "$HERE/ported-files/$rel" "$OUT/$rel"
done
find "$OUT" -name '*.orig' -delete
bad=0
for f in $(cd "$HERE/ported-files" && find vllm -name '*.py'); do
  cmp -s "$HERE/ported-files/$f" "$OUT/$f" || { echo "MISMATCH vs ported-files: $f" >&2; bad=1; }
done
[ $bad = 0 ] && echo "overlay tree at $OUT matches ported-files/ ($(find "$OUT" -name '*.py' | wc -l) files)"
