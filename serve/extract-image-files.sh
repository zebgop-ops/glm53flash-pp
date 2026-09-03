#!/bin/bash
# Copy the overlay-relevant vllm files out of an image WITHOUT running it (docker create + cp).
# usage: extract-image-files.sh <image> <outdir>
IMG=$1; OUT=$2; V=/usr/local/lib/python3.12/dist-packages/vllm
set -e
mkdir -p "$OUT"
C=$(docker create "$IMG")
trap 'docker rm "$C" >/dev/null' EXIT
for f in $(cd /home/r/glm53-run/fork && find vllm -type f -name '*.py' | sort); do
  rel=${f#vllm/}
  mkdir -p "$OUT/vllm/$(dirname "$rel")"
  docker cp "$C:$V/$rel" "$OUT/vllm/$rel" 2>/dev/null || echo "  (absent in image) $rel"
done
# also grab things we want to inspect
for extra in v1/attention/backends/mla/xpu_mla_sparse.py v1/attention/backends/registry.py model_executor/layers/sparse_attn_indexer_kpool.py models/glm5next/nvidia/model.py models/glm5next/nvidia/attention.py models/glm5next/nvidia/mtp.py utils/deep_gemm.py platforms/cuda.py model_executor/layers/quantization/inc/inc.py v1/worker/gpu/model_states/mamba_hybrid.py; do
  mkdir -p "$OUT/vllm/$(dirname "$extra")"
  docker cp "$C:$V/$extra" "$OUT/vllm/$extra" 2>/dev/null || echo "  (absent in image) $extra"
done
docker cp "$C:/usr/local/lib/python3.12/dist-packages/vllm/version.py" "$OUT/version.py" 2>/dev/null || true
docker cp "$C:/usr/local/lib/python3.12/dist-packages/vllm/_version.py" "$OUT/_version.py" 2>/dev/null || true
echo "extracted to $OUT"
