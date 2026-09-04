#!/bin/bash
# GLM-5.3-Flash-Uncensored-NVFP4 (orcarouter, abliterated; experts-only NVFP4 via Marlin on sm_80) with the
# MTP block transplanted from RedHatAI/GLM-5.3-Flash-NVFP4 (same byte format; FP8-block experts in layer 45).
# The model dir is a composite built by make-nvfp4-mtp-dir.py (symlinks in /hf container paths + merged
# index + RedHat config). Everything else (overlay, PP hand-off, sparse-MLA lane, loop fix) is the shared
# launcher's; this wrapper only sets the identity knobs. Shares the four cards with glm53flash-pp,
# qwen38-pp and dsv4-a100: only one of them can run.
#   container glm53nvfp4-pp, port 8003, served as GLM53Flash-Uncensored, dashboard entry GLM53U.
# KV presets do not apply (different checkpoint): plain util sizing until budgets are derived with kvbudget.py.
# GLM53_MAXLEN defaults to 262144 here until the first discovery boot has been profiled.
set -euo pipefail
export GLM53_NAME=${GLM53_NAME:-glm53nvfp4-pp}
export GLM53_PORT=${GLM53_PORT:-8003}
export GLM53_SERVED=${GLM53_SERVED:-GLM53Flash-Uncensored}
export GLM53_MODEL_DIR=${GLM53_MODEL_DIR:-/home/r/glm53-run/models/glm53-uncensored-nvfp4-mtp}
export GLM53_MAXLEN=${GLM53_MAXLEN:-262144}
export GLM53_PARTITION=${GLM53_PARTITION:-14,11,11,9}   # measured: 4.2 GiB per expert layer; rank 3 carries lm_head + 7.6 GB FP8 drafter + embed (~10 GiB extras)
export GLM53_UTIL=${GLM53_UTIL:-0.95}                    # 60.35 GiB target, under the 61.3 GiB allocator cap; 0.90 left rank 3 with 0.85 GiB of KV
export GLM53_MEM_CAP_FRACTION=${GLM53_MEM_CAP_FRACTION:-0.965}   # clean OOM instead of a box-wedging fault near the top of the card
[ -f "$GLM53_MODEL_DIR/config.json" ] || { echo "model dir $GLM53_MODEL_DIR not built: run /home/r/glm53-run/make-nvfp4-mtp-dir.py" >&2; exit 1; }
exec /home/r/glm53-run/run-glm53flash-pp4.sh "$@"
