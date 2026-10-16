#!/bin/bash
# GLM53UM: OUR GLM-5.3-Flash-Uncensored-NVFP4 composite (orcarouter experts + RedHat MTP block, built by
# make-nvfp4-mtp-dir.py) served by the MORROWMAKE engine (github.com/Morrowmake/glm53-flash-cmp170hx-recipe,
# vLLM 0.32-dev fork with sm_80 kernels; image ghcr.io/morrowmake/vllm-cmp170hx:engine-819146023f) instead of
# our glm53-flash image + overlay. Experiment: do their Ampere kernels help an NVFP4 checkpoint? Their fused
# Marlin MoE decode is W4A16-only (bf16 scales), so the experts stay on upstream's FP4 Marlin path; gains, if
# any, come from the attention/KDA/indexer kernels, PP4 hand-off work and the newer base vLLM.
#   container glm53u-mm, host port 8006 (container 8000), served GLM53Flash-Uncensored, dashboard GLM53UM.
#   LAYOUT=pp4, spec mode mtp (our transplanted drafter; their DFlash2 drafter is NOT used: CC BY-NC-ND and
#   trained on the base model's hidden states), SPEC_N=3, partition 14,11,11,9, util 0.95, 262k context first.
# Three of our overlay fixes are bind-mounted over the fork's source tree (/opt/vllm-src/vllm, editable install):
#   marlin_utils_fp4.py  chunked NVFP4 scale factor (stock builds a 6.75 GiB index per expert layer -> Xid 31
#                        near the top of these cards, wedges every GPU until reboot)
#   gpu_worker.py        GLM53_MEM_CAP_FRACTION allocator cap (clean OOM instead of that fault)
#   mtp.py               drafter loads its private embed_tokens under PP (else 6% acceptance)
# Shares the four cards with every other server here: only one can run.
set -euo pipefail
MM=/home/r/glm53-run/mm
IMG=${GLM53UM_IMG:-ghcr.io/morrowmake/vllm-cmp170hx:engine-819146023f}
NAME=${GLM53UM_NAME:-glm53u-mm}; PORT=${GLM53UM_PORT:-8006}
MODEL_DIR=${GLM53UM_MODEL_DIR:-/home/r/glm53-run/models/glm53-uncensored-nvfp4-mtp}
LAYOUT=${GLM53UM_LAYOUT:-pp4}; SPEC_MODE=${GLM53UM_SPEC_MODE:-mtp}; SPEC_N=${GLM53UM_SPEC:-3}
MAXLEN=${GLM53UM_MAXLEN:-524288}; UTIL=${GLM53UM_UTIL:-0.95}; PARTITION=${GLM53UM_PARTITION:-14,11,11,9}
SERVED=${GLM53UM_SERVED:-GLM53Flash-Uncensored}; REASONING=${GLM53UM_REASONING:-high}
CAP=${GLM53UM_MEM_CAP_FRACTION:-0.965}; TEMPLATE=${GLM53UM_CHAT_TEMPLATE:-/home/r/glm53-run/chat_template-flash-upstream.jinja}
EXTRA=${GLM53UM_EXTRA_ARGS:-}
[ -f "$MODEL_DIR/config.json" ] || { echo "model dir $MODEL_DIR missing (run make-nvfp4-mtp-dir.py)" >&2; exit 1; }
for c in dsv4-a100 qwen38-pp qwen38u-pp glm53flash-pp glm53nvfp4-pp glm53u-mm; do
  [ "$c" = "$NAME" ] && continue
  docker ps --format '{{.Names}}' | grep -qx "$c" && { echo "$c is running and holds the GPUs. Stop it first." >&2; exit 1; }
done
nvidia-smi -q 2>/dev/null | grep -qiE "Recovery Action\s*:\s*(Reboot|Reset)" && { echo "GPU needs reboot/reset (nvidia-smi Recovery Action)" >&2; exit 1; }
timeout 90 docker run --rm --gpus all --entrypoint /opt/venv/bin/python "$IMG" -c "import torch; [torch.ones(1,device=f'cuda:{i}') for i in range(torch.cuda.device_count())]; torch.cuda.synchronize()" >/dev/null 2>&1 \
  || { echo "CUDA init probe failed inside the image (driver wedged? reboot)" >&2; exit 1; }
docker rm -f "$NAME" >/dev/null 2>&1 || true
mkdir -p "$MM/cache/tmp"
V=/opt/vllm-src/vllm
# Their serve.sh only emits --limit-mm-per-prompt when MM_CAP=1, and then forces video count=1; vLLM's default
# without the flag is ONE image per prompt (= per chat session). Pass our own limit via EXTRA_ARGS (last flag wins):
# 64 images, video off (a video request killed the engine on our image; untested on theirs).
MM_LIMIT=${GLM53UM_MM:-'{"image":64,"video":0}'}
EXTRA_ARGS="--chat-template /chat_template.jinja --limit-mm-per-prompt $MM_LIMIT $EXTRA"
docker run -d --name "$NAME" --gpus all --ipc=host --shm-size=32g --user "$(id -u):$(id -g)" --workdir /cache \
  -p "$PORT:8000" \
  -e LAYOUT="$LAYOUT" -e P2P=off -e VENV=/opt/venv -e MODEL=/model -e HOST=0.0.0.0 -e PORT=8000 \
  -e PYTHONDONTWRITEBYTECODE=1 -e RECIPE_CACHE_ROOT=/cache -e TMPDIR=/cache/tmp -e HOME=/cache -e USER=recipe -e LOGNAME=recipe \
  -e XDG_CACHE_HOME=/cache/.cache -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e FLASHINFER_WORKSPACE_BASE=/cache \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor -e TRITON_CACHE_DIR=/cache/.triton/cache -e TORCH_EXTENSIONS_DIR=/cache/.cache/torch_extensions \
  -e VLLM_CACHE_ROOT=/cache/.cache/vllm -e TILELANG_CACHE_DIR=/cache/.tilelang/cache -e TILELANG_TMP_DIR=/cache/.tilelang/cache/tmp \
  -e CUDA_CACHE_PATH=/cache/.nv/ComputeCache -e VLLM_IMAGE_CACHE_SEED=0 \
  -e VLLM_PP_LAYER_PARTITION="$PARTITION" -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 \
  -e GPU_UTIL="$UTIL" -e MAX_LEN="$MAXLEN" -e SPEC_N="$SPEC_N" -e SERVED_MODEL_NAME="$SERVED" -e DEFAULT_REASONING_EFFORT="$REASONING" \
  -e GLM53_MEM_CAP_FRACTION="$CAP" -e EXTRA_ARGS="$EXTRA_ARGS" \
  -v "$MM/recipe/serve.sh:/recipe/serve.sh:ro" -v "$MM/recipe/boot_check.py:/recipe/boot_check.py:ro" \
  -v "$MM/recipe/p2p_check.py:/recipe/p2p_check.py:ro" -v "$MM/recipe/p2p_check.sh:/recipe/p2p_check.sh:ro" -v "$MM/recipe/p2p_probe.cu:/recipe/p2p_probe.cu:ro" \
  -v /home/r/.cache/huggingface:/hf:ro -v "$MODEL_DIR:/model:ro" -v "$MM/cache:/cache" -v "$TEMPLATE:/chat_template.jinja:ro" \
  -v "$MM/overlay/model_executor/layers/quantization/utils/marlin_utils_fp4.py:$V/model_executor/layers/quantization/utils/marlin_utils_fp4.py:ro" \
  -v "$MM/overlay/v1/worker/gpu_worker.py:$V/v1/worker/gpu_worker.py:ro" \
  -v "$MM/overlay/models/glm5next/common/mtp.py:$V/models/glm5next/common/mtp.py:ro" \
  --entrypoint /bin/bash "$IMG" /recipe/serve.sh "$SPEC_MODE" >/dev/null
echo "launched $NAME on :$PORT (Morrowmake engine, $LAYOUT $PARTITION, spec $SPEC_MODE/$SPEC_N, maxlen $MAXLEN, util $UTIL, cap $CAP)"
echo "follow:  docker logs -f $NAME"
