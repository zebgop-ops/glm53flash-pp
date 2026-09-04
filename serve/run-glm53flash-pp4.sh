#!/bin/bash
# GLM-5.3-Flash W4A16 (Intel/GLM-5.3-Flash-W4A16-AutoRound: int4 routed experts, BF16 rest)
# on 4x CMP 170HX (sm_80, 64 GiB, PCIe Gen2 x4, no P2P): PIPELINE PARALLEL 4 (+ MTP).
# Served as GLM53Flash on :8002, container glm53flash-pp. Dashboard: servers-top.py (GLM53).
#
# IMAGE: official vllm/vllm-openai:glm53-flash (vLLM PR #53906, built 2026-08-26) plus a
#   Python-only overlay bind-mounted from patch/vllm/ (pristine copies: image-official/).
#   The overlay = Malav-P/vllm@glm53flash-a100 (TRITON_MLA_SPARSE backend + kpool Triton
#   fallbacks for sm_80, vllm#54059) ported hunk-by-hunk onto the image, plus our fixes:
#   - model.py: PP hand-off of the mHC residual streams (materialize the deferred hc_post,
#     ship [T, 4*H]); the image gated PP off and dropped the streams.
#   - xpu_mla_sparse.py: num_decodes/num_prefills/num_decode_tokens on the metadata.
#   - mtp.py: SupportsPP marker; drafter experts resolved as int4 (INC block prefix);
#     drafter loads its OWN embed_tokens from the checkpoint (under PP vLLM skips
#     embedding sharing and the table lives on rank 0 -> random-init drafter embedding
#     = 6% acceptance; with the fix 80/48/24% at N=3).
#   - pp_utils.py + model_runner.py: vllm#46994 MTP-under-PP relay (from qwen38-run):
#     sampled-width padding + third broadcast for draft tokens (fixes the NCCL hang on
#     pp_broadcast after graph capture).
#   - sparse_attn_indexer*.py: sm8x radix persistent_topk gated off (DSv4 lesson).
# FLAGS THAT ARE LOAD-BEARING: --attention-backend TRITON_MLA_SPARSE (only sparse-MLA lane
#   on sm_80; DeepGEMM absent -> Triton MQA-logits fallback), --kv-cache-dtype bfloat16
#   (Triton lane is bf16-only). CUDA graphs: FULL_AND_PIECEWISE (default since 2026-09-03) --
#   the PP4 stages are CPU-dispatch-bound under PIECEWISE (same finding as qwen38-run): 45-47 ->
#   66-70 tok/s single-stream, 110 -> 158 agg @4, 180 @8; capture ~100 s, +0.3 GiB/rank; validated
#   200k needle exact + 5-min 4-stream soak 36/36 clean. The kpool tail cache runs as eager breaks
#   inside the breakable-cudagraph replay, so full graphs do not skip it.
# PARTITION 13,11,11,10: layers 0-2 are dense (~0.35 GiB), 3-44 MoE (~4.1 GiB each); rank 0
#   also holds embed (1.3 GiB), rank 3 lm_head (1.3) + MTP drafter (~5 GiB incl. its own
#   embedding). Measured: 40.9 / 41.9 / 42.0 / 44.4 GiB (with MTP). Over-committing a rank
#   faults instead of OOM and can wedge GSP firmware -> check fit before changing this.
# MEASURED (64k ctx, 2026-09-02, warm): no-MTP 34 tok/s single / 95 agg @4, KV pool
#   1,893,262 tokens; MTP N=3 (seqs 8): 55 single / 123 agg @4 / 151 @6 / 162 @8 (per-stream
#   55 / 31 / 25 / 20), acceptance 78/49/25%, pool 701,714 text-only / 607,406 with the
#   vision tower + encoder cache (multimodal default; ranks 42.0/43.0/43.0/45.4 GiB) (KDA spec-state pages + prefix
#   caching 'align' inflate attention blocks to 4480 tokens -- prefix caching OFF should
#   recover most of it). Needles exact at 31.9k and 60.2k tokens (prefill ~3-3.7k tok/s).
#   Native context is 1,048,576 (NoPE, no YaRN); GLM53_MAXLEN raises it (pool permitting).
# LONG CONTEXT (resolved 2026-09-03): the Xid 31 faults past ~130k tokens (three engine deaths,
#   three driver wedges/reboots) were the sm_80 Triton indexer PREFILL SCORING kernel
#   (_fp8_mqa_logits_kernel) failing in situ once N > ~32k pools -- with MTP on or off, under
#   CUDA graphs or eager. It is clean in isolation (all 12 autotune configs, N to 65536), the
#   CUDA gather was cleared with a torch-native gather, and the logged arguments are ordinary,
#   so the root cause is still open. The overlay now scores with a chunked torch path above
#   GLM53_SAFE_LOGITS_N=24000 pools (~96k tokens) and keeps Triton below it (thousands of clean
#   calls). Validated at MAXLEN=262144, MTP 3, PIECEWISE, vision on: needles exact at 111k and
#   219k tokens (prefill 3.2k / 2.8k tok/s), no fault; decode 45 tok/s single / 109 agg @4 (vs
#   55 / 123 at 64k). Pool at 262k: 996,147 tokens (3.8x) with
#   MTP+vision at util 0.90 (rank 3 binds: vLLM's per-rank maxima 14.7/13.6/13.6/11.3 GiB vs the
#   5.4 GiB rank 3 gets at 0.90). PER-RANK BUDGET PRESET (default since 2026-09-03): maxima minus
#   0.5/0.5/0.5/1.0 GiB -> pool 1,801,083 tokens (6.9x 262k, +81%); validated: 219k needle exact,
#   49 tok/s single / 145 agg @4, 16-request soak clean, no OOM, rank 3 at 62.8/64 GiB.
#   512k (DEFAULT since 2026-09-03): with the 524288 preset -> pool 1,856,853 tokens (3.5x 512k, same
#   as the 262k preset pool); validated 497,052-token needle exact (prefill 1.8k tok/s), 46 tok/s
#   single / 110 agg @4, 16-request soak clean, no OOM, rank 3 at 63.1/64 GiB. Plain util sizing at
#   512k gives only 910,222 tokens (rank 3 gets 4.1 GiB) -- the preset is what makes 512k useful.
#   1M (native max) NOT reachable yet: at MAXLEN=1048576 the context-sized buffers leave 1.81 GiB
#   of KV on rank 3 at util 0.90 vs 4.6 GiB needed for ONE 1M request (vLLM's estimate: 340k max
#   with plain sizing). Would need per-rank budgets re-derived at that setting (kvbudget.py from a
#   GLM53_KV_PRESET=0 boot with lower MAXLEN) and long-prompt transients are unprofiled -- untested.
#   Diagnostic knobs kept: GLM53_SYNC, GLM53_BOUNDS_CHECK, GLM53_SAFE_GATHER, GLM53_SAFE_LOGITS,
#   GLM53_MARLIN_DIAG (NVFP4 builds: log the scale tensor handed to the Marlin scale-factor step).
# Knobs: GLM53_PARTITION GLM53_GPUS GLM53_UTIL GLM53_MAXLEN GLM53_SEQS GLM53_CG GLM53_SPEC GLM53_MM GLM53_REASONING
#        GLM53_KV0..3 (per-rank KV budgets in bytes)  GLM53_BOUNDS_CHECK=1 (diagnostic slot range checks)
#        GLM53_SYNC=1 (CUDA_LAUNCH_BLOCKING=1: faults surface at the culprit kernel; slow, diagnostic only)
#        GLM53_SAFE_GATHER=1 (torch-native indexer K gather instead of the CUDA cp_gather op)
#        GLM53_SAFE_LOGITS=auto|1|0 (indexer prefill scoring: Triton up to GLM53_SAFE_LOGITS_N=24000
#          pools, torch above; the in-situ Triton kernel faults past ~130k tokens -- see header)
#        GLM53_EXTRA_ARGS (e.g. --max-num-batched-tokens 8192, --no-enable-prefix-caching)
#        GLM53_PORT GLM53_NAME GLM53_IMG GLM53_PATCH
# Reasoning: the chat template ALWAYS thinks. Its own default effort is "max"; the launcher
#   sets the SERVER default to "high" (GLM53_REASONING, since 2026-09-03) via
#   --default-chat-template-kwargs; requests can still pass reasoning_effort low|high|max
#   (top-level field or chat_template_kwargs). Measured on one arithmetic prompt: max 118
#   completion tokens, high 54-68, low 41 -- all correct.

NAME=${GLM53_NAME:-glm53flash-pp}
PORT=${GLM53_PORT:-8002}
SERVED=${GLM53_SERVED:-GLM53Flash}      # --served-model-name (the NVFP4 wrapper serves GLM53Flash-Uncensored)
IMG=${GLM53_IMG:-vllm/vllm-openai:glm53-flash}
HFCACHE=${GLM53_HF:-/home/r/.cache/huggingface}
SNAPSHOT=${GLM53_SNAPSHOT:-5eee1846f0321058ed73745f9aa16f2aaf0fc0a0}
INTEL_MODEL="/hf/hub/models--Intel--GLM-5.3-Flash-W4A16-AutoRound/snapshots/$SNAPSHOT"
# GLM53_MODEL: container path of a different checkpoint (e.g. another GLM-5.3-Flash quant under /hf/hub/...).
# The KV presets below are measured for the Intel W4A16 checkpoint ONLY and are skipped for any other model
# (an over-committed rank faults instead of OOM-ing cleanly) -> a new model boots with plain util sizing;
# derive its own budgets with kvbudget.py from that discovery boot.
MODEL_OVERRIDE=${GLM53_MODEL:-}
# GLM53_MODEL_DIR: HOST directory (e.g. a composite dir built by make-nvfp4-mtp-dir.py, symlinks in /hf
# container paths) mounted read-only at /model and served from there. Implies the same preset skip.
MODEL_DIR=${GLM53_MODEL_DIR:-}
if [ -n "$MODEL_DIR" ]; then MODEL_OVERRIDE=/model; MODEL_DIR_MOUNT=(-v "$MODEL_DIR:/model:ro"); else MODEL_DIR_MOUNT=(); fi
# 45 decoder layers: 0-2 dense (BF16, ~0.35 GiB each), 3-44 MoE (~4.1 GiB each, int4
# experts); rank 0 also holds embed (1.3 GiB), last rank holds lm_head (1.3 GiB) and,
# with MTP, the drafter layer (~4.1 GiB). 13,11,11,10 -> ~42/45/45/46 GiB weights.
PARTITION=${GLM53_PARTITION:-13,11,11,10}
PP=$(echo "$PARTITION" | tr "," "\n" | wc -l)
GPU_ORDER=${GLM53_GPUS:-all}
UTIL=${GLM53_UTIL:-0.90}
MAXLEN=${GLM53_MAXLEN:-524288}   # validated 2026-09-03 (497k needle exact, MTP + graphs, per-rank KV preset); 262144 also has a preset; native max 1048576 not yet reachable
SEQS=${GLM53_SEQS:-8}
SPEC_N=${GLM53_SPEC:-3}            # MTP draft depth (0 = off). N=3 validated; N=4 measured worse (57 vs 66-70 C1, 146 vs 158 @4, 155 vs 180 @8; pos-3 acceptance 9%)
MM=${GLM53_MM:-'{"image":64,"video":0}'}   # --limit-mm-per-prompt JSON; {"image":0,"video":0} = text-only. video=0 until the
                                          # video placeholder mismatch is fixed (a video request KILLS the engine otherwise).
CG=${GLM53_CG:-FULL_AND_PIECEWISE}  # validated 2026-09-03: 66-70 tok/s single (piecewise 45-47), 158 agg @4, 180 @8; 200k needle exact, soak 36/36 clean
EXTRA_ARGS=${GLM53_EXTRA_ARGS:-}
# Server-side default for the template's reasoning effort (the template's own default is
# "max"; only "low"|"high" change it). Per-request `reasoning_effort` / chat_template_kwargs
# still override. Set GLM53_REASONING=max to restore the model-card default.
REASONING=${GLM53_REASONING:-high}
PATCHDIR=${GLM53_PATCH:-/home/r/glm53-run/patch}
# Per-rank KV budgets (bytes; overlay gpu_worker.py reads VLLM_KV_CACHE_MEMORY_RANK<i>).
# Unset = vLLM's own profiling at GLM53_UTIL on every rank (the pool is then bounded by
# the tightest rank). Set from the "Available KV cache memory" lines of a discovery boot.
# Presets (GLM53_KV_PRESET, default 1; 0 = plain util-based sizing): per-rank budgets = vLLM's
# per-rank maxima from a discovery boot of the SAME configuration (partition 13,11,11,10, MTP 3,
# vision on) minus 0.5 GiB (1.0 GiB on the drafter rank) -- the qwen38-run margins validated on
# these cards. Keyed by MAXLEN; re-derive with kvbudget.py after ANY change to partition, MTP
# depth or multimodal settings: a budget that over-commits a rank faults instead of raising a
# clean OOM.
P0=; P1=; P2=; P3=
if [ "${GLM53_KV_PRESET:-1}" = "1" ] && [ -z "$MODEL_OVERRIDE" ] && [ "$PARTITION" = "13,11,11,10" ] && [ "$SPEC_N" = "3" ] \
   && [ "$MM" = '{"image":64,"video":0}' ]; then
  case "$MAXLEN" in
    262144) P0=15289155584; P1=14076575744; P2=14046167040; P3=11065227264 ;;  # maxima 14.74/13.61/13.58/11.31 GiB -> pool 1,801,083 (validated 2026-09-03)
    524288) P0=13900840960; P1=12688261120; P2=12657852416; P3=9674815488 ;;   # maxima 13.45/12.32/12.29/10.01 GiB (discovery 2026-09-03: plain sizing pool 910,222, 497k needle exact)
  esac
fi
KV0=${GLM53_KV0:-$P0}; KV1=${GLM53_KV1:-$P1}; KV2=${GLM53_KV2:-$P2}; KV3=${GLM53_KV3:-$P3}
KV_ENV=(${KV0:+-e VLLM_KV_CACHE_MEMORY_RANK0=$KV0} ${KV1:+-e VLLM_KV_CACHE_MEMORY_RANK1=$KV1} ${KV2:+-e VLLM_KV_CACHE_MEMORY_RANK2=$KV2} ${KV3:+-e VLLM_KV_CACHE_MEMORY_RANK3=$KV3})

V=/usr/local/lib/python3.12/dist-packages/vllm
MODEL=${MODEL_OVERRIDE:-$INTEL_MODEL}

# Overlay: every file under $PATCHDIR/vllm/ is mounted read-only over the same path in
# the image's vllm package.
MOUNTS=()
if [ -d "$PATCHDIR/vllm" ]; then
  while IFS= read -r f; do
    rel=${f#"$PATCHDIR/vllm/"}
    MOUNTS+=(-v "$f:$V/$rel:ro")
  done < <(find "$PATCHDIR/vllm" -type f -name '*.py' | sort)
fi

for c in dsv4-a100 qwen38-pp glm53flash-pp glm53nvfp4-pp; do
  [ "$c" = "$NAME" ] && continue   # the same server restarting is fine; any OTHER server holds the cards
  if docker inspect -f '{{.State.Running}}' $c 2>/dev/null | grep -q true; then
    echo "$c is running and holds the GPUs. Stop it first." >&2; exit 1
  fi
done

# Preflight: after a GPU fault (Xid 31/154) the driver refuses new CUDA contexts until the
# host reboots; every boot would fail at NCCL init ("CUDA driver initialization failed").
if nvidia-smi -q 2>/dev/null | grep -qE "GPU Recovery Action\s*:\s*(Reboot|Reset)"; then
  echo "nvidia-smi reports 'GPU Recovery Action: Reboot/Reset' -- the driver is wedged after a" >&2
  echo "GPU fault. Reboot the host before starting (see launcher header, 2026-09-02 Xid 31)." >&2
  exit 1
fi
# The status line above is not sufficient: after the 2nd Xid 31 it read 'None' while every
# CUDA context creation failed ("CUDA driver initialization failed"). Probe for real (~10 s).
if ! timeout 90 docker run --rm --gpus all --entrypoint python3 "$IMG" -c \
     "import torch; [torch.ones(1, device=f'cuda:{i}').sum().item() for i in range(torch.cuda.device_count())]" \
     >/dev/null 2>&1; then
  echo "CUDA context creation fails inside $IMG -- the driver is wedged (GPU fault). Reboot the host." >&2
  exit 1
fi

docker stop -t 60 "$NAME" >/dev/null 2>&1
docker rm "$NAME" >/dev/null 2>&1

SPEC_ARGS=()
if [ "$SPEC_N" -gt 0 ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_N}")
fi

docker run -d --name "$NAME" --gpus "$([ "$GPU_ORDER" = all ] && echo all || echo "\"device=$GPU_ORDER\"")" --ipc=host --shm-size=32g \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=5400 \
  -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 \
  -e VLLM_PP_LAYER_PARTITION="$PARTITION" \
  "${KV_ENV[@]}" \
  ${GLM53_BOUNDS_CHECK:+-e GLM53_BOUNDS_CHECK=$GLM53_BOUNDS_CHECK} \
  ${GLM53_MARLIN_DIAG:+-e GLM53_MARLIN_DIAG=$GLM53_MARLIN_DIAG} \
  ${GLM53_SYNC:+-e CUDA_LAUNCH_BLOCKING=1} \
  ${GLM53_SAFE_GATHER:+-e GLM53_SAFE_GATHER=1} \
  ${GLM53_SAFE_LOGITS:+-e GLM53_SAFE_LOGITS=$GLM53_SAFE_LOGITS} ${GLM53_SAFE_LOGITS_N:+-e GLM53_SAFE_LOGITS_N=$GLM53_SAFE_LOGITS_N} \
  -v "$HFCACHE":/hf:ro \
  "${MODEL_DIR_MOUNT[@]}" \
  "${MOUNTS[@]}" \
  -p "$PORT":8000 \
  "$IMG" "$MODEL" --served-model-name "$SERVED" \
  --pipeline-parallel-size "$PP" \
  --attention-backend TRITON_MLA_SPARSE --kv-cache-dtype bfloat16 \
  --gpu-memory-utilization "$UTIL" --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" \
  --limit-mm-per-prompt "$MM" \
  -cc.cudagraph_mode=$CG --no-enable-flashinfer-autotune \
  $EXTRA_ARGS \
  "${SPEC_ARGS[@]}" \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
  --default-chat-template-kwargs "{\"reasoning_effort\":\"$REASONING\"}" \
  >/dev/null

echo "launched $NAME on :$PORT  (PP$PP $PARTITION, cudagraph $CG, maxlen $MAXLEN, seqs $SEQS, mtp $SPEC_N, mm $MM, overlay files: $((${#MOUNTS[@]}/2)))"
echo "follow:  docker logs -f $NAME"
