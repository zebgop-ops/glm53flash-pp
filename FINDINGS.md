# Findings

Chronological forensics for bringing GLM-5.3-Flash (W4A16) up on 4× CMP 170HX with vLLM.
Each section: symptom, evidence, fix, and what was ruled out. Boots cost ~4.5 min each; a
GPU fault on these cards (Xid 31) usually leaves the driver refusing every new CUDA context
("CUDA driver initialization failed") until the host reboots, which is why the launcher
now probes CUDA init before starting.

Environment: `vllm/vllm-openai:glm53-flash` (vLLM `0.1.dev20051+g487ecf187`, torch
2.13+cu130, sm_80 in the arch list, FlashInfer 0.6.17, no DeepGEMM), driver-side VRAM
unlock to 64 GiB per card, PCIe Gen2 x4, no P2P. Checkpoint:
`Intel/GLM-5.3-Flash-W4A16-AutoRound` (`quant_method: auto-round`, packing
`auto_round:auto_gptq`, only the 288 routed experts are int4; attention, shared experts,
dense layers 0–2, indexer and vision tower are BF16). vLLM maps `auto-round` to its `inc`
config, which resolves the experts to GPTQ-Marlin MoE; Marlin's int4 repack works on
these cards.

## 1. No attention backend for sm_80

`ValueError: No valid attention backend found` — every sparse-MLA backend requires
Hopper/Blackwell or FlashInfer ≥ 0.6.18, and the model's MLA is rope-free with head size
512 (`FLASHINFER_MLA` rejects `qk_nope_head_dim=256`). Same wall as
[vllm#54059](https://github.com/vllm-project/vllm/issues/54059) (Ada) and #53963 (SM120).

Fix: the `TRITON_MLA_SPARSE` backend (vllm#47629 + the NoPE geometry from #54031) and the
kpool Triton fallbacks (MQA logits, fp8 stores via uint8 bitcast, tail-cache slot
recompute) from Malav-P's fork. The fork is based on a later state of the PR than the
image, so `patch --fuzz` mis-placed several hunks in `sparse_attn_indexer_kpool.py`
(prefill/decode branches spliced into the wrong `else`); those were redone by hand against
the image's own code. Two fork hunks that read `decode.query_start_loc` (absent in this
build) were replaced by the per-request `group_lens` the same code already computes.

Carried over from the DeepSeek-V4 work on this box: the sm8x radix `persistent_topk`
kernel returns wrong indices when the candidate count lands between k and 2k, so it is
gated on `has_device_capability(90)` in both indexer files (kpool's `select_k=512` hits
the buggy range).

## 2. Pipeline parallel refused, then silently wrong

`'Glm5NextForConditionalGeneration' object has no attribute
'make_empty_intermediate_tensors'`. The model comments say PP is "gated off". Reading the
decoder shows why it could not simply be enabled: under mHC (manifold hyper-connections)
the residual is `hc_mult=4` streams `[T, 4, H]`, and every non-final layer *defers* its
`hc_post` mix into the next layer's fused post+pre kernel. The existing non-last-rank
return shipped `hidden_states` and `residual` and dropped `post`/`comb`, so the receiving
rank would have mixed garbage.

Fix (`models/glm5next/nvidia/model.py`): the sending rank materializes the deferred
`hc_post` of its last layer (exactly what the final layer does before contracting) and
ships the resulting streams flattened to `[T, 4*H]` as the single intermediate tensor;
the receiving rank views it back to `[T, 4, H]` and its first layer runs the standalone
`hc_pre` path with `residual = streams` (the layer-0 path minus the expand). The
multimodal wrapper aliases the hook the way `Glm4vForConditionalGeneration` does.

Follow-up crash: `XPUMLASparseMetadata` lacked `num_decodes/num_prefills/num_decode_tokens`
which the shared MLA layer asserts; ported from the newer PR (all tokens routed through
the sparse MQA path, so the dense-MHA prefill branch is never entered).

## 3. MTP under pipeline parallel

Four separate problems, in the order they surfaced:

1. `Pipeline parallelism is not supported for this model` for the *draft* config: the
   gate inspects the class, so an instance attribute is invisible. `Glm5NextMTP` now
   carries the `SupportsPP` marker (its forward already accepts `intermediate_tensors`).
2. `KeyError: 'model.layers.45.mtp_block.mlp.experts.routed_experts.w2_qweight'`: the INC
   resolver marks a module quantized only when its name starts with a
   `block_name_to_quantize` entry (`model.language_model.layers`, mapped to
   `language_model.model.layers`); the drafter is built under `model.layers.45`, so its
   experts were created unquantized while the checkpoint stores int4. The drafter
   constructor appends its own layers prefix to the block list; the BF16 sub-modules stay
   unquantized through the prefix-agnostic regex entries of `extra_config`.
3. NCCL watchdog timeout on `pp_broadcast` (SeqNum 1, 16 elements) right after graph
   capture: the image predates the vllm#46994 MTP-under-PP relay. Without it the last rank
   sends a width-1 sampled-token tensor against a width-(spec+1) receive, and never relays
   draft tokens at all. The relay (`pp_utils.py`) is a strict superset drop-in from
   qwen38-flashnext-pp plus three runner hunks (relay flag, pop/scatter of relayed drafts,
   `broadcast_draft` after `propose`).
4. **Acceptance 6 % (17 % at position 0).** Outputs were correct, so the target was fine
   and the drafter was producing meaningful-but-wrong drafts. `load_eagle_model` skips
   embedding sharing under PP ("each rank owns its own embedding"); this checkpoint ships
   no MTP embedding (layer 45 has `eh_proj`, `enorm`, `hnorm`, `shared_head.norm` and its
   own MLA/MoE, nothing else), and the target's table lives on rank 0. On rank 3 the
   drafter embedded tokens with random-init weights. The drafter's `load_weights` now
   loads `model.embed_tokens.weight` from the checkpoint when it owns a private embedding.
   Acceptance: 80/48/24 % by position, 2.5 tokens per step. This applies to any MTP
   drafter under PP whose checkpoint lacks its own `embed_tokens`.

Validation of the combination: 40-request sampled soak (temperature 1.0, 7k-token
prompts, 4 streams) with zero degenerate loops, then repeated at 262k and 512k context.

## 4. Long-context GPU faults

Three events, each an Xid 31 MMU fault (write access) followed by a driver that would not
create CUDA contexts until reboot:

| # | config | when | rank |
|---|---|---|---|
| 1 | 262k, MTP 3, piecewise graphs | ~10 s into a 200k prompt | 3 |
| 2 | 262k, MTP off, piecewise | ~10 s into a 200k prompt | 1 |
| 3 | 262k, MTP off, eager, `CUDA_LAUNCH_BLOCKING=1`, bounds checks | 111k needle passed, 200k faulted | 0 and 1 |

Ruled out, in order: the MTP drafter (event 2), a plain length threshold (rank 0 runs
three chunks ahead and faulted last), the drafter's cache writes (bounds asserts on every
kpool slot write never fired), the CUDA indexer K-gather (`GLM53_SAFE_GATHER=1`, a
torch-native gather matching the page layout `[bs×head_dim K bytes][bs×4 scale bytes]`,
still faulted), the Triton scoring kernel *itself* in isolation (`tools/kernel_test2.py`:
all 12 autotune configs, N = 20k…65536 rows, causal bounds, all clean on GPU 0), and
argument corruption (the in-situ arguments logged right before the faulting launch are
ordinary: q fp8 `[M,32,128]` contiguous, k fp8 `[N,128]`, scale f32 `[N]`, w f32
`[M,32]`, ks int32 0, ke int32 ≤ N).

What worked: `GLM53_SAFE_LOGITS=1`, a chunked torch implementation of the same score
(`sum_h relu(q·k) * scale * w` under the causal `[ks, ke)` mask) — 219k and 497k needles
recalled exactly, no fault. The synchronous-launch attribution points at the Triton
launch in situ and nothing before it, so the remaining hypotheses are process-specific
(cross-rank Triton cache interaction, or an in-flight NCCL/side-stream fault landing in
the kernel's window). Open. The overlay defaults to Triton up to 24,000 pool rows (~96k
tokens, thousands of clean calls) and torch above (`GLM53_SAFE_LOGITS=auto|1|0`,
`GLM53_SAFE_LOGITS_N`).

Diagnostic switches kept in the overlay and launcher: `GLM53_SYNC=1`
(`CUDA_LAUNCH_BLOCKING`), `GLM53_BOUNDS_CHECK=1` (host-synchronised range checks on every
kpool slot write and on the draft/target slot mappings; logs gather/scoring shapes),
`GLM53_SAFE_GATHER=1`, `GLM53_SAFE_LOGITS`.

## 5. KV cache under pipeline parallel

vLLM applies one KV budget to all ranks: the tightest rank's. Here rank 3 (LM head +
drafter + its private embedding) had 5.4 GiB available at utilization 0.90 while ranks
0–2 had 7.7–8.9 GiB, and vLLM's own per-rank "maximize KV" suggestions were 14.7 / 13.6 /
13.6 / 11.3 GiB. `gpu_worker.py` (from qwen38-flashnext-pp) reads
`VLLM_KV_CACHE_MEMORY_RANK<i>`; `tools/kvbudget.py` parses the per-rank suggestions from a
discovery boot and applies the margins validated on these cards (0.5 GiB per rank, 1.0
GiB on the drafter rank). Pool: 996k → 1.80M tokens at 262k, 910k → 1.86M at 512k. The
launcher's preset table is keyed by context length and only applies to the exact
partition/MTP/multimodal configuration it was derived for, because an over-committed rank
faults instead of raising a clean OOM.

Other observations: MTP roughly halves the pool at a given context (the speculative slots
enlarge the KDA state page, which forces the aligned attention block to 4,480 tokens);
the vision tower costs ~180k tokens of pool at 262k; raising `max_model_len` changes the
accounting (the pool at 262k was larger than at 64k with identical hardware budgets).

## 6. Speed

The PP4 stages are CPU-dispatch-bound under `PIECEWISE` graphs, as measured on the Qwen
sister project: `FULL_AND_PIECEWISE` took single-stream decode from 45–47 to 66–70 tok/s
and 4-stream aggregate from 110 to 158 tok/s, capture ~100 s and +0.3 GiB per rank, with
the kpool tail cache still running (it executes as eager breaks inside the breakable
CUDA-graph replay). MTP depth 4 was slower than 3 at every concurrency (57 vs 66–70
single-stream; the fourth draft position is accepted 9 % of the time).

## 7. Multimodal

Image input works out of the box once enabled (`--limit-mm-per-prompt`), with the vision
tower on every rank (+1 GiB) and the encoder on rank 0 via FlashAttention. The drafter is
not registered as multimodal-capable, so no cross-rank embedding gather is attempted.
Video is disabled: a 4-second test clip produced 432 encoder tokens against 1,296 prompt
placeholders and the engine died in `_merge_multimodal_embeddings`. The processor and
vision-model diffs against the PR head are cosmetic; the rework is in
`vllm/multimodal/video.py` (2256 → 1287 lines) and a new OpenCV decoder, which post-date
the image.

## 9. Constant-token loops in production chat (state corruption under PP + MTP + cache hits)

2026-09-03, first day of real multi-turn use (prefix-cache hit rate 85 %): one response
degenerated into `Think!locklocklock…` and ran to its token limit; the next request was
fine. `lock` is token id **1023** in this vocabulary, and the Qwen investigation
(qwen38-flashnext-pp FINDINGS, Addendum 11) established by logits dumps that id 1023 is
what vLLM's sampler deterministically emits for an all-NaN logits row (`duct` was 1023 in
Qwen's vocabulary). So this is NaN generation in the target forward, not a model quirk.

Mechanism (proven on the Qwen project by instrumentation and ablation, Addenda 15, 16, 19,
22; the same stock code is in this image): `MambaSpecDecodeGPUContext` captures the
`data_ptr()` of the block tables it is handed *once*, and the V2 runner hands it the
per-step **gathered** tables (batch order). Under PP, `pp_size + 1` steps are in flight and
later steps re-gather into those same buffers, while a non-last rank's deferred
postprocess runs `pp_size` steps after its batch was gathered. Its state-copy kernels then
walk the *current* tables with a *stale* batch mapping, reading and writing KDA state
through other requests' (freed or reallocated) block ids. Corrupted state produces NaN
logits, and a NaN request never recovers.

Why the soaks missed it: fresh random prompts at 4 streams give no prefix-cache hits; with
cache hits the mamba tables are wide and re-gathered every step, which is the exposed
window. A chat session re-sends the whole conversation each turn and is exactly that.

Fix (overlay, from qwen38 patch 0010, two hunks apply with a 5-line offset): the runner
passes `tuple(bt.gpu for bt in self.block_tables.block_tables)` (the per-request source
tables, stable pointers, req-indexed, mutated only by stream-ordered staged writes) to
`preprocess_state`, and the two copy kernels in `mamba_utils.py` index rows by `req_idx`
instead of the batch row. The block-table pool (qwen38 patch 0009) is deliberately *not*
ported: it is optional with 0010 and is a hazard under full CUDA graphs (Addendum 22),
which this deployment uses. The divisor fix (qwen38 patch 0012) is not needed here: in
`align` mode this image sets `mamba_block_size = block_size`.

Validation tool: `tools/hitpath.py` (N conversations × turns over a shared ~20k-token
document, concurrent, sampled, planted-code recall, loop detector). Baseline before the
fix is the production incident; the Qwen baseline for the same trigger was 14–27 % of
responses.

## 8. Operational lessons

- Do not edit the launcher while it is executing: bash reads scripts incrementally, and
  one boot failed with a torn-file syntax error.
- `nvidia-smi` can report "GPU Recovery Action: None" after a fault while every CUDA
  context creation fails; the only reliable preflight is to create one.
- Non-TTY `docker pull` prints no per-layer progress; a quiet pull is not a stalled pull.
