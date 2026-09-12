# GLM-5.3-Flash (W4A16) on 4× sm_80: pipeline parallel + MTP + 512k context on vLLM

A runnable serving setup, a Python-only vLLM overlay, and the forensics behind it for
**[Intel/GLM-5.3-Flash-W4A16-AutoRound](https://huggingface.co/Intel/GLM-5.3-Flash-W4A16-AutoRound)**
(GLM-5.3-Flash, 321B total / 18B active, int4 routed experts) on vLLM with
**pipeline parallelism (PP=4), MTP speculative decoding, image input and a 512k-token
context**, on hardware the model officially does not support: 4× **CMP 170HX**
(GA100 silicon, sm_80, 64 GB each after the VRAM unlock, PCIe Gen2 x4, no NVLink/P2P).

The official `vllm/vllm-openai:glm53-flash` image (built from
[vllm#53906](https://github.com/vllm-project/vllm/pull/53906)) supports "Hopper and newer".
Everything here is a bind-mount overlay on that image: no rebuild, no compiled code, and
the builder reproduces the overlay byte-for-byte from the pristine image plus
`patches/`.

Headline numbers (details in **[RESULTS.md](RESULTS.md)**): **66–70 tok/s single-stream
decode, 158 tok/s at 4 streams, 180 at 8**, prefill 3k tok/s at short context and 1.8k
tok/s at 500k, **1.86M-token KV pool at 524,288 context** (3.5 concurrent full-length
requests), MTP acceptance 78/49/25 % by position, exact planted-value recall at 497,052
tokens, and 5-minute sampled soaks with zero degenerate loops.

Sister project (same box, same relay port, shared lessons):
[qwen38-flashnext-pp](https://github.com/zebgop-ops/qwen38-flashnext-pp).

## Quickstart

```bash
./serve/make-patched-tree.sh                       # docker create/cp + patches -> ./patch (never runs the image)
GLM53_HF=/path/to/hf-cache GLM53_PATCH=$PWD/patch ./serve/run-glm53flash-pp4.sh
```

`serve/run-glm53flash-pp4.sh` is the exact production launcher: every flag, env var and
bind-mount, with inline commentary on why each load-bearing flag is load-bearing. Defaults
are the validated production configuration: partition `13,11,11,10`, MTP depth 3,
`FULL_AND_PIECEWISE` CUDA graphs, `--max-model-len 524288`, images on (64 per prompt; the limit is per prompt and a chat
session is one prompt; video off, see below), per-rank KV budgets from the preset table, server default `reasoning_effort=high`
(the template's own default is `max`; requests may pass `low|high|max`). The launcher serves the upstream `zai-org/GLM-5.3-Flash` chat template of 2026-09-04
(`serve/chat_template-flash-upstream.jinja`, `GLM53_CHAT_TEMPLATE`): same reasoning levels and vision handling as
the checkpoints' copy plus three tool-calling fixes (a null assistant content no longer renders as the word "None"). Knobs: `GLM53_MAXLEN`, `GLM53_SPEC`,
`GLM53_CG`, `GLM53_SEQS`, `GLM53_MM`, `GLM53_PARTITION`, `GLM53_KV0..3`, plus the
diagnostic switches described in [FINDINGS.md](FINDINGS.md).

The launcher pre-flights the driver state (a GPU fault on these cards wedges the driver
until reboot; the check refuses to start and says so) and refuses to start while the
box's other servers hold the cards.

## What had to change, and why

Each item cost a boot (about 4.5 minutes here) to find. Patches are per file in
`patches/`, whole files in `ported-files/`.

1. **sm_80 has no sparse-MLA lane.** GLM-5.3's rope-free sparse MLA (head size 512, DSA
   "kpool" indexer) only has FlashMLA/FlashInfer backends in the image, and DeepGEMM is
   absent. The `TRITON_MLA_SPARSE` backend and kpool Triton fallbacks come from
   [Malav-P/vllm@glm53flash-a100](https://github.com/Malav-P/vllm/tree/glm53flash-a100)
   (8×A100, announced in [vllm#54059](https://github.com/vllm-project/vllm/issues/54059)),
   ported hunk by hunk onto the image's older files.
2. **Pipeline parallel was gated off in the model.** The multimodal wrapper lacked
   `make_empty_intermediate_tensors`, and the decoder *dropped the mHC residual streams*
   at the stage boundary. The overlay materializes the deferred hyper-connection
   post-mix on the sending rank, ships the four residual streams flattened, and the
   receiving rank's first layer takes the standalone pre-mix path. Numerically identical
   to the single-rank fused path.
3. **MTP under PP** needed four fixes: the `SupportsPP` gate on the drafter; the drafter's
   experts being built unquantized (the AutoRound/INC block-name prefix does not cover
   `model.layers.45`); the vllm#46994 sampled/draft-token relay (an NCCL hang after graph
   capture otherwise); and, the big one, **the drafter embedding tokens with random
   weights** because vLLM skips embedding sharing under PP and this checkpoint ships no
   MTP embedding. Acceptance went from 6 % to 80 % at the first draft position.
4. **Long context faulted the GPU.** Past ~130k tokens the sm_80 Triton indexer prefill
   scoring kernel raised an illegal memory access *in situ* (clean in isolation, all
   autotune configs). Three engine deaths and three driver wedges to pin down; the
   overlay scores with a chunked torch path above 24k pool rows and keeps Triton below.
   Root cause still open, see FINDINGS.md.
5. **KV cache.** vLLM sizes the pool from the tightest rank; under PP the last rank
   carries the LM head and the drafter and strands 2–3 GiB on every other rank. Per-rank
   budgets (from vLLM's own per-rank maxima, `tools/kvbudget.py`) doubled the pool.
6. **Speed.** The PP4 stages are CPU-dispatch-bound under piecewise CUDA graphs (same
   finding as the Qwen sister project); full graphs took single-stream decode from 45 to
   66–70 tok/s. MTP depth 4 measured worse than 3 at every concurrency.
7. **Constant-token loops in multi-turn chat** (`Think!locklocklock…`). Token 1023 is what
   the sampler emits for all-NaN logits, the same signature as the Qwen sister project's
   `duct` loops: under PP the mamba/KDA spec-decode context captured raw pointers to the
   per-step *gathered* block tables and a non-last rank's deferred postprocess walked them
   with a stale batch mapping, corrupting recurrent state through other requests' block
   ids. Fix ported from qwen38-flashnext-pp patch 0010: the context is handed the
   per-request *source* tables and the copy kernels index by request slot. Trigger is
   prefix-cache hits + several busy slots + MTP, which a chat session produces naturally
   and the fresh-prompt soaks did not.
8. **A second checkpoint, NVFP4 on Ampere** (`orcarouter/GLM-5.3-Flash-Uncensored-NVFP4`,
   served as `GLM53Flash-Uncensored` via `serve/run-glm53flash-uncensored-nvfp4-pp4.sh`).
   NVFP4 runs on sm_80 through vLLM's Marlin FP4 lane, but its load-time scale-factor step
   materialises ~9.5 GB of transient per expert layer (a boolean-mask index over a 3-D
   tensor builds an int64 `[N,3]` index of 6.75 GiB), and **on these VRAM-unlocked cards an
   allocation near the top of the 63.5 GiB card faults instead of OOM-ing**, wedging every GPU
   until reboot. The overlay computes the factor per expert, caps PyTorch's allocator below
   the fault zone (`GLM53_MEM_CAP_FRACTION`, so the worst case is a clean OOM), and the
   repo-dropped MTP block is transplanted from `RedHatAI/GLM-5.3-Flash-NVFP4`
   (`tools/make-nvfp4-mtp-dir.py`): acceptance 75/48/25 %, the base head's numbers.

## Known gaps

- **Video input is off** (`{"video":0}`): this image's video processor expands 3× more
  placeholders than the encoder emits and the engine dies on the mismatch. Upstream
  reworked the video pipeline after the image was built. Image input works.
- **1M context** (the model's native maximum) is not reachable yet: the context-sized
  buffers leave too little KV on the tightest rank with plain sizing, and long-prompt
  transients are unprofiled.
- The **in-situ Triton scoring fault** is worked around, not explained.

## Layout

```
serve/run-glm53flash-pp4.sh     production launcher (all knobs documented inline)
serve/run-glm53flash-uncensored-nvfp4-pp4.sh  wrapper for the NVFP4 checkpoint (partition, cap, model dir)
serve/make-patched-tree.sh      builds the overlay from the image + patches, verifies it
serve/extract-image-files.sh    pull pristine files out of the image without running it
patches/                        unified diffs, one per overlaid vLLM file
ported-files/                   the overlay as whole files (3 are new files from the fork)
tools/                          smoke, bench, needle, soak+detect, hitpath (multi-turn
                                cache-hit soak), spec_stats, mmtest, mmvideo, bracket,
                                kvbudget, kernel_test*, make-nvfp4-mtp-dir (MTP transplant),
                                nvfp4_marlin_test / memsweep / memtop (memory-ceiling forensics)
FINDINGS.md                     the bugs, the evidence, the dead ends
RESULTS.md                      every number, with the configuration it was measured on
```

## License

Apache-2.0, same as vLLM. `ported-files/` and `patches/` are derived from vLLM
(vllm-project, ZJY0516's PR branch, Malav-P's sm_80 fork) and from
zebgop-ops/qwen38-flashnext-pp.

## Quality on this box, measured against the other servers

`tools/quality.py` + `tools/qtable.py` score every model here on identical inputs and the same
1000 MMLU questions (cloze-scored, no chat template; perplexity reported as bits per byte, which
is comparable across tokenizers). Full table and method:
[dsv41reap-pp/CROSS-MODEL.md](https://github.com/zebgop-ops/dsv41reap-pp/blob/main/CROSS-MODEL.md).

| model | wikitext bits/byte | code bits/byte | MMLU (1000 q) |
|---|---|---|---|
| Qwen3.8-Flash-Next FP8 | 0.4394 | 0.0805 | 89.0% |
| GLM-5.3-Flash W4A16 | 0.3884 | 0.1428 | 85.6% |
| DeepSeek-V4.1-Flash | 0.3275 | 0.1044 | 84.4% |
| DeepSeek-V4.1-Flash REAP-272E | 0.4091 | 0.1046 | 76.7% |
