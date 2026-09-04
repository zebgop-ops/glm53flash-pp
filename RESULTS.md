# Results

All numbers measured 2026-09-02/03 on 4× CMP 170HX (sm_80, 64 GiB each, PCIe Gen2 x4, no
P2P, 200 W), Threadripper PRO 5955WX host on the `powersave` governor, with
`Intel/GLM-5.3-Flash-W4A16-AutoRound` under `vllm/vllm-openai:glm53-flash` plus this
overlay. Decode figures are 300-token generations, warm (first run after boot discarded);
`bench.py C` = C concurrent streams, aggregate tok/s and per-stream in parentheses.
Prefill figures come from single needle prompts and include the sparse indexer's
context-dependent work.

## Production configuration (launcher defaults)

Partition 13,11,11,10 · MTP depth 3 · `FULL_AND_PIECEWISE` CUDA graphs ·
`--max-model-len 524288` · 8 slots · images on / video off · per-rank KV preset.

| metric | value |
|---|---|
| KV pool | 1,856,853 tokens (3.54× a full 512k request) |
| decode, 1 stream | 66–70 tok/s |
| decode, 4 streams | 158 tok/s (39 per stream) |
| decode, 8 streams | 180 tok/s (22.5 per stream) |
| MTP acceptance | 51 % of draft tokens; 78 / 49 / 25 % by position; ~2.5 tokens per step |
| prefill, short context | ~3.2k tok/s (111k-token prompt) |
| prefill, 497k-token prompt | 1.8k tok/s |
| planted-value recall | exact at 31.9k, 60.2k, 111.5k, 219.2k and 497.1k tokens |
| soak (4 streams, temperature 1.0, 7k-token prompts) | 36 requests in 5 min, 0 degenerate loops |
| hit-path soak (12 conversations × 6 turns over a shared 20k-token document, 6 concurrent, temperature 1.0, MTP on, prefix-cache hits) | 0 loops / 72 responses, 11/12 planted-code recalls, after the §9 fix (before it: production loop within the first day) |
| hit-path soak, larger (16 conversations × 8 turns, 8 concurrent) | 0 loops / 128 responses; 14/16 recalls, both misses were the model *refusing* to relay the planted code as "injected content" (correct code visible in its reasoning), not corruption |
| per-rank memory (weights + non-torch) | 44.7 / 45.8 / 45.8 / 48.2 GiB; rank 3 at 63.1/64 GiB with KV |
| boot | ~95 s weight load, ~100 s graph capture, ~5 min to ready |

## GLM53U: orcarouter/GLM-5.3-Flash-Uncensored-NVFP4 + transplanted MTP (first day)

Partition 14,11,11,9 · MTP depth 3 (RedHat's layer 45, FP8 experts) · `FULL_AND_PIECEWISE` ·
`--max-model-len 262144` · utilization 0.95 · allocator cap 0.965 · images on.

| metric | value |
|---|---|
| per-rank consumed (weights + non-torch) | 50.3 / 47.6 / 47.6 / 49.0 GiB; peak activation ~4.7 GiB per rank |
| KV pool | 1,310,720 tokens (5.0× a full 262k request); tightest rank is rank 0 (vision tower + embedding), per-rank budgets gain <0.5 GiB |
| decode, 1 stream | 59–67 tok/s |
| decode, 4 streams | 166 tok/s (41.6 per stream) |
| MTP acceptance (base head on the abliterated target) | 49 %; 75 / 48 / 25 % by position; ~2.5 tokens per step |
| image test | exact ("Blue square, upper left / Red circle, lower right") |
| hit-path soak (12 conversations × 6 turns, 6 concurrent, temperature 1.0, cache hits) | 0 loops / 72 responses, **12/12** planted-code recalls (the abliterated model relays the code the base model refused as "injected content") |
| boot | ~100 s weight load, ~6 min to ready |

## How the numbers moved

| step | context | KV pool | decode 1 stream | decode 4 streams | notes |
|---|---|---|---|---|---|
| first working boot, no MTP, text-only | 64k | 1,893,262 | 34 | 95 | piecewise graphs |
| + MTP depth 3 (before the embedding fix) | 64k | 703,313 | 24 | 64 | acceptance 6 % |
| + drafter embedding fix | 64k | 701,714 | 55 | 123 (151 @6, 162 @8) | acceptance 78/49/25 % |
| + vision tower (images on) | 64k | 607,406 | 47 | — | +1 GiB per rank |
| context 262k, hybrid scoring | 262k | 996,147 | 45 | 109 | |
| + per-rank KV budgets | 262k | 1,801,083 | 49 | 145 | |
| context 512k, plain sizing | 512k | 910,222 | 46 | — | 497k needle exact |
| + per-rank KV budgets | 512k | 1,856,853 | 46 | 110 | |
| + full CUDA graphs (**production**) | 512k | 1,856,853 | **66–70** | **158** (180 @8) | |
| MTP depth 4 (rejected) | 512k | 851,528 | 57 | 146 (155 @8) | position-3 acceptance 9 % |

Context at 1,048,576 (the model's native maximum) failed KV sizing with plain
utilization: 1.81 GiB available on rank 3 against 4.6 GiB needed for one request; vLLM's
estimate for that sizing is a 340k maximum.

## Per-rank KV picture (why the budgets matter)

At 262k / MTP 3 / images on / utilization 0.90, vLLM's per-rank suggestions:

| rank | consumed (weights + non-torch) | KV at util 0.90 | KV maximum (all free memory) | preset budget |
|---|---|---|---|---|
| 0 | 43.4 GiB | 8.88 GiB | 14.74 GiB | 14.24 GiB |
| 1 | 44.5 | 7.73 | 13.61 | 13.11 |
| 2 | 44.6 | 7.70 | 13.58 | 13.08 |
| 3 | 46.9 | 5.43 | 11.31 | 10.31 |

The default pool is bounded by rank 3's 5.43 GiB applied to every rank. The preset gives
each rank its own maximum minus 0.5 GiB (1.0 GiB on rank 3, which also carries the
drafter's transient scratch). The 512k preset was derived the same way
(13.45 / 12.32 / 12.29 / 10.01 GiB maxima).

## Image input

A generated 256×256 PNG (blue square top-left, red circle bottom-right) was described
exactly ("Blue square, upper left / Red circle, lower right") from a 132-token prompt.
Video input is disabled (see FINDINGS.md §7).

## Cost of each speculative token position

Under full graphs at 512k, depth 3: positions accepted 78 / 49 / 25 % of the time. Depth
4 adds a fourth position accepted 9 % of the time and loses 13 % single-stream and 14 % at
8 streams, so depth 3 is the default.
