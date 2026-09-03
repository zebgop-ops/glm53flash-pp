# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.models.glm5next.nvidia.ops.kpool_compress import (
    expand_pools_and_append_tail,
    expand_pools_to_tokens,
    kpool_compress_and_write_cache,
    kpool_decode_update_and_maybe_write_cache_batched,
    kpool_seed_tail_cache,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
    is_deep_gemm_supported,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._xpu_ops import xpu_ops

logger = init_logger(__name__)

# glm53-run diagnostic: GLM53_BOUNDS_CHECK=1 turns every cache-slot write in this op into a
# host-synchronised range check that raises a Python error naming the offending buffer,
# instead of letting an out-of-range slot become an MMU fault (Xid 31) that wedges the
# driver until reboot. Costs one sync per call; leave unset in production.
_BOUNDS_CHECK = os.environ.get("GLM53_BOUNDS_CHECK") == "1"
# GLM53_SAFE_GATHER=1 replaces the CUDA cp_gather_indexer_k_quant_cache op with a
# torch-native gather (bounds-safe by construction; slower). Diagnostic/fallback for
# the long-context Xid 31 seen on 2026-09-02/03 (fault surfaced at the first error
# check after this gather).
_SAFE_GATHER = os.environ.get("GLM53_SAFE_GATHER") == "1"
# GLM53_SAFE_LOGITS=1 replaces the Triton prefill MQA-logits kernel with a chunked torch
# implementation (slow; diagnostic). GLM53_BOUNDS_CHECK=1 also logs the exact arguments of
# every prefill logits launch with N > 24000 pools (the in-situ fault window).
# GLM53_SAFE_LOGITS: "1" = always torch, "0" = always Triton, unset/"auto" = Triton up to
# _SAFE_LOGITS_N pools, torch above. Measured 2026-09-03: the Triton kernel is clean in
# isolation at every autotune config up to N=65536, yet in situ it raised an illegal
# memory access on every run once a prompt passed ~130k tokens (N > ~32k pools) --
# three engine deaths, two driver wedges. With the torch path a 219k-token needle recalled
# exactly. Short contexts have thousands of clean Triton calls (soaks, needles to 111k).
_SAFE_LOGITS_MODE = os.environ.get("GLM53_SAFE_LOGITS", "auto")
_SAFE_LOGITS_N = int(os.environ.get("GLM53_SAFE_LOGITS_N", "24000"))


def _use_safe_logits(n: int) -> bool:
    if _SAFE_LOGITS_MODE == "1":
        return True
    if _SAFE_LOGITS_MODE == "0":
        return False
    return n > _SAFE_LOGITS_N


def _mqa_logits_torch(q_fp8, k_fp8, k_scale, weights, ks, ke, n_chunk=2048):
    """logits[m, n] = sum_h relu(q[m,h,:] . k[n,:]) * k_scale[n] * w[m,h] for ks[m] <= n < ke[m],
    else -inf. q [M,H,D] fp8, k [N,D] fp8, k_scale [N] f32, weights [M,H] f32, ks/ke [M] int32."""
    M, H, D = q_fp8.shape
    N = k_fp8.shape[0]
    q = q_fp8.to(torch.bfloat16).reshape(M * H, D)
    k = k_fp8.to(torch.bfloat16)
    w = weights.to(torch.float32)
    out = torch.full((M, N), float("-inf"), dtype=torch.float32, device=q.device)
    n_idx = torch.arange(N, device=q.device, dtype=torch.int32)
    valid = (n_idx[None, :] >= ks[:, None]) & (n_idx[None, :] < ke[:, None])
    for n0 in range(0, N, n_chunk):
        n1 = min(N, n0 + n_chunk)
        s_ = (q @ k[n0:n1].T).view(M, H, n1 - n0).to(torch.float32) * k_scale[None, None, n0:n1]
        s_ = torch.relu(s_) * w[:, :, None]
        out[:, n0:n1] = torch.where(valid[:, n0:n1], s_.sum(dim=1), float("-inf"))
    return out
_GATHER_LOGGED = [0]


def _gather_k_torch(kv_cache, dst_k, dst_scale, block_table, cu_seq_lens, head_dim):
    """Torch-native replacement for ops.cp_gather_indexer_k_quant_cache.
    kv_cache: 3-D [num_blocks, block_size, head_dim+4] uint8; INSIDE a page the layout is
    [block_size x head_dim K bytes] followed by [block_size x 4 fp32-scale bytes]
    (S_OFFSET_NBYTES_IN_PAGE = block_size*head_dim, see kpool_compress.py), NOT per-row.
    dst_k [T, head_dim] fp8, dst_scale [T, 4] uint8; cu_seq_lens [R+1] (pool units);
    block_table [R, max_blocks]."""
    num_blocks, block_size, row_bytes = kv_cache.shape
    dev = kv_cache.device
    cu = cu_seq_lens.to(torch.int64)
    lens = cu[1:] - cu[:-1]
    total = int(cu[-1].item())
    if total == 0:
        return
    req_of_pool = torch.repeat_interleave(torch.arange(lens.shape[0], device=dev), lens)
    pos_in_req = torch.arange(total, device=dev) - cu[:-1][req_of_pool]
    blk_col = pos_in_req // block_size
    off = pos_in_req % block_size
    if _BOUNDS_CHECK:
        assert int(blk_col.max().item()) < block_table.shape[1], (
            f"GLM53 gather: block column {int(blk_col.max().item())} >= table width "
            f"{block_table.shape[1]} (block_size={block_size}, max_len={int(lens.max().item())})")
    blk = block_table.to(torch.int64)[req_of_pool, blk_col]
    if _BOUNDS_CHECK:
        assert int(blk.max().item()) < num_blocks and int(blk.min().item()) >= 0, (
            f"GLM53 gather: block id range [{int(blk.min().item())},{int(blk.max().item())}] "
            f"outside num_blocks={num_blocks}")
    pages = kv_cache.view(num_blocks, block_size * row_bytes)          # [B, page_bytes]
    k_region = pages[:, : block_size * head_dim].view(num_blocks, block_size, head_dim)
    s_region = pages[:, block_size * head_dim :].view(num_blocks, block_size, 4)
    dst_k.view(torch.uint8)[:total].copy_(k_region[blk, off])
    dst_scale[:total].copy_(s_region[blk, off])


def _bc(tag: str, slots: torch.Tensor, capacity: int) -> None:
    if not _BOUNDS_CHECK or slots is None or slots.numel() == 0:
        return
    s64 = slots.to(torch.int64)
    mx = int(s64.max().item())
    mn = int(s64.min().item())
    if mx >= capacity or mn < -1:
        raise RuntimeError(
            f"GLM53_BOUNDS_CHECK: {tag}: slot range [{mn}, {mx}] outside capacity "
            f"{capacity} (numel={slots.numel()})"
        )

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32

# kpool write helper: form pools from the current token batch and compress them
# into the index K cache via the fused Triton kernel.


def _kpool_compress_insert(
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kpool: int,
    head_dim: int,
    round_scale: bool,
) -> None:
    """Pool ``kpool`` consecutive tokens into one fp8 K and write at pool slots.

    ``slot_mapping`` is pool-granular (compress_ratio == kpool on the spec):
    only the *last* token of each complete pool carries a valid (>=0) slot;
    intra-pool tokens are -1. Every position is treated as a pool-completion
    candidate and non-completions are masked off inside the kernel. Compacting
    the valid rows first (``torch.nonzero`` + boolean-mask gather + an
    ``ok.all()`` check) costs two device syncs on the eager prefill path and
    buys nothing numerically. Assumes pool-aligned chunk starts (same
    invariant as sglang).
    """
    n = slot_mapping.shape[0]
    # No pool can complete in a batch smaller than one pool; also keeps the
    # clamped gather indices below in bounds.
    if n < kpool:
        return
    pos = torch.arange(n, device=k.device)
    valid = slot_mapping >= 0
    # Drop pools whose start falls before the batch (leading padding); their
    # gate/k data is undefined anyway.
    write_mask = valid & (pos >= kpool - 1)
    offs = torch.arange(kpool, device=k.device)
    idx = (pos - (kpool - 1)).clamp_min(0)[:, None] + offs[None, :]
    kpool_compress_and_write_cache(
        kv_cache,
        k[idx],  # [n, kpool, head_dim]
        gate_score[idx],
        ape,
        slot_mapping.to(torch.int64),
        pool_size=kpool,
        head_dim=head_dim,
        write_mask=write_mask,
        round_scale=round_scale,
        write_cache=True,
        return_compressed=False,
    )


def _build_decode_scatter_indices(
    decode_lens: torch.Tensor,
    num_requests: int,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token (request id, intra-request index) for a non-uniform decode
    batch, with ``n == decode_lens.sum()`` as a host int (avoids a
    device sync and keeps both repeat_interleaves sync-free).

    Shared by every ``_scatter_decode_tokens_by_request`` call in a step:
    building it per call would repeat the same repeat_interleave/cumsum chain
    up to 5x per layer on the eager decode break.
    """
    device = decode_lens.device
    dl = decode_lens.to(torch.int64)
    req_id = torch.repeat_interleave(
        torch.arange(num_requests, device=device, dtype=torch.int64),
        dl,
        output_size=n,
    )
    req_starts = torch.cumsum(
        torch.cat([torch.zeros(1, device=device, dtype=torch.int64), dl[:-1]]),
        dim=0,
    )
    # Broadcast the per-request start offsets to per-token (length n ==
    # dl.sum()) so each token's intra-request index subtracts its own
    # request's start.
    starts = torch.repeat_interleave(req_starts, dl, output_size=n)
    intra = torch.arange(n, device=device, dtype=torch.int64) - starts
    return req_id, intra


def _scatter_decode_tokens_by_request(
    tokens: torch.Tensor,
    pad_value,
    num_requests: int,
    lmax: int,
    scatter_indices: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Group ``[N, ...]`` decode tokens into a padded ``[num_requests, lmax, ...]``
    layout: request ``r``'s tokens at row ``r`` in order; short requests padded.

    Unlike ``pack_seq_triton`` this is dtype-agnostic (needed for the int32
    slot/pos tensors) — it scatters with the shared per-step indices from
    ``_build_decode_scatter_indices``. Used only for the non-uniform
    (``requires_padding``) decode batch; uniform batches use a zero-copy
    reshape.
    """
    req_id, intra = scatter_indices
    out = torch.full(
        (num_requests, lmax, *tokens.shape[1:]),
        pad_value,
        dtype=tokens.dtype,
        device=tokens.device,
    )
    out[req_id, intra] = tokens
    return out


def _decode_topk_seq_lens(
    positions: torch.Tensor,
    decode_lens: torch.Tensor,
    num_decode_tokens: int,
    batch_size: int,
    next_n: int,
    requires_padding: bool,
) -> torch.Tensor:
    """Token-granular seq_len (pos + 1) per pool-topk row, layout-aware.

    ``pool_topk`` (and the logits it comes from) follow the padded
    ``[batch_size, next_n]`` grid whenever ``requires_padding`` is set, so row
    ``(b, t)`` corresponds to flat decode token ``offset_b + t`` -- NOT
    ``b * next_n + t``. Slicing flat ``positions[: batch_size * next_n]``
    (the uniform-layout shortcut) misaligns every row after the first
    non-uniform request and, past the decode region, reads prefill tokens'
    positions; ``expand_pools_and_append_tail`` then anchors the tail at
    another request's length, dropping the row's real tail tokens or emitting
    indices past its sequence (out-of-bounds block-table reads). Padded rows
    get 0 (empty tail); they are dropped by ``unpack_seq_triton`` anyway.
    """
    n = batch_size * next_n
    if not requires_padding:
        return positions[:n].to(torch.int32) + 1
    scatter_idx = _build_decode_scatter_indices(
        decode_lens, batch_size, num_decode_tokens
    )
    padded = _scatter_decode_tokens_by_request(
        positions[:num_decode_tokens].to(torch.int32),
        -1,
        batch_size,
        next_n,
        scatter_idx,
    )
    return padded.reshape(n) + 1  # pad rows: -1 + 1 = 0 -> empty tail


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


@eager_break_during_capture
def sparse_attn_indexer_kpool(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    # kpool params (Plan-A: gate is consumed at write time and read back at
    # topk time to softmax-weight the pool).
    gate_score: torch.Tensor | None = None,
    compress_ape: torch.Tensor | None = None,
    index_kpool: int = 1,
    positions: torch.Tensor | None = None,
    # Paged tail cache (in-progress pool's raw K + gate score), replacing the
    # transient _DECODE_TAIL ring. tail_prefix resolves attn_metadata[tail_prefix]
    # for the tail group's token-granular slot_mapping. None on the dummy/profiling
    # path and when the tail cache is disabled.
    tail_kv_cache: torch.Tensor | None = None,
    tail_prefix: str | None = None,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # The CUDA cp_gather / indexer_k_quant_and_cache kernels assume a 3-D
    # kv_cache [num_blocks, block_size, cache_stride] and use kv_cache.size(1)
    # as cache_block_size.  With LBHNC layout the per-layer slice is 4-D
    # [num_blocks, num_heads, block_size, head_dim]; num_heads is always 1 for
    # MLA, so collapsing it makes size(1) == block_size and stride(0) unchanged.
    if kv_cache.dim() == 4:
        kv_cache = kv_cache.view(kv_cache.shape[0], -1, kv_cache.shape[-1])

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Sentinel allocation so the profiler's peak-memory measurement covers
        # the runtime logits tensor. The decode-path fp8_fp4_paged_mqa_logits
        # output is [B*next_n, max_model_len] float32 -- sized by max_model_len,
        # NOT bounded by the prefill chunk cap. This profiling branch returns
        # the fake before ever calling that kernel, so its output tensor is
        # invisible unless we size this sentinel to the real worst-case decode
        # batch; otherwise large max_model_len / max_num_batched_tokens OOMs at
        # warmup (the old fixed VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512MiB was
        # ~10x too small at max_model_len=1M / b8192).
        cfg = get_current_vllm_config_or_none()
        worst_decode_tokens = 0
        if cfg is not None:
            sched = cfg.scheduler_config
            num_spec = (
                cfg.speculative_config.num_speculative_tokens
                if cfg.speculative_config is not None
                else 0
            )
            worst_decode_tokens = min(
                sched.max_num_seqs * (num_spec + 1),
                sched.max_num_batched_tokens,
            )
        # float32 logits -> 4 bytes/element; uint8 sentinel so elems == bytes.
        decode_logits_elems = worst_decode_tokens * max_model_len * 4
        prefill_cap_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        max_logits_elems = max(decode_logits_elems, prefill_cap_elems)
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_kpool_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        if index_kpool > 1 and gate_score is not None and compress_ape is not None:
            # kpool prefill write: pool kpool consecutive prefill tokens via
            # softmax(gate+ape)-weighted sum -> Hadamard -> fp8 -> pool slots.
            # Decode tokens (the first num_decode_tokens in the batch) cannot be
            # pooled here — their pool's earlier tokens are not in this batch —
            # so they are deferred to the tail-buffer kernel in has_decode.
            # compress_ratio == index_kpool makes slot_mapping pool-granular.
            n_prefill = num_tokens - num_decode_tokens
            if n_prefill > 0:
                # decode tokens are batched first; prefill tokens follow.
                prefill_slice = slice(num_decode_tokens, num_tokens)
                _bc("kpool prefill compress slots", slot_mapping[prefill_slice],
                    kv_cache.shape[0] * kv_cache.shape[1])
                _kpool_compress_insert(
                    k[prefill_slice],
                    gate_score[prefill_slice],
                    compress_ape,
                    kv_cache,
                    slot_mapping[prefill_slice],
                    index_kpool,
                    head_dim,
                    round_scale=(scale_fmt is not None),
                )
                # Persist the prefill tail (trailing incomplete pool's raw K +
                # gate score) into the paged tail cache, so the decode side can
                # compress the boundary pool correctly -- including across PD
                # transfer, where the connector ships this block. Their tail
                # slots land at offsets pos % kpool of the request's tail block,
                # exactly where the decode reconstruction reads them.
                #
                # This must run PER REQUEST (sglang writes the tail inside its
                # per-request extend loop, `set_compress_tail_for_request`).
                # Taking the batch's trailing `n_prefill % kpool` tokens only
                # covers the LAST request: every other request in a
                # multi-request prefill batch then compresses its boundary pool
                # against a stale tail block (the ring is reused across
                # requests), corrupting one pool at each request's
                # prompt->decode boundary. Invisible on single-request probes;
                # hit by every concurrent-serving batch.
                if (
                    tail_kv_cache is not None
                    and tail_prefix is not None
                    and os.environ.get("VLLM_KPOOL_SKIP_TAIL_CACHE") != "1"
                ):
                    tail_meta = attn_metadata.get(_resolve_layer_name(tail_prefix))
                    if tail_meta is not None:
                        assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
                        # Seed each request's trailing <= kpool raw K + gate
                        # into the paged tail ring with one kernel. The old
                        # scatter chain (block-id compare + nonzero/boolean
                        # gathers + 2 indexed writes) cost ~12 elementwise ops
                        # and 4 device syncs per layer; the kernel derives the
                        # same per-request tail membership in-kernel: token i
                        # is in its request's tail iff the token kpool ahead
                        # maps to a different tail block (1 block/req) or is
                        # past the batch. Writes are one-per-token to distinct
                        # pos % kpool offsets, so the result is identical.
                        _bc("kpool prefill tail slots", tail_meta.slot_mapping[prefill_slice],
                            tail_kv_cache.shape[0] * index_kpool)
                        kpool_seed_tail_cache(
                            tail_kv_cache,
                            k[prefill_slice],
                            gate_score[prefill_slice],
                            tail_meta.slot_mapping[prefill_slice],
                            index_kpool,
                            head_dim,
                        )
        else:
            # standard: per-token fp8 quant + scatter (all tokens).
            assert scale_fmt is not None
            _bc("indexer k_quant_and_cache slots", slot_mapping,
                kv_cache.shape[0] * kv_cache.shape[1])
            ops.indexer_k_quant_and_cache(
                k,
                kv_cache,
                slot_mapping,
                quant_block_size,
                scale_fmt,
            )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Short-sequence full-attention fast path (mirrors sglang
        # IndexerKPool._full_topk_for_short_sequence). When every prefill
        # request's full context is <= topk_tokens, sparse selection would
        # pick ALL pools anyway (topk_pool = topk_tokens // index_kpool >=
        # num_pools, plus the always-selected tail == every token), so running
        # the MQA-logits is pointless and (in this port) triggers OOBs. Skip
        # it and attend to every token causally instead. The index-K cache was
        # already written above; this only fills the topk buffer. Real
        # sparsity only kicks in for contexts > topk_tokens.
        n_prefill_sf = num_tokens - num_decode_tokens
        # Host-side short-prefill predicate: max_prefill_seq_len is computed
        # in the metadata builder (exact for prefill rows) and equals
        # positions[prefill_slice].max() + 1, so this replaces a
        # positions.max().item() device sync per layer. -1 (unknown metadata)
        # falls back to the device-side check.
        if prefill_metadata.max_prefill_seq_len >= 0:
            short_prefill = (
                n_prefill_sf > 0
                and positions is not None
                and prefill_metadata.max_prefill_seq_len <= topk_tokens
            )
        else:
            short_prefill = (
                n_prefill_sf > 0
                and positions is not None
                and int(positions[num_decode_tokens:num_tokens].max().item()) + 1
                <= topk_tokens
            )
        if short_prefill:
            # short_prefill is only True when positions is not None (above),
            # but narrow explicitly for the indexer below.
            assert positions is not None
            _arange = torch.arange(
                topk_indices_buffer.shape[1],
                device=topk_indices_buffer.device,
                dtype=torch.int32,
            )
            _pos = positions[num_decode_tokens:num_tokens].to(torch.int32)
            _buf = topk_indices_buffer[num_decode_tokens:num_tokens]
            _buf[:] = _arange[None, :]
            _buf[_arange[None, :] > _pos[:, None]] = -1

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks if not short_prefill else ():
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            if not chunk.skip_kv_gather:
                if _BOUNDS_CHECK and _GATHER_LOGGED[0] < 6:
                    _GATHER_LOGGED[0] += 1
                    logger.info(
                        "GLM53 gather#%d: kv_cache %s stride %s | k_quant %s | block_table %s "
                        "max_id %d | cu_seq_lens[-1] %d total_seq_lens %d | q rows %d",
                        _GATHER_LOGGED[0], tuple(kv_cache.shape), tuple(kv_cache.stride()),
                        tuple(k_quant.shape), tuple(chunk.block_table.shape),
                        int(chunk.block_table.max().item()), int(chunk.cu_seq_lens[-1].item()),
                        int(chunk.total_seq_lens), int(chunk.token_end - chunk.token_start),
                    )
                if _BOUNDS_CHECK:
                    assert int(chunk.cu_seq_lens[-1].item()) <= k_quant.shape[0], (
                        f"GLM53 gather: cu_seq_lens[-1]={int(chunk.cu_seq_lens[-1].item())} > "
                        f"workspace rows {k_quant.shape[0]}")
                    assert int(chunk.block_table.max().item()) < kv_cache.shape[0], (
                        f"GLM53 gather: block id {int(chunk.block_table.max().item())} >= "
                        f"num_blocks {kv_cache.shape[0]}")
                if _SAFE_GATHER:
                    _gather_k_torch(kv_cache, k_quant, k_scale, chunk.block_table,
                                    chunk.cu_seq_lens, head_dim)
                else:
                    ops.cp_gather_indexer_k_quant_cache(
                        kv_cache,
                        k_quant,
                        k_scale,
                        chunk.block_table,
                        chunk.cu_seq_lens,
                    )

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
            # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
            if use_fp4_cache:
                q_slice_cast = q_slice.view(torch.int8)
                k_quant_cast = k_quant.view(torch.int8)
                k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_quant
                k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
            if is_deep_gemm_supported():
                logits = fp8_fp4_mqa_logits(
                    (q_slice_cast, q_scale_slice),
                    (k_quant_cast, k_scale_cast),
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    clean_logits=False,
                )
            else:
                # sm80 (no DeepGEMM): Triton MQA-logits fallback (Malav-P fork).
                from vllm.v1.attention.ops.mqa_logits_triton import (
                    fp8_mqa_logits_triton,
                )

                if _BOUNDS_CHECK:
                    _ke = int(chunk.cu_seqlen_ke.max().item()); _ks = int(chunk.cu_seqlen_ks.min().item())
                    assert 0 <= _ks and _ke <= k_quant_cast.shape[0], (
                        f"GLM53 logits: ks/ke range [{_ks},{_ke}] vs N={k_quant_cast.shape[0]}")
                    assert chunk.cu_seqlen_ks.shape[0] >= q_slice_cast.shape[0], (
                        f"GLM53 logits: ks rows {chunk.cu_seqlen_ks.shape[0]} < M={q_slice_cast.shape[0]}")
                if _BOUNDS_CHECK and k_quant_cast.shape[0] > 24000:
                    _w = weights[chunk.token_start : chunk.token_end]
                    logger.info(
                        "GLM53 logits args: M=%d N=%d | q %s %s stride %s ptr %#x | k %s %s stride %s "
                        "| scale %s %s stride %s numel %d | w %s %s stride %s | ks %s [%d,%d] rows %d "
                        "| ke %s [%d,%d] | alloc %.2f GiB",
                        q_slice_cast.shape[0], k_quant_cast.shape[0],
                        tuple(q_slice_cast.shape), q_slice_cast.dtype, tuple(q_slice_cast.stride()), q_slice_cast.data_ptr(),
                        tuple(k_quant_cast.shape), k_quant_cast.dtype, tuple(k_quant_cast.stride()),
                        tuple(k_scale_cast.shape), k_scale_cast.dtype, tuple(k_scale_cast.stride()), k_scale_cast.numel(),
                        tuple(_w.shape), _w.dtype, tuple(_w.stride()),
                        chunk.cu_seqlen_ks.dtype, int(chunk.cu_seqlen_ks.min().item()), int(chunk.cu_seqlen_ks.max().item()), chunk.cu_seqlen_ks.shape[0],
                        chunk.cu_seqlen_ke.dtype, int(chunk.cu_seqlen_ke.min().item()), int(chunk.cu_seqlen_ke.max().item()),
                        torch.cuda.memory_allocated() / 2**30,
                    )
                if _use_safe_logits(k_quant_cast.shape[0]):
                    logits = _mqa_logits_torch(
                        q_slice_cast, k_quant_cast, k_scale_cast,
                        weights[chunk.token_start : chunk.token_end],
                        chunk.cu_seqlen_ks, chunk.cu_seqlen_ke,
                    )
                else:
                    logits = fp8_mqa_logits_triton(
                        q_slice_cast,
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        chunk.cu_seqlen_ks,
                        chunk.cu_seqlen_ke,
                        clean_logits=False,
                    )
            num_rows = logits.shape[0]

            # kpool: logits are pool-granular (compress_ratio == index_kpool),
            # so topk selects pools. We pick topk_tokens // kpool pools then
            # expand each pool back to its kpool constituent tokens.
            select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
            if index_kpool > 1:
                pool_topk = torch.empty(
                    (num_rows, select_k), dtype=torch.int32, device=logits.device
                )
                topk_dst = pool_topk
            else:
                topk_dst = topk_indices_buffer[
                    chunk.token_start : chunk.token_end, :topk_tokens
                ]

            if current_platform.is_xpu():
                xpu_ops.top_k_per_row_prefill(  # type: ignore[attr-defined]
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )
            else:
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )

            if index_kpool > 1:
                pool_ids = pool_topk.to(torch.int64)
                if positions is not None:
                    # Fused expand-pools + append-tail into one Triton kernel
                    # (replaces ~25 elementwise ops). seq_len is token-granular
                    # (pos+1); the kernel derives pool_len internally.
                    q_seq = (
                        positions[chunk.token_start : chunk.token_end].to(torch.int32)
                        + 1
                    )
                    expanded = expand_pools_and_append_tail(
                        pool_ids, q_seq, index_kpool
                    )
                else:
                    valid = pool_ids >= 0
                    expanded = expand_pools_to_tokens(
                        pool_ids, valid, topk_tokens, index_kpool
                    )
                topk_indices_buffer[
                    chunk.token_start : chunk.token_end, : expanded.shape[-1]
                ] = expanded

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache_raw = kv_cache  # raw [num_blocks, block_size, head_dim+4] for writes
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)

        # kpool decode write (must precede the logits read). Append each decode
        # token's k/gate to its REQUEST's tail ring; when a pool fills
        # (pos % kpool == kpool-1) compress + write at the pool slot that
        # compress_ratio hands us via slot_mapping.
        #
        # Spec verify batches next_n (>1) tokens per request. The per-request
        # tail ring must accumulate a request's tokens IN POSITION ORDER, so we
        # group tokens by request ([num_requests, next_n, ...]) and run the
        # per-request kernel once per token-slot — sequential launches keep each
        # request's tokens ordered (token t stashes before token t+1 reads it
        # for pool completion). Mirrors sglang's _forward_cuda_target_verify
        # (per-request kpool write plan, seqlen_per_q = write_start + k + 1).
        # Plain decode (next_n == 1) collapses to a single launch.
        #
        # NOTE: positions must be TOKEN-granular (per-token position, not the
        # pool-granular decode_metadata.seq_lens which is divided by
        # compress_ratio). The kernel derives the pool phase and tail-ring index
        # from pos % kpool, so a pool-granular pos misaligns every pool; a
        # per-request pos under spec is also too short (B entries for B*next_n
        # tokens) and reads out of bounds.
        if (
            index_kpool > 1
            and gate_score is not None
            and compress_ape is not None
            and positions is not None
            and not skip_k_cache_insert
            and os.environ.get("VLLM_KPOOL_SKIP_DECODE_WRITE") != "1"
        ):
            num_requests = attn_metadata_narrowed.num_decodes
            # The indexer's flatten decode path rewrites decode_lens to all-1s
            # and reports requires_padding=False even for a variable MTP-verify
            # batch (e.g. one request verifies 3 tokens while the rest verify
            # 4). The logits read is fine with that, but the kpool WRITE must
            # group tokens by their original request. Uniformity and the scatter
            # lmax are precomputed on the host in build()
            # (decode_is_uniform / write_max_decode_len), so this branch needs
            # no runtime .item() -- a .item() under cudagraph capture forces a
            # host sync and invalidates the stream.
            per_req_lens = decode_metadata.per_req_decode_lens
            if per_req_lens is not None:
                use_uniform = (
                    decode_metadata.decode_is_uniform
                    and num_decode_tokens
                    == num_requests * decode_metadata.write_max_decode_len
                )
                group_lens = per_req_lens
                lmax = decode_metadata.write_max_decode_len
            else:
                # Legacy metadata without per-request lens: fall back to the
                # host-side requires_padding flag. Unreached now (per-request
                # lens is always populated for decode), kept defensive.
                use_uniform = not decode_metadata.requires_padding
                group_lens = decode_metadata.decode_lens
                lmax = int(decode_metadata.decode_lens.max().item())
            if not use_uniform:
                # Non-uniform decode_lens (mixed plain-decode + spec-verify, or
                # a variable MTP-verify batch): scatter actual tokens into a
                # padded [B, lmax] layout. int32 tensors can't go through
                # pack_seq_triton (float/uint8 only). The scatter indices are
                # shared by all five scatters below (and the tail slot one).
                scatter_idx = _build_decode_scatter_indices(
                    group_lens, num_requests, num_decode_tokens
                )
                dec_k = _scatter_decode_tokens_by_request(
                    k[:num_decode_tokens], 0, num_requests, lmax, scatter_idx
                )
                dec_gate = _scatter_decode_tokens_by_request(
                    gate_score[:num_decode_tokens],
                    0,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
                dec_slot = _scatter_decode_tokens_by_request(
                    slot_mapping[:num_decode_tokens],
                    -1,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
                dec_pos = _scatter_decode_tokens_by_request(
                    positions[:num_decode_tokens].to(torch.int32),
                    -1,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
            else:
                next_n = num_decode_tokens // num_requests
                shape2 = (num_requests, next_n)
                dec_k = k[:num_decode_tokens].view(*shape2, head_dim)
                dec_gate = gate_score[:num_decode_tokens].view(*shape2, head_dim)
                dec_slot = slot_mapping[:num_decode_tokens].view(shape2)
                dec_pos = positions[:num_decode_tokens].to(torch.int32).view(shape2)
            tail_meta = (
                attn_metadata.get(_resolve_layer_name(tail_prefix))
                if tail_prefix is not None
                else None
            )
            # Paged tail cache replaces the transient _DECODE_TAIL ring. Group
            # the tail group's token-granular slot_mapping per-request, mirroring
            # dec_slot / dec_pos, so the kernel gets each request's current-token
            # tail slot (block * kpool + pos % kpool).
            if tail_meta is not None:
                assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
            if (tail_meta is None or tail_kv_cache is None
                    or tail_meta.tail_own_blocks is None
                    or torch.cuda.is_current_stream_capturing()):
                dec_tail_slot = None
            else:
                # Recompute the tail slot mapping on-the-fly from Python
                # ints (immune to GPU memory corruption during CUDA graph
                # replay) and the positions tensor (correct via weak_ref).
                _kpool = tail_meta.tail_kpool
                _own = tail_meta.tail_own_blocks
                # Cache the tiny GPU tensor on the metadata so the CPU->GPU
                # transfer happens once per step, not once per kpool layer.
                _own_t = getattr(tail_meta, '_own_blocks_gpu', None)
                if _own_t is None or _own_t.shape[0] != num_requests:
                    _own_t = torch.tensor(
                        _own, dtype=torch.int64,
                        device=positions.device)[:num_requests]
                    tail_meta._own_blocks_gpu = _own_t
                _pos_i64 = positions[:num_decode_tokens].to(torch.int64)
                if not use_uniform:
                    _req_ids = torch.arange(
                        num_requests, device=positions.device)
                    # (image port) this build's decode metadata carries no
                    # query_start_loc; `group_lens` is the same per-request
                    # decode length used to build scatter_idx above.
                    _own_exp = _own_t[_req_ids].repeat_interleave(
                        group_lens[:num_requests].to(torch.int64))
                    _tail_flat = (
                        _own_exp[:num_decode_tokens] * _kpool
                        + _pos_i64 % _kpool)
                    dec_tail_slot = _scatter_decode_tokens_by_request(
                        _tail_flat, -1, num_requests, lmax, scatter_idx,
                    )
                else:
                    _own_col = _own_t.unsqueeze(1)  # [B, 1]
                    _pos_v = _pos_i64.view(shape2)
                    dec_tail_slot = _own_col * _kpool + _pos_v % _kpool
            # The compress kernel writes the raw fp8 cache (not the quant view);
            # pass the underlying kv_cache, not kv_cache_quant_view.
            if dec_tail_slot is not None:
                # Single batched launch over [num_requests, next_n] replaces the
                # per-token sequential loop. The kernel iterates each request's
                # tokens in position order internally, preserving the
                # pool-completion read-after-stash dependency that the loop
                # provided. Inputs are already grouped per request (uniform:
                # view; non-uniform: _scatter_decode_tokens_by_request padded to
                # [B, lmax]) — no per-token .contiguous() copies needed.
                _bc("kpool decode main slots", dec_slot, kv_cache.shape[0] * kv_cache.shape[1])
                _bc("kpool decode tail slots", dec_tail_slot, tail_kv_cache.shape[0] * index_kpool)
                kpool_decode_update_and_maybe_write_cache_batched(
                    kv_cache_raw,
                    tail_kv_cache,
                    dec_tail_slot,
                    dec_k,
                    dec_gate,
                    compress_ape,
                    dec_slot,
                    dec_pos,
                    index_kpool,
                    head_dim,
                    round_scale=(scale_fmt is not None),
                )
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
            padded_weights = pack_seq_triton(
                weights[:num_decode_tokens], decode_lens, pad_value=0
            ).reshape(-1, *weights.shape[1:])
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
            padded_weights = weights[:num_decode_tokens]
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        if is_deep_gemm_supported():
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                padded_weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        else:
            # sm80 (no DeepGEMM): Triton paged MQA-logits fallback (Malav-P fork).
            # Downstream topk reads only up to seq_lens, so size the buffer to
            # the active batch max rather than the configured model max.
            from vllm.v1.attention.ops.mqa_logits_triton import (
                fp8_paged_mqa_logits_triton,
            )

            active_max_model_len = attn_metadata_narrowed.max_seq_len
            # Triton fallback expects [NB, BS, 1, HD+4]; squeeze the H=1
            # heads dim that the 4D BNHC allocator inserts.
            kv_cache_4d = kv_cache
            if kv_cache_4d.ndim == 5:
                kv_cache_4d = kv_cache_4d.squeeze(1)
            logits = fp8_paged_mqa_logits_triton(
                padded_q_quant_cast,
                kv_cache_4d,
                padded_weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                max_model_len=active_max_model_len,
                clean_logits=False,
            )
        num_rows = logits.shape[0]
        # kpool: logits are pool-granular -> select topk_tokens//kpool pools,
        # then expand each pool back to its kpool tokens.
        select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
        if index_kpool > 1:
            pool_topk = torch.empty(
                (num_rows, select_k), dtype=torch.int32, device=logits.device
            )
            topk_dst = pool_topk
        else:
            topk_dst = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        # sm8x: the radix persistent_topk kernel returns WRONG indices when the
        # candidate count lands between k and 2k (found on DSv4 on these cards,
        # deepseek-v4-cmp170hx patch 0001). Mirror the SM90 gate the sibling
        # cooperative path carries; sm_80 then takes top_k_per_row_decode.
        if (
            current_platform.is_cuda()
            and select_k in (512, 1024, 2048)
            and current_platform.has_device_capability(90)
        ):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_dst,
                topk_workspace,
                select_k,
                attn_metadata_narrowed.max_seq_len,
            )
        else:
            if current_platform.is_xpu():
                xpu_ops.top_k_per_row_decode(  # type: ignore[attr-defined]
                    logits,
                    next_n,
                    seq_lens,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )
            else:
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    next_n,
                    seq_lens,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )

        # Resolve to token-level indices in the output buffer.
        if index_kpool > 1:
            pool_ids = pool_topk.to(torch.int64)
            n = pool_topk.shape[0]
            # NOTE: decode_metadata.seq_lens is POOL-granular (divided by
            # compress_ratio in the indexer metadata builder) because it feeds
            # the paged-MQA logits. The fused kernel needs TOKEN-granular seq_len,
            # so recover it from the decode tokens' positions (pos == seq_len-1).
            # Using the compressed seq_lens yields dec_seq=0 for seq_len<kpool
            # -> empty topk -> the sparse MLA attends to nothing -> decode
            # degradation. The row->token mapping must follow the PADDED
            # [B, next_n] layout on non-uniform batches (see
            # _decode_topk_seq_lens).
            if positions is not None:
                dec_seq = _decode_topk_seq_lens(
                    positions,
                    decode_lens,
                    num_decode_tokens,
                    batch_size,
                    next_n,
                    decode_metadata.requires_padding,
                )
            else:
                dec_seq = decode_metadata.seq_lens[:n]
                if dec_seq.ndim == 2:
                    dec_seq = dec_seq[:, -1]
                dec_seq = dec_seq.to(torch.int32)
            out = expand_pools_and_append_tail(pool_ids, dec_seq, index_kpool)
        else:
            out = topk_dst

        if decode_metadata.requires_padding:
            # Drop padded query rows introduced by the next_n padding above.
            out = unpack_seq_triton(
                out.reshape(batch_size, -1, out.shape[-1]), decode_lens
            )
        topk_indices_buffer[: out.shape[0], : out.shape[-1]] = out

    return topk_indices_buffer


def sparse_attn_indexer_kpool_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    gate_score: torch.Tensor | None = None,
    compress_ape: torch.Tensor | None = None,
    index_kpool: int = 1,
    positions: torch.Tensor | None = None,
    tail_kv_cache: torch.Tensor | None = None,
    tail_prefix: str | None = None,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer_kpool",
    op_func=sparse_attn_indexer_kpool,
    # The indexer writes the index-K cache in place (prefill k-cache insert +
    # kpool decode write), so kv_cache must be declared as mutated — otherwise
    # under full-graph compile dynamo assumes it is unchanged across the
    # indexer→MLA boundary and the MLA reads stale/misaligned KV. The paged tail
    # cache is likewise written in place (prefill tail scatter + decode stash).
    mutates_args=["topk_indices_buffer", "kv_cache", "tail_kv_cache"],
    fake_impl=sparse_attn_indexer_kpool_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer_kpool")
class SparseAttnIndexerKpool(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        tail_cache=None,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.tail_cache = tail_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        if current_platform.is_cuda() and not has_deep_gemm():
            if is_deep_gemm_supported():
                raise RuntimeError(
                    "Sparse Attention Indexer CUDA op requires DeepGEMM"
                    " to be installed.")
            import warnings
            warnings.warn(
                "DeepGEMM not found but not required on this GPU "
                "(SM80) — kpool will use fallback kernels.",
                stacklevel=2,
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(
                hidden_states,
                q_quant,
                k,
                weights,
                gate_score=gate_score,
                compress_ape=compress_ape,
                index_kpool=index_kpool,
                positions=positions,
            )
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer_kpool(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
            gate_score,
            compress_ape,
            index_kpool,
            positions,
            self.tail_cache.kv_cache if self.tail_cache is not None else None,
            self.tail_cache.prefix if self.tail_cache is not None else None,
        )

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path is only supported on AITER. "
            "Please enable aiter with VLLM_ROCM_USE_AITER=1"
        )
