# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 (A100) paged attention with Inkling2's query-dependent relative bias.

The sheared tml_fa4 kernel is Blackwell-only and the cute FA4 fallback needs
SM90+ (score_mod gate, paged KV asserts), so pre-SM90 devices run this Triton
implementation instead. It adapts the ROCm port
(``vllm/models/inkling/amd/ops/fa4_rel_attention.py``) — written for the same
reason: no score-mod hook — with two contract extensions for Inkling2: the
output is written in fp32 and the natural-log LSE ``(H, T)`` is emitted for
the gate/sink epilogue.

The relative bias arrives UNSHEARED: ``(T, Hq, rel_extent)`` with
``rel[t, h, d]`` the bias for the key at distance ``d`` behind query ``t``.
The caller unshears ``attention_prologue``'s table via
``_unshear_rel_logits`` (see ``ops/attention.py``).
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

_M_LN2 = tl.constexpr(0.6931471805599453)


@triton.heuristics(
    {
        "BLOCK_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_QH": lambda args: (
            args["BLOCK_Q"] * triton.next_power_of_2(args["gqa_group_size"])
        ),
    }
)
@triton.jit(do_not_specialize_on_alignment=["cache_seqlens", "cu_seqlens_q"])
def _sm80_rel_attention_kernel(
    q_ptr,  # [total_q, Hq, D]
    k_ptr,  # [blocks, page, Hkv, D]
    v_ptr,  # [blocks, page, Hkv, D]
    rel_ptr,  # [total_q, Hq, rel_extent]
    out_ptr,  # [total_q, Hq, D] (fp32)
    lse_ptr,  # [Hq, total_q] (fp32, natural log)
    block_table_ptr,  # [batch, max_pages]
    cache_seqlens,
    cu_seqlens_q,
    gqa_group_size,
    head_dim,
    softmax_scale,
    num_splits,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_b,
    stride_k_p,
    stride_k_h,
    stride_k_d,
    stride_v_b,
    stride_v_p,
    stride_v_h,
    stride_v_d,
    stride_r_t,
    stride_r_h,
    stride_r_e,
    stride_o_s,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_lse_s,
    stride_lse_h,
    stride_lse_t,
    stride_bt_b,
    page_size: tl.constexpr,
    rel_extent: tl.constexpr,
    window_left: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_QH: tl.constexpr,
):
    # A program owns BLOCK_Q query positions and every query head sharing one
    # KV head; rows are flattened to (BLOCK_Q * BLOCK_H, BLOCK_D) for the QK
    # and PV dots. Adapted from the ROCm kernel of the same shape.
    pid = tl.program_id(0)
    pid_q = pid // num_splits
    pid_s = pid % num_splits
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)

    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block = pid_q * BLOCK_Q
    if q_block >= q_len:
        return

    kv_len = tl.load(cache_seqlens + pid_b)
    prefix_len = kv_len - q_len
    q_head_start = pid_kh * gqa_group_size
    bt_row = block_table_ptr + pid_b * stride_bt_b

    # Keys this q-block can attend: causally bounded by its last query,
    # lower-bounded by the sliding window at its first query. Split-KV then
    # shards the surviving range; empty shards still emit their -inf LSE
    # partial for the combine pass in the launcher.
    q_abs_first = prefix_len + q_block
    q_abs_last = prefix_len + tl.minimum(q_block + BLOCK_Q, q_len) - 1
    k_begin = 0
    if window_left >= 0:
        k_begin = tl.maximum(0, q_abs_first - window_left)
    k_end = tl.minimum(kv_len, q_abs_last + 1)
    tiles = (k_end - k_begin + BLOCK_K - 1) // BLOCK_K
    tiles_per_split = (tiles + num_splits - 1) // num_splits
    split_begin = k_begin + pid_s * tiles_per_split * BLOCK_K
    split_end = tl.minimum(k_end, split_begin + tiles_per_split * BLOCK_K)

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_start * stride_q_t + q_head_start * stride_q_h,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_q_t, stride_q_h, stride_q_d),
        offsets=(q_block, 0, 0),
        block_shape=(BLOCK_Q, BLOCK_H, BLOCK_D),
        order=(2, 1, 0),
    )
    q = tl.load(q_block_ptr, boundary_check=(0, 1, 2), padding_option="zero")
    q = tl.reshape(q, (BLOCK_QH, BLOCK_D))

    q_rows = q_block + tl.arange(0, BLOCK_Q)
    q_abs = prefix_len + q_rows
    q_valid = q_rows < q_len
    off_d = tl.arange(0, BLOCK_D)
    d_valid = off_d < head_dim
    off_h = tl.arange(0, BLOCK_H)

    m_i = tl.full((BLOCK_QH,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_QH,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_QH, BLOCK_D), dtype=tl.float32)
    log2e: tl.constexpr = 1.4426950408889634

    for k_start in range(split_begin, split_end, BLOCK_K):
        k_pos = k_start + tl.arange(0, BLOCK_K)
        k_valid = k_pos < kv_len
        # Masked lanes still need an in-range page-table load.
        safe_pos = tl.minimum(k_pos, tl.maximum(kv_len - 1, 0))
        page = tl.load(bt_row + safe_pos // page_size).to(tl.int64)
        page_off = safe_pos % page_size

        k = tl.load(
            k_ptr
            + page[None, :] * stride_k_b
            + page_off[None, :] * stride_k_p
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d,
            mask=d_valid[:, None] & k_valid[None, :],
            other=0.0,
        )
        # (qk + bias) * scale in the exp2 domain: exp(qk * scale + bias).
        scores = tl.dot(q, k) * (softmax_scale * log2e)

        dist = q_abs[:, None] - k_pos[None, :]
        score_valid = q_valid[:, None] & k_valid[None, :] & (dist >= 0)
        if window_left >= 0:
            score_valid &= dist <= window_left

        # rel_logits is query- and head-dependent. Expand [Q, K] distance
        # indices across the GQA heads, then flatten to match QK's row order.
        # The bias only reaches rel_extent back; a tile whose every (q, k)
        # pair is older than that contributes zero bias, so skip its gather.
        if q_abs_first - (k_start + BLOCK_K - 1) < rel_extent:
            rel_dist = dist[:, None, :]
            rel_valid = (
                q_valid[:, None, None]
                & (off_h[None, :, None] < gqa_group_size)
                & score_valid[:, None, :]
                & (rel_dist < rel_extent)
            )
            safe_dist = tl.maximum(0, tl.minimum(rel_dist, rel_extent - 1))
            bias = tl.load(
                rel_ptr
                + (q_start + q_rows[:, None, None]) * stride_r_t
                + (q_head_start + off_h[None, :, None]) * stride_r_h
                + safe_dist * stride_r_e,
                mask=rel_valid,
                other=0.0,
            )
            bias = tl.reshape(bias, (BLOCK_QH, BLOCK_K))
            scores += bias.to(tl.float32) * log2e
        score_valid = tl.reshape(
            score_valid[:, None, :] & (off_h[None, :, None] < gqa_group_size),
            (BLOCK_QH, BLOCK_K),
        )
        scores = tl.where(score_valid, scores, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        has_scores = tl.sum(score_valid.to(tl.int32), axis=1) > 0
        # A sliding-window query can have entire early KV tiles masked. Avoid
        # the -inf - -inf NaN in online softmax until its first live tile.
        alpha = tl.where(has_scores, tl.exp2(m_i - m_ij), 1.0)
        p = tl.where(
            score_valid,
            tl.exp2(scores - m_ij[:, None]),
            0.0,
        )
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc *= alpha[:, None]

        v = tl.load(
            v_ptr
            + page[:, None] * stride_v_b
            + page_off[:, None] * stride_v_p
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d,
            mask=k_valid[:, None] & d_valid[None, :],
            other=0.0,
        )
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    # An empty shard (or a fully window-masked one) has l_i == 0; keep its
    # partial at zero so the combine pass discards it via its -inf LSE.
    acc = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
    acc = tl.reshape(acc, (BLOCK_Q, BLOCK_H, BLOCK_D))
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr
        + pid_s * stride_o_s
        + q_start * stride_o_t
        + q_head_start * stride_o_h,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_o_t, stride_o_h, stride_o_d),
        offsets=(q_block, 0, 0),
        block_shape=(BLOCK_Q, BLOCK_H, BLOCK_D),
        order=(2, 1, 0),
    )
    tl.store(
        out_block_ptr,
        acc.to(out_ptr.dtype.element_ty),
        boundary_check=(0, 1, 2),
    )

    # Natural-log LSE for the gate/sink epilogue: the online-softmax state
    # is in the exp2 domain, so lse = (m_i + log2(l_i)) * ln2.
    lse = tl.where(
        l_i > 0,
        (m_i + tl.log2(tl.maximum(l_i, 1e-30))) * _M_LN2,
        float("-inf"),
    )
    lse = tl.reshape(lse, (BLOCK_Q, BLOCK_H))
    lse_ptrs = (
        lse_ptr
        + pid_s * stride_lse_s
        + (q_head_start + off_h[None, :]) * stride_lse_h
        + (q_start + q_rows[:, None]) * stride_lse_t
    )
    tl.store(
        lse_ptrs,
        lse,
        mask=q_valid[:, None] & (off_h[None, :] < gqa_group_size),
    )


@torch.no_grad()
def sm80_rel_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    window_size: tuple[int, int],
    rel_extent: int,
    rel_logits: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    num_splits: int | None = None,
) -> None:
    """Paged varlen relative attention writing fp32 ``out`` and the
    ``(num_heads, total_q)`` natural-log ``lse``.

    Bias semantics match the sheared kernel: ``exp(qk * scale + bias)`` with
    ``bias = rel_logits[t, h, q_abs - k_pos]`` (no bias past ``rel_extent``).

    The KV scan is clamped to the keys the q-block can actually attend
    (causal upper bound, sliding-window lower bound), tiles older than
    ``rel_extent`` skip the bias gather, and small launches (decode) shard
    the scan across programs, merging the per-split partials by LSE weight.
    """
    if window_size[1] not in (0, -1):
        raise NotImplementedError("Inkling2 SM80 attention requires causal masking")
    if q.ndim != 3 or key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("expected q [T,H,D] and paged K/V [B,P,Hkv,D]")
    if rel_logits.shape != (q.shape[0], q.shape[1], rel_extent):
        raise ValueError(
            f"relative logits have shape {tuple(rel_logits.shape)}, expected "
            f"{(q.shape[0], q.shape[1], rel_extent)}"
        )
    num_kv_heads = key_cache.shape[2]
    if q.shape[1] % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if out.dtype != torch.float32 or out.shape != q.shape:
        raise ValueError("out must be fp32 with q's shape")
    if lse.shape != (q.shape[1], q.shape[0]) or lse.dtype != torch.float32:
        raise ValueError("lse must be fp32 (num_heads, total_q)")

    gqa_group_size = q.shape[1] // num_kv_heads
    batch = cu_seqlens_q.shape[0] - 1
    block_h = triton.next_power_of_2(gqa_group_size)
    # tl.dot needs >= 16 rows; BLOCK_QH = BLOCK_Q * BLOCK_H.
    block_q = max(16 // block_h, 1)
    grid_q = triton.cdiv(max_seqlen_q, block_q)
    if num_splits is None:
        # A decode step launches few programs (grid_q * kv_heads * batch can
        # be 1), leaving most of the GPU dark over a long KV scan; shard the
        # scan so the launch has ~512 CTAs. Prefill already has enough
        # q-block programs and stays unsplit.
        base_ctas = grid_q * num_kv_heads * batch
        num_splits = max(1, min(64, (512 + base_ctas - 1) // base_ctas))
    num_splits = max(1, int(num_splits))
    total_q, num_heads = q.shape[0], q.shape[1]
    if num_splits > 1:
        part_out = torch.empty(
            (num_splits, total_q, num_heads, q.shape[2]),
            dtype=torch.float32,
            device=q.device,
        )
        part_lse = torch.empty(
            (num_splits, num_heads, total_q),
            dtype=torch.float32,
            device=q.device,
        )
        out_arg, lse_arg = part_out, part_lse
        stride_o_s, stride_o_t, stride_o_h, stride_o_d = part_out.stride()
        stride_lse_s, stride_lse_h, stride_lse_t = part_lse.stride()
    else:
        out_arg, lse_arg = out, lse
        stride_o_s = 0
        stride_o_t, stride_o_h, stride_o_d = out.stride()
        stride_lse_s = 0
        stride_lse_h, stride_lse_t = lse.stride()
    grid = (grid_q * num_splits, num_kv_heads, batch)
    _sm80_rel_attention_kernel[grid](
        q,
        key_cache,
        value_cache,
        rel_logits,
        out_arg,
        lse_arg,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        gqa_group_size,
        q.shape[2],
        softmax_scale,
        num_splits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        rel_logits.stride(0),
        rel_logits.stride(1),
        rel_logits.stride(2),
        stride_o_s,
        stride_o_t,
        stride_o_h,
        stride_o_d,
        stride_lse_s,
        stride_lse_h,
        stride_lse_t,
        block_table.stride(0),
        page_size=key_cache.shape[1],
        rel_extent=rel_extent,
        window_left=window_size[0],
        BLOCK_Q=block_q,
        BLOCK_K=64,
        num_warps=4,
        num_stages=2,
    )
    if num_splits > 1:
        merge_split_partials(part_out, part_lse, out, lse)


def merge_split_partials_torch(
    part_out: torch.Tensor,
    part_lse: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    """Reference LSE-weighted merge of per-split normalized outputs.

    ``part_out`` is ``(S, T, H, D)``, ``part_lse`` ``(S, H, T)``; empty
    shards carry -inf LSE and zero weight.
    """
    lse_max = part_lse.amax(dim=0)
    safe_max = torch.where(lse_max.isfinite(), lse_max, torch.zeros_like(lse_max))
    w = torch.exp(part_lse - safe_max.unsqueeze(0))
    denom = w.sum(dim=0)
    weighted = (part_out * w.permute(0, 2, 1).unsqueeze(-1)).sum(dim=0)
    out.copy_(weighted / torch.clamp(denom, min=1e-30).transpose(0, 1).unsqueeze(-1))
    lse.copy_(
        torch.where(
            denom > 0,
            safe_max + torch.log(torch.clamp(denom, min=1e-30)),
            torch.full_like(denom, float("-inf")),
        )
    )


@triton.jit
def _merge_split_partials_kernel(
    part_out_ptr,  # [S, T, H, D] fp32
    part_lse_ptr,  # [S, H, T] fp32
    out_ptr,  # [T, H, D] fp32
    lse_ptr,  # [H, T] fp32
    num_splits,
    num_heads,
    head_dim,
    stride_po_s,
    stride_po_t,
    stride_po_h,
    stride_po_d,
    stride_pl_s,
    stride_pl_h,
    stride_pl_t,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_l_h,
    stride_l_t,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid // num_heads
    h = pid % num_heads
    off_s = tl.arange(0, BLOCK_S)
    s_valid = off_s < num_splits
    part_lse = tl.load(
        part_lse_ptr + off_s * stride_pl_s + h * stride_pl_h + t * stride_pl_t,
        mask=s_valid,
        other=float("-inf"),
    )
    lse_max = tl.max(part_lse, axis=0)
    safe_max = tl.where(lse_max > float("-inf"), lse_max, 0.0)
    w = tl.where(s_valid, tl.exp(part_lse - safe_max), 0.0)
    denom = tl.sum(w, axis=0)
    off_d = tl.arange(0, BLOCK_D)
    d_valid = off_d < head_dim
    part = tl.load(
        part_out_ptr
        + off_s[:, None] * stride_po_s
        + t * stride_po_t
        + h * stride_po_h
        + off_d[None, :] * stride_po_d,
        mask=s_valid[:, None] & d_valid[None, :],
        other=0.0,
    )
    # Zero-weight shards contribute nothing even if their partial is
    # unwritten memory (rows past a request's q_len are never stored).
    acc = tl.sum(tl.where(w[:, None] > 0, part * w[:, None], 0.0), axis=0)
    tl.store(
        out_ptr + t * stride_o_t + h * stride_o_h + off_d * stride_o_d,
        acc / tl.maximum(denom, 1e-30),
        mask=d_valid,
    )
    tl.store(
        lse_ptr + h * stride_l_h + t * stride_l_t,
        tl.where(denom > 0, safe_max + tl.log(tl.maximum(denom, 1e-30)), -float("inf")),
    )


def merge_split_partials(
    part_out: torch.Tensor,
    part_lse: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    """One-launch LSE merge; same semantics as ``merge_split_partials_torch``."""
    num_splits, total_q, num_heads, head_dim = part_out.shape
    _merge_split_partials_kernel[(total_q * num_heads,)](
        part_out,
        part_lse,
        out,
        lse,
        num_splits,
        num_heads,
        head_dim,
        *part_out.stride(),
        *part_lse.stride(),
        *out.stride(),
        *lse.stride(),
        BLOCK_S=triton.next_power_of_2(num_splits),
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
