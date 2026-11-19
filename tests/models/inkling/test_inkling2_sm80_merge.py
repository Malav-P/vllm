# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-KV merge of Inkling2's SM80 relative attention.

The decode path shards the KV scan and merges the per-split partials by
LSE weight; the Triton merge must match the torch reference, ignore
zero-weight shards (even unwritten/NaN ones), and make a split launch agree
with an unsplit one.
"""

import pytest
import torch

from vllm.models.inkling2.ops.sm80_attention import (
    merge_split_partials,
    merge_split_partials_torch,
    sm80_rel_attention,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


@pytest.mark.parametrize("num_splits", [2, 7, 64])
@pytest.mark.parametrize("total_q", [1, 5])
def test_merge_matches_torch_reference(num_splits: int, total_q: int) -> None:
    torch.manual_seed(0)
    heads, dim = 8, 128
    part_out = torch.randn(num_splits, total_q, heads, dim, device="cuda")
    part_lse = torch.randn(num_splits, heads, total_q, device="cuda") * 4
    part_lse[0, :, 0] = float("-inf")  # an empty shard
    part_lse[:, 1, 0] = float("-inf")  # a fully empty (t, h) row
    part_out[:, 0, 1] = 0.0
    ref_out = torch.empty(total_q, heads, dim, device="cuda")
    ref_lse = torch.empty(heads, total_q, device="cuda")
    merge_split_partials_torch(part_out, part_lse, ref_out, ref_lse)
    out = torch.empty_like(ref_out)
    lse = torch.empty_like(ref_lse)
    merge_split_partials(part_out, part_lse, out, lse)
    torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(lse, ref_lse, rtol=1e-5, atol=1e-6)
    assert torch.isneginf(lse[1, 0])


def test_merge_ignores_unwritten_zero_weight_shards() -> None:
    torch.manual_seed(0)
    part_out = torch.randn(4, 3, 8, 128, device="cuda")
    part_lse = torch.randn(4, 8, 3, device="cuda")
    part_lse[2] = float("-inf")
    clean = part_out.clone()
    clean[2] = 0.0
    part_out[2] = float("nan")
    ref_out = torch.empty(3, 8, 128, device="cuda")
    ref_lse = torch.empty(8, 3, device="cuda")
    merge_split_partials_torch(clean, part_lse, ref_out, ref_lse)
    out = torch.empty_like(ref_out)
    lse = torch.empty_like(ref_lse)
    merge_split_partials(part_out, part_lse, out, lse)
    torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(lse, ref_lse, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("window", [-1, 512])
def test_split_launch_matches_unsplit(window: int) -> None:
    torch.manual_seed(0)
    heads, kv_heads, dim, page, rel_extent = 8, 1, 128, 64, 256
    seqlens = [3000, 700]  # decode: one query token per request
    batch = len(seqlens)
    pages = [(s + page - 1) // page for s in seqlens]
    num_blocks = sum(pages)
    key_cache = torch.randn(
        num_blocks, page, kv_heads, dim, device="cuda", dtype=torch.bfloat16
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.zeros(batch, max(pages), dtype=torch.int32, device="cuda")
    perm = torch.randperm(num_blocks, device="cuda").to(torch.int32)
    start = 0
    for b, n in enumerate(pages):
        block_table[b, :n] = perm[start : start + n]
        start += n
    cache_seqlens = torch.tensor(seqlens, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    rel = torch.randn(batch, heads, rel_extent, device="cuda")

    def run(num_splits: int) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty(batch, heads, dim, device="cuda")
        lse = torch.empty(heads, batch, device="cuda")
        sm80_rel_attention(
            q,
            key_cache,
            value_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            softmax_scale=dim**-0.5,
            window_size=(window, 0),
            rel_extent=rel_extent,
            rel_logits=rel,
            out=out,
            lse=lse,
            num_splits=num_splits,
        )
        return out, lse

    ref_out, ref_lse = run(1)
    out, lse = run(8)
    torch.testing.assert_close(out, ref_out, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(lse, ref_lse, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("window", [-1, 200])
def test_sheared_bias_matches_unshear_path(window: int) -> None:
    """Reading the sheared table in-kernel must equal unshearing it first."""
    from vllm.models.inkling2.ops.attention import _unshear_rel_logits

    torch.manual_seed(0)
    heads, kv_heads, dim, page, rel_extent = 8, 1, 128, 64, 256
    kv_lens, q_lens = [700, 300], [300, 1]  # prefill over a prefix + decode
    batch = len(kv_lens)
    pages = [(s + page - 1) // page for s in kv_lens]
    num_blocks = sum(pages)
    key_cache = torch.randn(
        num_blocks, page, kv_heads, dim, device="cuda", dtype=torch.bfloat16
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.zeros(batch, max(pages), dtype=torch.int32, device="cuda")
    perm = torch.randperm(num_blocks, device="cuda").to(torch.int32)
    start = 0
    for b, n in enumerate(pages):
        block_table[b, :n] = perm[start : start + n]
        start += n
    cache_seqlens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    cu = [0]
    for n in q_lens:
        cu.append(cu[-1] + n)
    cu_seqlens_q = torch.tensor(cu, dtype=torch.int32, device="cuda")
    total_q = cu[-1]
    q = torch.randn(total_q, heads, dim, device="cuda", dtype=torch.bfloat16)
    sheared = torch.randn(total_q + 128, heads, rel_extent + 256, device="cuda")
    sheared[torch.rand_like(sheared) < 0.1] = float("nan")  # padding

    def run(rel: torch.Tensor, is_sheared: bool) -> tuple[torch.Tensor, ...]:
        out = torch.empty(total_q, heads, dim, device="cuda")
        lse = torch.empty(heads, total_q, device="cuda")
        sm80_rel_attention(
            q,
            key_cache,
            value_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max(q_lens),
            softmax_scale=dim**-0.5,
            window_size=(window, 0),
            rel_extent=rel_extent,
            rel_logits=rel,
            out=out,
            lse=lse,
            sheared=is_sheared,
        )
        return out, lse

    rel = _unshear_rel_logits(sheared, total_q, cache_seqlens, cu_seqlens_q, rel_extent)
    ref_out, ref_lse = run(rel, False)
    out, lse = run(sheared, True)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(lse, ref_lse, rtol=1e-5, atol=1e-5)


def test_merge_compiles_once_across_batch_shapes() -> None:
    """Shape changes mid-serve must not trigger new JIT compiles."""
    from vllm.models.inkling2.ops.sm80_attention import (
        _merge_split_partials_kernel,
    )

    for num_splits, total_q in [(2, 1), (7, 5), (64, 16), (33, 3)]:
        part_out = torch.randn(num_splits, total_q, 8, 128, device="cuda")
        part_lse = torch.randn(num_splits, 8, total_q, device="cuda")
        out = torch.empty(total_q, 8, 128, device="cuda")
        lse = torch.empty(8, total_q, device="cuda")
        merge_split_partials(part_out, part_lse, out, lse)
    torch.cuda.synchronize()
    caches = _merge_split_partials_kernel.device_caches
    variants = sum(len(c[0]) for c in caches.values())
    assert variants == 1, f"{variants} compiled variants"
