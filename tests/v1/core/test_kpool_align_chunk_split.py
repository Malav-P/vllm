# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compressed-cache prefill chunk splitting (`_pool_aligned_split`).

A kpool indexer (GLM-5.3-Flash) compresses every `kpool` consecutive tokens
into one cache slot, gathering those tokens only from the current prefill
batch. A chunk boundary that is not a multiple of `kpool` therefore splits a
pool across two forward passes and the slot straddling it is never written
correctly. The scheduler must end every intermediate prefill chunk on a pool
boundary; the final chunk is exempt because the tail cache holds the trailing
partial pool for decode.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KpoolTailSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
)

from .utils import create_requests

pytestmark = pytest.mark.cpu_test

ATTN_BLOCK_SIZE = 64
KPOOL = 16


def _attn_spec(**kw) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=ATTN_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        **kw,
    )


def _tail_spec(kpool: int = KPOOL) -> KpoolTailSpec:
    # Mirrors Glm5NextTailCache.get_kv_cache_spec.
    return KpoolTailSpec(
        block_size=kpool,
        num_kv_heads=2,
        head_size=128,
        head_size_v=0,
        dtype=torch.bfloat16,
        sliding_window=kpool,
    )


def _config(*specs) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer.{i}"], spec) for i, spec in enumerate(specs)
        ],
    )


def test_alignment_is_one_without_kpool_tail():
    assert _config(_attn_spec()).prefill_chunk_alignment == 1


def test_alignment_ignores_checkpointed_compressed_cache():
    # DeepSeek-V4 compresses tokens_per_state tokens per slot but carries its
    # partial pool across chunks itself, so it must not constrain the split.
    dsv4 = MLAAttentionSpec(
        block_size=ATTN_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    assert _config(_attn_spec(), dsv4).prefill_chunk_alignment == 1


def test_alignment_is_kpool_with_tail_cache():
    indexer = MLAAttentionSpec(
        block_size=ATTN_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=KPOOL,
    )
    cfg = _config(_attn_spec(), indexer, _tail_spec())
    assert cfg.prefill_chunk_alignment == KPOOL


def test_alignment_is_lcm_of_distinct_pool_sizes():
    cfg = _config(_attn_spec(), _tail_spec(16), _tail_spec(24))
    assert cfg.prefill_chunk_alignment == 48


def _split(request, num_new_tokens, **kw) -> int:
    stub = SimpleNamespace(prefill_chunk_alignment=KPOOL)
    return Scheduler._pool_aligned_split(stub, request, num_new_tokens, **kw)


def _request(prompt_len: int, num_computed: int = 0):
    request = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)[0]
    request.num_computed_tokens = num_computed
    return request


@pytest.mark.parametrize(
    "start,num_new,expected",
    [
        # Budget-limited chunk ending mid-pool is cut back to the pool boundary.
        (0, 8187, 8176),
        # Already aligned: unchanged.
        (0, 8192, 8192),
        # Chunk starting mid-way (aligned start) is cut on the absolute
        # position, not the chunk length.
        (8176, 1000, 992),
        # Fewer tokens than one pool and not the last chunk: wait a step.
        (0, 10, 0),
        (8176, 15, 0),
    ],
)
def test_intermediate_chunk_aligned(start, num_new, expected):
    request = _request(10000, num_computed=start)
    assert _split(request, num_new) == expected


def test_last_prefill_chunk_is_not_cut():
    # 8176 + 1824 == prompt length: the tail cache captures the partial pool.
    request = _request(10000, num_computed=8176)
    assert _split(request, 1824) == 1824
    # Whole prompt in one chunk, unaligned length.
    request = _request(10000)
    assert _split(request, 10000) == 10000


def test_decode_step_is_not_cut():
    request = _request(10000, num_computed=10000)
    request.append_output_token_ids(1)
    assert _split(request, 1) == 1


def test_resumed_request_aligns_against_full_sequence():
    # A preempted request replays prompt + generated tokens; everything up to
    # the last token is prefill and must obey the alignment.
    request = _request(100)
    for tok in range(50):
        request.append_output_token_ids(tok)
    assert request.num_tokens == 150
    assert _split(request, 120) == 112
    # Reaching the last token (num_tokens - 1) ends the prefill: not cut.
    assert _split(request, 149) == 149


def test_prefix_cache_hits_count_toward_start():
    # 4000 local + 100 external computed tokens shift the chunk start; the cut
    # is on the absolute position (4100 + 500 = 4600 -> 4592).
    request = _request(10000)
    assert (
        _split(
            request,
            500,
            num_new_local_computed_tokens=4000,
            num_external_computed_tokens=100,
        )
        == 492
    )


def test_external_tokens_clamped_to_pool_boundary():
    # A KV connector may report an external hit that is not a multiple of the
    # pool size. If adopted as-is, the pool straddling the resume boundary can
    # never be written (its pre-boundary tokens' raw K is not in any batch),
    # so the slot scores ~0 forever and hides its tokens from the indexer.
    # The scheduler must round the external prefix down to the boundary.
    stub = SimpleNamespace(prefill_chunk_alignment=KPOOL)
    align = Scheduler._align_external_computed_tokens
    assert align(stub, 4100) == 4096  # 4100 % 16 = 4 -> drop the tail
    assert align(stub, 4096) == 4096  # already aligned: unchanged
    assert align(stub, 8) == 0  # smaller than one pool: adopt nothing
    assert align(stub, 0) == 0


def test_external_tokens_untouched_without_kpool():
    # alignment == 1 (no kpool tail cache): external hits pass through.
    stub = SimpleNamespace(prefill_chunk_alignment=1)
    align = Scheduler._align_external_computed_tokens
    assert align(stub, 4100) == 4100
    assert align(stub, 0) == 0


def test_clamped_external_keeps_chunk_start_aligned():
    # End-to-end through _pool_aligned_split: external 4100 clamps to 4096,
    # then the first chunk starts on a pool boundary and its END cut lands on
    # the next boundary (4096 + 500 = 4596 -> 4592).
    stub = SimpleNamespace(prefill_chunk_alignment=KPOOL)
    request = _request(10000)
    ext = Scheduler._align_external_computed_tokens(stub, 4100)
    assert Scheduler._pool_aligned_split(stub, request, 500, 0, ext) == 496
