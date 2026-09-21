# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical correctness of the SM80 kpool fp8 (e4m3fn) quantization path.

A100 (SM80) Triton cannot emit float8_e4m3fn stores, so ``kpool_compress``
synthesizes the bytes via fp8e4b15 (see ``_to_fp8_u8``). e4b15 and e4m3fn share
the 1-4-3 fp8 layout but differ in exponent bias by 8, so the store must scale
by 1/256 to land byte-exact e4m3fn values. These tests guard that identity: the
regression they catch is silent index-key corruption (values read back 256x off
and/or saturated), which only degrades quality past the ``index_topk`` context
length and is invisible to smoke tests.

Run on the target GPU (A100):
    .venv/bin/python -m pytest tests/kernels/attention/test_kpool_fp8_quant.py -v
"""

import pytest
import torch

from vllm.triton_utils import tl, triton

kpool = pytest.importorskip(
    "vllm.models.glm5next.nvidia.ops.kpool_compress",
    reason="glm5next kpool ops not available",
)


@triton.jit
def _cast_to_e4m3_bytes_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(x_ptr + i, mask=m, other=0.0)
    tl.store(out_ptr + i, kpool._to_fp8_u8(x), mask=m)


def _triton_encode(x_f32: torch.Tensor) -> torch.Tensor:
    """Encode fp32 -> e4m3fn bytes via the production _to_fp8_u8 helper."""
    x_f32 = x_f32.contiguous()
    n = x_f32.numel()
    out = torch.empty(n, dtype=torch.uint8, device=x_f32.device)
    block = 256
    _cast_to_e4m3_bytes_kernel[(triton.cdiv(n, block),)](
        x_f32, out, n, BLOCK=block
    )
    return out


def _e4m3_step(v: torch.Tensor) -> torch.Tensor:
    """Grid spacing of float8_e4m3fn at magnitude ``v`` (normals + subnormals)."""
    absv = v.abs()
    # Smallest normal is 2**-6; below that the grid is uniform at 2**-9.
    exp = torch.floor(torch.log2(absv.clamp_min(2.0 ** -6)))
    step = torch.exp2(exp - 3.0)
    return torch.maximum(step, torch.full_like(step, 2.0 ** -9))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_to_fp8_u8_within_one_ulp_of_e4m3fn():
    """_to_fp8_u8 must match a real e4m3fn cast to within one fp8 ULP.

    Reading the bytes back as e4m3fn must reproduce PyTorch's reference
    float8_e4m3fn conversion across the whole clamped [-448, 448] range
    (incl. subnormals, signed zero, the 448 boundary). The only permitted
    divergence is exact-midpoint rounding ties: Triton's fp8e4b15 cast rounds
    half away from zero while torch uses round-half-to-even, a <=1-ULP
    difference on a handful of values that is far below fp8's quantization
    step. Anything larger (e.g. the old e4b15 256x / saturation corruption)
    must fail this test.
    """
    torch.manual_seed(0)
    vals = torch.cat(
        [
            torch.linspace(-448.0, 448.0, 8193),
            torch.tensor(
                [0.0, -0.0, 448.0, -448.0, 447.5, 1.0, 256.0, 1e-3, -1e-3,
                 2.0 ** -9, -(2.0 ** -9), 2.0 ** -6, 0.015625],
            ),
            (torch.rand(16384) * 2.0 - 1.0) * 448.0,
        ]
    ).float().cuda()

    got = _triton_encode(vals).view(torch.float8_e4m3fn).float()
    ref = vals.to(torch.float8_e4m3fn).float()

    step = _e4m3_step(ref)
    diff = (got - ref).abs()
    # got must be identical to, or the immediate neighbour of, the reference
    # code (<=1 ULP) -- never further.
    too_far = diff > step * 1.001
    n_too_far = int(too_far.sum().item())
    if n_too_far:
        idx = too_far.nonzero()[:8].flatten().tolist()
        detail = [(round(vals[i].item(), 5), got[i].item(), ref[i].item(),
                   round(step[i].item(), 6)) for i in idx]
        pytest.fail(
            f"{n_too_far}/{vals.numel()} values differ from torch e4m3fn by "
            f">1 ULP. First (input, got, ref, step): {detail}"
        )

    # Ties are rare; a large mismatch count would signal a systematic error.
    n_mismatch = int((got != ref).sum().item())
    assert n_mismatch < vals.numel() // 100, (
        f"{n_mismatch}/{vals.numel()} tie mismatches -- expected only "
        "exact-midpoint rounding ties"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_to_fp8_u8_not_saturated():
    """Guard against the e4b15 regression: values must not collapse.

    The old (buggy) cast quantized to [-448, 448] then bitcast through e4b15
    (max ~1.9), saturating every significant component to one code. A correct
    encoder spreads ~O(100) values across many distinct fp8 codes.
    """
    torch.manual_seed(0)
    vals = ((torch.rand(4096) * 2.0 - 1.0) * 400.0).float().cuda()
    got = _triton_encode(vals)
    # e4m3fn should use a broad spread of codes for wide-range inputs; the
    # buggy path pinned nearly everything to the saturation byte.
    n_distinct = int(torch.unique(got).numel())
    assert n_distinct > 64, (
        f"only {n_distinct} distinct fp8 codes — values are collapsing "
        "(e4b15 saturation regression?)"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fwht128_quant_roundtrip_preserves_norm():
    """End-to-end q-path: dequantized fp8 must reconstruct the input vector.

    The Hadamard-128 rotation is orthonormal, so ||rotated|| == ||q||. Reading
    the stored bytes back as e4m3fn and applying the per-row scale must recover
    that norm within fp8 quantization error. The e4b15 bug threw this off by
    ~256x / saturation, so a tight norm-preservation bound is a decisive guard
    that does not depend on the exact FWHT butterfly ordering.
    """
    torch.manual_seed(0)
    rows = 1024
    q = torch.randn(rows, 128, dtype=torch.bfloat16, device="cuda")

    q_fp8, scale = kpool.fwht128_quant_fp8(q)
    assert q_fp8.dtype == torch.float8_e4m3fn
    assert scale.shape == (rows, 1)

    # Dequantize exactly as the downstream logits kernels do.
    deq = q_fp8.float() * scale

    q_norm = q.float().norm(dim=1)
    deq_norm = deq.norm(dim=1)
    rel = ((deq_norm - q_norm).abs() / q_norm.clamp_min(1e-6))

    # e4m3fn ~3-bit mantissa => a few % per-element; norm error stays small.
    assert rel.median().item() < 0.02, f"median rel norm err {rel.median():.4f}"
    assert rel.max().item() < 0.10, f"max rel norm err {rel.max():.4f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fwht128_quant_empty():
    """Zero-row input returns correctly-typed empty tensors (no launch)."""
    q = torch.empty(0, 128, dtype=torch.bfloat16, device="cuda")
    q_fp8, scale = kpool.fwht128_quant_fp8(q)
    assert q_fp8.dtype == torch.float8_e4m3fn
    assert q_fp8.shape == (0, 128)
    assert scale.shape == (0, 1)
