"""L0 oracle tests: the reference implementations agree with each other, and the
two numerical levers (schedule/tile size and accumulator width) behave as the
floating-point model predicts.
"""

from __future__ import annotations

import torch

from vllab.numerics import compare, reduction_atol
from vllab.reference.attention import online_softmax_attention, softmax_attention
from vllab.reference.kvcache import incremental_decode
from vllab.reference.matmul import reference_matmul, tiled_matmul


def _seed(s: int = 0) -> torch.Generator:
    g = torch.Generator().manual_seed(s)
    return g


# --------------------------------------------------------------------------- #
# matmul
# --------------------------------------------------------------------------- #
def test_tiled_matmul_matches_oracle_fp32() -> None:
    """Goal: a tiled matmul is the *same computation* as the oracle, just summed in
    chunks, so for any reduction tiling it must land within the precision-justified
    tolerance.

    Expectation: sweeping ``block_k`` over sizes that both divide and exceed the
    contraction dimension ``K`` (16, 32, 64, 256=full) changes only the reduction
    schedule, not the math. Asserts every tiling stays ``within`` the fp32
    ``reduction_atol`` for length ``K`` — i.e. the tile size never pushes the result
    beyond the rounding budget of a length-``K`` sum.
    """
    g = _seed()
    m, k, n = 64, 256, 48
    a = torch.randn(m, k, generator=g)
    b = torch.randn(k, n, generator=g)
    oracle = reference_matmul(a, b)
    atol = reduction_atol(torch.float32, k, scale=float(oracle.abs().mean()) + 1.0)
    for block_k in (16, 32, 64, 256):
        rep = compare(
            tiled_matmul(a, b, block_k=block_k, accum_dtype=torch.float32), oracle, atol=atol
        )
        assert rep.within, f"block_k={block_k}: {rep}"


def test_tile_size_changes_low_bits_but_stays_in_tolerance() -> None:
    """Goal: pin down the central premise of the lab — floating-point summation is
    *not* associative, so reduction order matters, but only in the low bits.

    Expectation: two tile sizes (16 vs 128) reorder the same additions. Asserts the
    two results are NOT bitwise identical (order genuinely changes the low bits) yet
    their gap is still ``within`` the fp32 length-512 ``reduction_atol``. Together
    these say: schedule dependence is real but bounded — you cannot demand bitwise
    equality across schedules, only tolerance-level agreement.
    """
    # Same math, different reduction order -> results differ, but only in the low
    # bits (schedule dependence), so they are NOT bitwise identical.
    g = _seed(1)
    a = torch.randn(32, 512, generator=g)
    b = torch.randn(512, 32, generator=g)
    c16 = tiled_matmul(a, b, block_k=16, accum_dtype=torch.float32)
    c128 = tiled_matmul(a, b, block_k=128, accum_dtype=torch.float32)
    assert not torch.equal(c16, c128), "different tile sizes should reorder the sum"
    rep = compare(c16, c128.to(torch.float64), atol=reduction_atol(torch.float32, 512, scale=2.0))
    assert rep.within, f"cross-schedule gap too large: {rep}"


def test_accumulator_width_dominates_error() -> None:
    """Goal: show that *accumulator precision*, not tile size, is the dominant error
    lever — the reason production kernels accumulate in fp32 even for fp16 inputs.

    Expectation: run the same long (``K=1024``) reduction twice against the fp64
    oracle, once accumulating in fp32 and once in fp16, and measure each max abs
    error (``atol=inf`` so nothing gates — we only read ``max_abs``). Asserts the
    fp16-accumulator error is more than 10x the fp32 one, i.e. narrowing the
    accumulator degrades accuracy by an order of magnitude, dwarfing the low-bit
    schedule effects of the previous tests.
    """
    # Narrow accumulator (fp16) must be materially worse than a wide one (fp32).
    g = _seed(2)
    k = 1024
    a = torch.randn(16, k, generator=g)
    b = torch.randn(k, 16, generator=g)
    oracle = reference_matmul(a, b)
    err_fp32 = compare(
        tiled_matmul(a, b, accum_dtype=torch.float32), oracle, atol=float("inf")
    ).max_abs
    err_fp16 = compare(
        tiled_matmul(a, b, accum_dtype=torch.float16), oracle, atol=float("inf")
    ).max_abs
    assert err_fp16 > 10 * err_fp32, f"fp16={err_fp16:.3e} not >> fp32={err_fp32:.3e}"


# --------------------------------------------------------------------------- #
# attention
# --------------------------------------------------------------------------- #
def test_online_softmax_matches_oracle() -> None:
    """Goal: the streaming/online-softmax attention (the FlashAttention-style
    formulation that never materialises the full score matrix) must be numerically
    equivalent to the plain, all-at-once softmax attention oracle.

    Expectation: for both non-causal and causal masking, and for every KV
    ``block_size`` (4, 8, 16, and 24=full sequence), the online result must match the
    oracle. Asserts each configuration is ``within`` the fp32 ``reduction_atol`` for
    the length-24 reduction — i.e. the running-max/running-sum rescaling trick is
    exact up to reduction rounding, independent of how the KV axis is tiled or masked.
    """
    g = _seed(3)
    for causal in (False, True):
        q = torch.randn(2, 8, 24, 32, generator=g)  # (B, H, S, D)
        k = torch.randn(2, 8, 24, 32, generator=g)
        v = torch.randn(2, 8, 24, 32, generator=g)
        oracle = softmax_attention(q, k, v, causal=causal)
        atol = reduction_atol(torch.float32, 24, scale=float(v.abs().mean()) + 1.0)
        for block_size in (4, 8, 16, 24):
            out = online_softmax_attention(q, k, v, block_size=block_size, causal=causal)
            rep = compare(out, oracle, atol=atol)
            assert rep.within, f"causal={causal} block={block_size}: {rep}"


def test_online_softmax_block_size_reorders_sum() -> None:
    """Goal: the attention analogue of the matmul tile-size test — confirm that the
    KV block size is a genuine reduction-schedule knob, not a no-op.

    Expectation: running online-softmax attention with a small KV tile (5) versus one
    spanning the whole sequence (40) reorders the weighted-value accumulation.
    Asserts the two outputs are NOT bitwise identical, proving block size really does
    reorder the sum (its effect is small — bounded by the tolerance checked in
    ``test_online_softmax_matches_oracle`` — but non-zero).
    """
    g = _seed(4)
    q = torch.randn(4, 40, 32, generator=g)
    k = torch.randn(4, 40, 32, generator=g)
    v = torch.randn(4, 40, 32, generator=g)
    o_small = online_softmax_attention(q, k, v, block_size=5)
    o_big = online_softmax_attention(q, k, v, block_size=40)
    assert not torch.equal(o_small, o_big), "different KV tiles should reorder the sum"


# --------------------------------------------------------------------------- #
# kv cache
# --------------------------------------------------------------------------- #
def test_incremental_decode_equals_oneshot_causal() -> None:
    """Goal: KV-cache decoding correctness — generating token-by-token while caching
    past keys/values must reproduce exactly what a single causal attention pass over
    the whole sequence would produce.

    Expectation: ``incremental_decode`` (step-wise, cached) and ``softmax_attention``
    with ``causal=True`` (one-shot) walk the same reductions in the same order, both
    in fp64. Asserts the two agree ``within`` a very tight ``atol=1e-10`` — not just
    tolerance-level but essentially bit-tight, since here the schedules are identical
    and only fp64 round-off separates them. This is the guarantee that streaming
    decode introduces no drift versus prefill.
    """
    g = _seed(5)
    q = torch.randn(3, 16, 32, generator=g)  # (H, T, D)
    k = torch.randn(3, 16, 32, generator=g)
    v = torch.randn(3, 16, 32, generator=g)
    step = incremental_decode(q, k, v)
    oneshot = softmax_attention(q, k, v, causal=True)
    rep = compare(step, oneshot, atol=1e-10)  # same schedule in fp64 -> tiny gap
    assert rep.within, f"incremental vs one-shot: {rep}"
