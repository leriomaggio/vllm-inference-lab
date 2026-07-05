"""L2 tests: paged decode reproduces the reference exactly, and a block-table
fault corrupts the output silently (same shape/dtype, different numbers)."""

from __future__ import annotations

import pytest
import torch

from vllab.numerics import compare
from vllab.paged.block_table import BlockTable
from vllab.paged.paged_attention import block_table_fault_demo, paged_decode
from vllab.reference.attention import softmax_attention
from vllab.reference.kvcache import incremental_decode


def _qkv(seed: int, heads: int = 3, seqlen: int = 17, dim: int = 16):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(heads, seqlen, dim, generator=g)
    k = torch.randn(heads, seqlen, dim, generator=g)
    v = torch.randn(heads, seqlen, dim, generator=g)
    return q, k, v


# --------------------------------------------------------------------------- #
# block table
# --------------------------------------------------------------------------- #
def test_block_table_allocation_and_locate() -> None:
    bt = BlockTable(num_blocks=8, block_size=4)
    bt.ensure_capacity(10)  # needs ceil(10/4) = 3 blocks
    assert bt.logical_blocks == 3
    # token 5 -> logical block 1, slot 1
    phys, slot = bt.locate(5)
    assert slot == 1
    assert phys == bt.mapping[1]


def test_block_table_pool_exhaustion_raises() -> None:
    bt = BlockTable(num_blocks=2, block_size=4)
    with pytest.raises(RuntimeError, match="out of KV blocks"):
        bt.ensure_capacity(100)


# --------------------------------------------------------------------------- #
# paged decode equivalence
# --------------------------------------------------------------------------- #
def test_paged_equals_reference_incremental() -> None:
    q, k, v = _qkv(0)
    reference = incremental_decode(q, k, v)
    for block_size in (1, 3, 4, 8, 17):
        paged = paged_decode(q, k, v, block_size=block_size)
        rep = compare(paged, reference, atol=1e-10)  # same fp64 schedule -> exact-ish
        assert rep.within, f"block_size={block_size}: {rep}"


def test_paged_equals_oneshot_causal() -> None:
    q, k, v = _qkv(1)
    paged = paged_decode(q, k, v, block_size=5)
    oneshot = softmax_attention(q, k, v, causal=True)
    rep = compare(paged, oneshot, atol=1e-10)
    assert rep.within, f"paged vs one-shot causal: {rep}"


# --------------------------------------------------------------------------- #
# fault injection
# --------------------------------------------------------------------------- #
def test_block_table_fault_is_silent_but_wrong() -> None:
    q, k, v = _qkv(2)
    report = block_table_fault_demo(q, k, v, block_size=4, corrupt_logical=0)
    # Silent: the output is still a valid tensor of the same shape and dtype.
    assert report.output_shape_unchanged
    # Wrong: the numbers diverge materially -- a latency benchmark would miss this.
    assert report.max_abs_diff > 1e-3, f"expected a visible divergence, got {report}"
    assert report.from_physical != report.to_physical
