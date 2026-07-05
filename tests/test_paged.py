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
    """Goal: the block table is the logical->physical indirection at the heart of
    paging; reserving capacity must claim exactly the right number of physical blocks
    and the address arithmetic must resolve a token to the correct ``(block, slot)``.

    Expectation: asking for room for 10 tokens at ``block_size=4`` allocates
    ``ceil(10/4) = 3`` logical blocks, and token 5 resolves to logical block 1, slot 1
    (``divmod(5, 4)``). Asserts the block count, the slot, and that ``locate`` agrees
    with the published ``mapping`` — i.e. the divmod addressing and the pool
    bookkeeping stay consistent, so no token can be silently mis-addressed.
    """
    bt = BlockTable(num_blocks=8, block_size=4)
    bt.ensure_capacity(10)  # needs ceil(10/4) = 3 blocks
    assert bt.logical_blocks == 3
    # token 5 -> logical block 1, slot 1
    phys, slot = bt.locate(5)
    assert slot == 1
    assert phys == bt.mapping[1]


def test_block_table_pool_exhaustion_raises() -> None:
    """Goal: the block pool is finite, so requesting more KV blocks than exist must
    fail loudly rather than over-allocate or wrap around — the paging analogue of OOM.

    Expectation: a 2-block pool cannot back 100 tokens; ``ensure_capacity`` must raise
    ``RuntimeError`` naming the exhausted pool. Asserts the raise (and its message), so
    exhaustion surfaces as an explicit error and never as a quietly corrupted mapping.
    """
    bt = BlockTable(num_blocks=2, block_size=4)
    with pytest.raises(RuntimeError, match="out of KV blocks"):
        bt.ensure_capacity(100)


# --------------------------------------------------------------------------- #
# paged decode equivalence
# --------------------------------------------------------------------------- #
def test_paged_equals_reference_incremental() -> None:
    """Goal: the defining invariant of L2 — paging is a storage layout, not a change
    in math — so paged decode must reproduce the non-paged reference decode no matter
    how the sequence is chopped into blocks.

    Expectation: sweeping ``block_size`` over sizes that divide, don't divide, and
    exceed the 17-token sequence (1, 3, 4, 8, 17) must all match ``incremental_decode``.
    Both walk the same fp64 schedule, so asserts each stays ``within`` a near-exact
    ``atol=1e-10`` — the block size changes where the K/V physically live, never the
    result. This is the correctness bar every faithful paging scheme has to clear.
    """
    q, k, v = _qkv(0)
    reference = incremental_decode(q, k, v)
    for block_size in (1, 3, 4, 8, 17):
        paged = paged_decode(q, k, v, block_size=block_size)
        rep = compare(paged, reference, atol=1e-10)  # same fp64 schedule -> exact-ish
        assert rep.within, f"block_size={block_size}: {rep}"


def test_paged_equals_oneshot_causal() -> None:
    """Goal: chain the L2 invariant back to first principles — paged decode must equal
    not just the reference cache but a one-shot causal attention pass over the whole
    sequence, closing the loop ``paged == incremental == prefill``.

    Expectation: ``paged_decode`` with an arbitrary ``block_size`` (5) versus
    ``softmax_attention`` with ``causal=True``. Asserts agreement ``within``
    ``atol=1e-10`` — the same fp64 schedule, so only round-off separates them,
    confirming paging is invisible end-to-end and not merely against its own cache.
    """
    q, k, v = _qkv(1)
    paged = paged_decode(q, k, v, block_size=5)
    oneshot = softmax_attention(q, k, v, causal=True)
    rep = compare(paged, oneshot, atol=1e-10)
    assert rep.within, f"paged vs one-shot causal: {rep}"


# --------------------------------------------------------------------------- #
# fault injection
# --------------------------------------------------------------------------- #
def test_block_table_fault_is_silent_but_wrong() -> None:
    """Goal: the payoff experiment — a mis-mapped block table is the classic paging
    bug, and the point is that it corrupts the output *silently*: the tensor keeps a
    valid shape and dtype, so a latency benchmark sees nothing while the numbers are
    wrong.

    Expectation: ``block_table_fault_demo`` repoints logical block 0 at a different
    physical block and recomputes the final-token output. Asserts three things —
    ``output_shape_unchanged`` (silent: still a valid-looking tensor), ``max_abs_diff
    > 1e-3`` (materially wrong, not low-bit noise), and that the fault actually moved
    the mapping (``from_physical != to_physical``). Together: paging correctness bugs
    hide from any check that only inspects shape or timing — you have to compare values.
    """
    q, k, v = _qkv(2)
    report = block_table_fault_demo(q, k, v, block_size=4, corrupt_logical=0)
    # Silent: the output is still a valid tensor of the same shape and dtype.
    assert report.output_shape_unchanged
    # Wrong: the numbers diverge materially -- a latency benchmark would miss this.
    assert report.max_abs_diff > 1e-3, f"expected a visible divergence, got {report}"
    assert report.from_physical != report.to_physical
