"""Pure KV-cache footprint math (``vllab.engine.types.KVCacheReport``)."""

from __future__ import annotations

import pytest

from vllab.engine.types import KVCacheReport


def test_kv_blocks_for_rounds_up() -> None:
    """Goal: a sequence occupies whole KV pages, so the logical-block count must round
    a partial page *up* to a full block — the accounting that explains why short
    prompts still consume an entire 128-token block.

    Expectation: at ``block_size=128``, 0 tokens need 0 blocks, 1 and 128 tokens both
    need 1 (a partial page still costs a whole block), and 129 needs 2. Asserts each
    boundary, plus that a non-positive ``block_size`` raises ``ValueError`` — the
    ceil-division is exact at the edges and rejects nonsensical page sizes.
    """
    assert KVCacheReport.blocks_for(0, 128) == 0
    assert KVCacheReport.blocks_for(1, 128) == 1
    assert KVCacheReport.blocks_for(128, 128) == 1
    assert KVCacheReport.blocks_for(129, 128) == 2
    with pytest.raises(ValueError, match="block_size"):
        KVCacheReport.blocks_for(10, 0)


def test_kv_capacity_tokens() -> None:
    """Goal: the headline "capacity" the introspection reports is the total KV
    token-slots, which is simply block size times block count; this pins that
    derivation so the reported number always means the same thing.

    Expectation: a report with ``block_size=128`` and ``num_blocks=455`` (the measured
    CPU-backend layout). Asserts ``capacity_tokens == 128 * 455`` — capacity is the
    product, not any of the byte-budget fields alongside it.
    """
    rep = KVCacheReport(block_size=128, num_blocks=455, cache_dtype="auto", kv_bytes=2**32)
    assert rep.capacity_tokens == 128 * 455
