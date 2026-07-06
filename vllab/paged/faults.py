"""Block-table fault injection.

A block-table fault surfaces only as a numerical divergence — never as a shape or
dtype change — which is exactly the kind of silent corruption a latency benchmark
cannot see. The demo here fills a paged cache, records the clean output, repoints one
logical block at a different physical block, and recomputes; the returned
:class:`FaultReport` lets a human (via the CLI) and a test assert the same evidence:
same shape, same dtype, wrong numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .paged_attention import PagedKVCache


@dataclass(frozen=True)
class FaultReport:
    """Result of a block-table fault-injection experiment."""

    max_abs_diff: float
    output_shape_unchanged: bool
    corrupted_logical: int
    from_physical: int
    to_physical: int


def block_table_fault_demo(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int = 4,
    corrupt_logical: int = 0,
    scale: float | None = None,
) -> FaultReport:
    """Show that a mis-mapped block table silently corrupts the output.

    Fills a paged cache with the sequence, records the clean final-token output,
    then repoints one logical block at a different physical block and recomputes.
    The output stays the same shape and dtype — only the numbers change.
    """
    heads, seqlen, dim = q.shape
    # Two extra blocks so there is a valid-but-wrong physical target to point at.
    num_blocks = -(-seqlen // block_size) + 2
    cache = PagedKVCache(
        num_heads=heads,
        head_dim=dim,
        block_size=block_size,
        num_blocks=num_blocks,
        scale=scale,
    )
    for t in range(seqlen):
        cache.append(k[:, t, :], v[:, t, :])

    q_last = q[:, -1:, :]
    clean = cache.attend(q_last)

    mapping = cache.block_table.mapping
    if corrupt_logical >= len(mapping):
        raise IndexError("corrupt_logical is beyond the allocated blocks")
    # Point at some other in-range physical block (wrap to stay valid).
    wrong = (mapping[corrupt_logical] + 1) % num_blocks
    prev = cache.block_table.corrupt(corrupt_logical, wrong)
    faulty = cache.attend(q_last)

    diff = float((clean.to(torch.float64) - faulty.to(torch.float64)).abs().max())
    return FaultReport(
        max_abs_diff=diff,
        output_shape_unchanged=(clean.shape == faulty.shape and clean.dtype == faulty.dtype),
        corrupted_logical=corrupt_logical,
        from_physical=prev,
        to_physical=wrong,
    )
