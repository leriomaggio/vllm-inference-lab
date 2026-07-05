"""Paged KV cache and paged incremental decode.

Keys and values live in a fixed-size **block pool**; a :class:`BlockTable` maps the
sequence's logical blocks to physical blocks in that pool. Appending a token writes
into the current block (allocating a new physical block when it fills); attending
gathers the scattered blocks back into a contiguous ``(H, T, D)`` view through the
table and runs the ordinary attention oracle.

The invariant that makes this testable: **paged decode must equal the non-paged
reference decode** (see :mod:`vllab.reference.kvcache`). Paging is a storage
optimisation; it must not change the output. A block-table fault therefore surfaces
only as a numerical divergence — never as a shape or dtype change — which is
exactly the kind of silent corruption a latency benchmark cannot see.

Convention: a sequence is ``(H, T, D)`` (heads, tokens, head_dim); a single token's
K/V is ``(H, D)``. The pools are ``(num_blocks, H, block_size, D)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllab.reference.attention import softmax_attention

from .block_table import BlockTable


class PagedKVCache:
    """A single sequence's KV cache backed by a paged block pool."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        block_size: int,
        num_blocks: int,
        scale: float | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self._scale = scale
        self._table = BlockTable(num_blocks, block_size)
        self._length = 0
        self._k = torch.zeros((num_blocks, num_heads, block_size, head_dim), dtype=dtype)
        self._v = torch.zeros((num_blocks, num_heads, block_size, head_dim), dtype=dtype)

    @property
    def length(self) -> int:
        return self._length

    @property
    def block_table(self) -> BlockTable:
        return self._table

    def append(self, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        """Append one token's keys/values, each shaped ``(H, D)``."""
        expected = (self.num_heads, self.head_dim)
        if k_t.shape != expected:
            raise ValueError(f"expected k_t {expected}, got {tuple(k_t.shape)}")
        self._length += 1
        self._table.ensure_capacity(self._length)
        phys, slot = self._table.locate(self._length - 1)
        self._k[phys, :, slot, :] = k_t
        self._v[phys, :, slot, :] = v_t

    def _gather(self, pool: torch.Tensor) -> torch.Tensor:
        """Reassemble ``(H, T, D)`` from scattered blocks via the block table."""
        if self._length == 0:
            raise RuntimeError("cache is empty; append before gather")
        chunks = []
        remaining = self._length
        logical = 0
        while remaining > 0:
            phys, _ = self._table.locate(logical * self.block_size)
            n = min(self.block_size, remaining)
            chunks.append(pool[phys, :, :n, :])  # (H, n, D)
            remaining -= n
            logical += 1
        return torch.cat(chunks, dim=1)  # (H, T, D)

    def attend(self, q_t: torch.Tensor) -> torch.Tensor:
        """Attend ``q_t`` ``(H, Sq, D)`` against the full gathered cache."""
        k_full = self._gather(self._k)
        v_full = self._gather(self._v)
        return softmax_attention(q_t, k_full, v_full, causal=False, scale=self._scale)


def paged_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int = 4,
    num_blocks: int | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Decode a sequence token-by-token through a :class:`PagedKVCache`.

    Parameters
    ----------
    q, k, v : torch.Tensor
        ``(H, T, D)`` sequences.
    block_size : int, optional
        Tokens per KV block.
    num_blocks : int or None, optional
        Pool size; defaults to exactly enough for the sequence.

    Returns
    -------
    torch.Tensor
        ``(H, T, D)`` — must equal the non-paged reference decode within fp64 noise.
    """
    heads, seqlen, dim = q.shape
    if num_blocks is None:
        num_blocks = -(-seqlen // block_size)  # ceil
    cache = PagedKVCache(
        num_heads=heads,
        head_dim=dim,
        block_size=block_size,
        num_blocks=num_blocks,
        scale=scale,
    )
    outs = []
    for t in range(seqlen):
        cache.append(k[:, t, :], v[:, t, :])
        outs.append(cache.attend(q[:, t : t + 1, :]))
    return torch.cat(outs, dim=1)


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
