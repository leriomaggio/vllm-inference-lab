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
        block_table_cls: type[BlockTable] = BlockTable,
    ) -> None:
        """Allocate empty K and V block pools for a single sequence.

        Parameters
        ----------
        num_heads : int
            Number of attention heads ``H``.
        head_dim : int
            Per-head feature dimension ``D``.
        block_size : int
            Tokens per KV block (page).
        num_blocks : int
            Physical blocks in the pool. Caps the sequence at
            ``num_blocks * block_size`` tokens before the pool is exhausted.
        scale : float or None, optional
            Softmax scale forwarded to the attention oracle; ``None`` uses
            ``1 / sqrt(head_dim)``.
        dtype : torch.dtype, optional
            Storage dtype of the K and V pools (default ``torch.float32``).
        block_table_cls : type[BlockTable], optional
            Block-table implementation to back the cache. Defaults to the plain
            :class:`BlockTable`; the fault-injection demo swaps in
            :class:`vllab.paged.faults.FaultyBlockTable` to corrupt the mapping.
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self._scale = scale
        self._table = block_table_cls(num_blocks, block_size)
        self._length = 0
        self._k = torch.zeros((num_blocks, num_heads, block_size, head_dim), dtype=dtype)
        self._v = torch.zeros((num_blocks, num_heads, block_size, head_dim), dtype=dtype)

    @property
    def length(self) -> int:
        """Number of tokens currently cached."""
        return self._length

    @property
    def block_table(self) -> BlockTable:
        """The underlying logical->physical block mapping (for inspection/faults)."""
        return self._table

    def append(self, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        """Append one token's keys and values to the cache.

        Grows the block table when the new token spills into a fresh block, then
        writes the token into its ``(physical_block, slot)`` cell.

        Parameters
        ----------
        k_t : torch.Tensor
            This token's keys, shaped ``(H, D)``.
        v_t : torch.Tensor
            This token's values, shaped ``(H, D)``.

        Raises
        ------
        ValueError
            If ``k_t`` is not shaped ``(num_heads, head_dim)``.
        """
        expected = (self.num_heads, self.head_dim)
        if k_t.shape != expected:
            raise ValueError(f"expected k_t {expected}, got {tuple(k_t.shape)}")
        self._length += 1
        self._table.ensure_capacity(self._length)
        phys, slot = self._table.locate(self._length - 1)
        self._k[phys, :, slot, :] = k_t
        self._v[phys, :, slot, :] = v_t

    def _gather(self, pool: torch.Tensor) -> torch.Tensor:
        """Reassemble a contiguous ``(H, T, D)`` view from the scattered blocks.

        Walks the logical blocks in order, resolves each through the block table,
        slices out its live tokens, and concatenates along the token axis. This is
        the read path a block-table fault corrupts silently.

        Parameters
        ----------
        pool : torch.Tensor
            Either the key or the value pool, shaped
            ``(num_blocks, H, block_size, D)``.

        Returns
        -------
        torch.Tensor
            ``(H, T, D)`` with ``T == self.length`` tokens in sequence order.

        Raises
        ------
        RuntimeError
            If the cache is empty (nothing has been appended yet).
        """
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
        """Attend a query against the full gathered cache.

        Parameters
        ----------
        q_t : torch.Tensor
            Query block shaped ``(H, Sq, D)`` (``Sq == 1`` for a single decode
            step).

        Returns
        -------
        torch.Tensor
            ``(H, Sq, D)`` attention output (fp64, from the oracle softmax).
        """
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
        Pool size; defaults to exactly enough for the sequence
        (``ceil(T / block_size)``).
    scale : float or None, optional
        Softmax scale forwarded to each attention step; ``None`` uses
        ``1 / sqrt(head_dim)``.

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
