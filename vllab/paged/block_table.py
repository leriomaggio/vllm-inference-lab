"""Block table: the logical-to-physical indirection behind paged KV storage.

A sequence's KV cache is split into fixed-size **blocks** (pages). The block table
maps the sequence's *logical* block indices (0, 1, 2, ... in token order) to
*physical* block ids in a shared pool. This is the same idea as virtual memory:
contiguous logical addresses backed by scattered physical frames, which is what
lets vLLM pack many sequences into one pool without fragmentation and share a
prefix by pointing two tables at the same physical block.

This module models a **single sequence's** table over a pool of ``num_blocks``
physical blocks. The pool bookkeeping is deliberately explicit so a mis-mapping
(the classic paging bug) can be injected and observed.
"""

from __future__ import annotations

import math


class BlockTable:
    """Maps logical block indices to physical block ids from a shared free pool."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.block_size = block_size
        self._free: list[int] = list(range(num_blocks))
        self._table: list[int] = []  # logical index -> physical block id

    @property
    def logical_blocks(self) -> int:
        """Number of logical blocks currently allocated."""
        return len(self._table)

    @property
    def mapping(self) -> list[int]:
        """A copy of the logical->physical mapping."""
        return list(self._table)

    def ensure_capacity(self, num_tokens: int) -> None:
        """Allocate physical blocks until the sequence can hold ``num_tokens``."""
        needed = math.ceil(num_tokens / self.block_size)
        while len(self._table) < needed:
            if not self._free:
                raise RuntimeError("out of KV blocks: pool exhausted")
            self._table.append(self._free.pop(0))

    def locate(self, token_index: int) -> tuple[int, int]:
        """Return ``(physical_block_id, slot)`` for a token's position.

        ``slot`` is the offset within the block. Raises if the token's logical
        block has not been allocated yet.
        """
        if token_index < 0:
            raise IndexError("token_index must be non-negative")
        logical, slot = divmod(token_index, self.block_size)
        if logical >= len(self._table):
            raise IndexError(f"token {token_index} is beyond allocated capacity")
        return self._table[logical], slot

    def corrupt(self, logical_index: int, wrong_physical: int) -> int:
        """Point a logical block at the wrong physical block (fault injection).

        Returns the previous physical id so the fault can be reverted. This models
        a page-table bug: the output stays a valid-looking tensor, but the KV it
        reads comes from the wrong block.
        """
        if logical_index >= len(self._table):
            raise IndexError("logical_index not allocated")
        prev = self._table[logical_index]
        self._table[logical_index] = wrong_physical
        return prev
