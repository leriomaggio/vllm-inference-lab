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
        """Create an empty table over a pool of ``num_blocks`` free physical blocks.

        Parameters
        ----------
        num_blocks : int
            Size of the shared physical pool this table draws from. Must be
            positive; allocation raises once the pool is exhausted, which caps
            the sequence at ``num_blocks * block_size`` tokens.
        block_size : int
            Tokens per block (page). Must be positive. Fixes the ``divmod`` that
            splits an absolute token index into ``(logical_block, slot)``.

        Raises
        ------
        ValueError
            If ``num_blocks`` or ``block_size`` is not positive.
        """
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
        """Allocate physical blocks until the sequence can hold ``num_tokens``.

        Pops blocks from the free pool (lowest id first) and appends them to the
        logical->physical table until there are enough logical blocks to cover
        ``num_tokens``. A no-op when the current capacity already suffices.

        Parameters
        ----------
        num_tokens : int
            Total number of tokens the table must be able to address after this
            call.

        Raises
        ------
        RuntimeError
            If the free pool is exhausted before enough blocks are allocated.
        """
        needed = math.ceil(num_tokens / self.block_size)
        while len(self._table) < needed:
            if not self._free:
                raise RuntimeError("out of KV blocks: pool exhausted")
            self._table.append(self._free.pop(0))

    def locate(self, token_index: int) -> tuple[int, int]:
        """Resolve a token's absolute position to its physical block and slot.

        Parameters
        ----------
        token_index : int
            Zero-based position of the token in the sequence.

        Returns
        -------
        tuple[int, int]
            ``(physical_block_id, slot)`` where ``slot`` is the offset within the
            block (``token_index % block_size``).

        Raises
        ------
        IndexError
            If ``token_index`` is negative, or its logical block has not been
            allocated yet (call :meth:`ensure_capacity` first).
        """
        if token_index < 0:
            raise IndexError("token_index must be non-negative")
        logical, slot = divmod(token_index, self.block_size)
        if logical >= len(self._table):
            raise IndexError(f"token {token_index} is beyond allocated capacity")
        return self._table[logical], slot

    def corrupt(self, logical_index: int, wrong_physical: int) -> int:
        """Point a logical block at the wrong physical block (fault injection).

        Models a page-table bug: the output stays a valid-looking tensor, but the
        KV it reads comes from the wrong block. See :mod:`vllab.paged.faults` for
        the end-to-end demonstration.

        Parameters
        ----------
        logical_index : int
            Logical block whose mapping to overwrite. Must already be allocated.
        wrong_physical : int
            Physical block id to redirect it to. No bounds check is performed
            here; callers pick an in-range-but-wrong id so the read stays valid.

        Returns
        -------
        int
            The previous physical id, so the fault can be reverted.

        Raises
        ------
        IndexError
            If ``logical_index`` has not been allocated.
        """
        if logical_index >= len(self._table):
            raise IndexError("logical_index not allocated")
        prev = self._table[logical_index]
        self._table[logical_index] = wrong_physical
        return prev
