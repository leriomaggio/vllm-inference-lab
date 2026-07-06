"""Shared result types for the engine layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil


@dataclass(frozen=True)
class GenResult:
    """A single prompt's greedy generation from one engine.

    Attributes
    ----------
    prompt : str
        The input prompt text.
    token_ids : list[int]
        Generated token ids (the continuation only, not the prompt).
    text : str
        Decoded continuation text.
    step_logprobs : list[float]
        Log-probability each engine assigned to the token it actually emitted at
        each step (one per generated token). Used for a fine-grained differential
        beyond exact token-id matching.
    """

    prompt: str
    token_ids: list[int]
    text: str
    step_logprobs: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class PromptFootprint:
    """How many KV-cache blocks one prompt's prefill occupies.

    Attributes
    ----------
    prompt : str
        The prompt text.
    tokens : int
        Number of tokens the prompt encodes to (its prefill length).
    blocks : int
        Logical KV blocks the prefill occupies, ``ceil(tokens / block_size)``.
        A decode of ``g`` further tokens grows this to
        ``ceil((tokens + g) / block_size)``.
    """

    prompt: str
    tokens: int
    blocks: int


@dataclass(frozen=True)
class KVCacheReport:
    """Introspected KV-cache configuration of a running engine.

    On vLLM's CPU backend the device KV pool is reported through the engine's
    *gpu-blocks* slot of its unified device abstraction (``num_cpu_blocks`` is
    ``None``); :attr:`num_blocks` holds whichever slot the backend populated.

    Attributes
    ----------
    block_size : int
        Tokens per KV block (page).
    num_blocks : int
        Total KV blocks the engine allocated for the device it runs on.
    cache_dtype : str
        The engine's KV-cache dtype (e.g. ``"auto"``).
    kv_bytes : int
        Bytes reserved for the KV-cache pool (``VLLM_CPU_KVCACHE_SPACE`` on CPU).
    per_prompt : list[PromptFootprint]
        Per-prompt prefill footprint, empty if not computed.

    Properties
    ----------
    capacity_tokens : int
        Total KV token-slots, ``block_size * num_blocks``.
    """

    block_size: int
    num_blocks: int
    cache_dtype: str
    kv_bytes: int
    per_prompt: list[PromptFootprint] = field(default_factory=list)

    @property
    def capacity_tokens(self) -> int:
        """Total KV token-slots the cache can hold (``block_size * num_blocks``)."""
        return self.block_size * self.num_blocks

    @staticmethod
    def blocks_for(tokens: int, block_size: int) -> int:
        """Logical blocks a sequence of ``tokens`` occupies at ``block_size``.

        Parameters
        ----------
        tokens : int
            Sequence length in tokens.
        block_size : int
            Tokens per KV block.

        Returns
        -------
        int
            ``ceil(tokens / block_size)``, and ``0`` when ``tokens`` is ``0``.
        """
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        return ceil(tokens / block_size)
