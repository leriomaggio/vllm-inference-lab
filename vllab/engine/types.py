"""Shared result types for the engine layer."""

from __future__ import annotations

from dataclasses import dataclass, field


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
