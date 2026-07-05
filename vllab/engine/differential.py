"""Differential comparison between two engines' greedy generations.

Greedy decoding is deterministic, so two correct engines should emit the *same*
tokens. The primary signal is therefore the length of the identical token prefix
(where the sequences first diverge); the secondary signal is how far apart the two
engines' per-step log-probabilities are over that shared prefix.

Once the token sequences diverge, every later token is conditioned on a different
context, so comparisons past the first divergence are not meaningful — hence the
prefix framing.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import GenResult


@dataclass(frozen=True)
class PromptDiff:
    """Per-prompt comparison of two engines."""

    prompt: str
    prefix_match: int  # length of identical leading token run
    length: int  # max of the two continuation lengths
    first_divergence: int  # index of first mismatch, or -1 if identical
    max_logprob_gap: float  # over the shared prefix

    @property
    def identical(self) -> bool:
        return self.first_divergence == -1


@dataclass(frozen=True)
class DifferentialReport:
    """Aggregate comparison over a prompt set."""

    per_prompt: list[PromptDiff]

    @property
    def exact_match_fraction(self) -> float:
        if not self.per_prompt:
            return 0.0
        return sum(p.identical for p in self.per_prompt) / len(self.per_prompt)

    @property
    def token_agreement_rate(self) -> float:
        """Fraction of positions that agree, pooled over prompts (prefix / length)."""
        total = sum(p.length for p in self.per_prompt)
        if total == 0:
            return 0.0
        return sum(p.prefix_match for p in self.per_prompt) / total

    @property
    def max_logprob_gap(self) -> float:
        return max((p.max_logprob_gap for p in self.per_prompt), default=0.0)


def _prefix_match(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def compare_generations(
    reference: list[GenResult],
    candidate: list[GenResult],
) -> DifferentialReport:
    """Compare candidate generations against a reference, prompt by prompt."""
    if len(reference) != len(candidate):
        raise ValueError("reference and candidate must have the same number of prompts")

    diffs: list[PromptDiff] = []
    for ref, cand in zip(reference, candidate, strict=True):
        if ref.prompt != cand.prompt:
            raise ValueError(f"prompt mismatch: {ref.prompt!r} vs {cand.prompt!r}")
        match = _prefix_match(ref.token_ids, cand.token_ids)
        length = max(len(ref.token_ids), len(cand.token_ids))
        first_div = -1 if match == length else match

        gap = 0.0
        for i in range(min(match, len(ref.step_logprobs), len(cand.step_logprobs))):
            gap = max(gap, abs(ref.step_logprobs[i] - cand.step_logprobs[i]))

        diffs.append(PromptDiff(ref.prompt, match, length, first_div, gap))
    return DifferentialReport(diffs)
