"""Pure differential-comparison logic (``vllab.engine.differential``).

Greedy decoding is deterministic, so two correct engines emit the same tokens;
these tests exercise the prefix/divergence/logprob-gap accounting offline.
"""

from __future__ import annotations

import pytest

from vllab.engine.differential import compare_generations
from vllab.engine.types import GenResult


def test_compare_identical_generations() -> None:
    """Goal: the baseline the whole differential rests on — two engines that emitted
    the exact same tokens with the exact same logprobs must be scored as perfect
    agreement, with no spurious divergence signal.

    Expectation: identical reference and candidate generations. Asserts
    ``exact_match_fraction`` and ``token_agreement_rate`` are both 1.0 and the
    ``max_logprob_gap`` is exactly 0.0 — the comparator reports zero difference when
    there genuinely is none, so any non-zero gap later is a real signal.
    """
    ref = [GenResult("p", [1, 2, 3], "x", [-0.1, -0.2, -0.3])]
    cand = [GenResult("p", [1, 2, 3], "x", [-0.1, -0.2, -0.3])]
    report = compare_generations(ref, cand)
    assert report.exact_match_fraction == 1.0
    assert report.token_agreement_rate == 1.0
    assert report.max_logprob_gap == 0.0


def test_compare_detects_first_divergence() -> None:
    """Goal: once greedy sequences diverge, every later token is conditioned on a
    different context, so the comparator must locate the *first* mismatch and stop
    trusting anything past it rather than counting later coincidental matches.

    Expectation: sequences share a 2-token prefix then differ at index 2 (with a
    later coincidental match at index 3). Asserts ``prefix_match == 2``,
    ``first_divergence == 2``, ``not identical``, and that the logprob gap is measured
    only over the shared prefix (0.0 here, ignoring the divergent position) — the
    prefix framing, not a naive position-wise equality count.
    """
    ref = [GenResult("p", [1, 2, 3, 4], "x", [-0.1, -0.2, -0.3, -0.4])]
    cand = [GenResult("p", [1, 2, 9, 4], "y", [-0.1, -0.2, -0.9, -0.4])]
    report = compare_generations(ref, cand)
    d = report.per_prompt[0]
    assert d.prefix_match == 2
    assert d.first_divergence == 2
    assert not d.identical
    # logprob gap is measured only over the shared prefix (positions 0..1).
    assert d.max_logprob_gap == pytest.approx(0.0, abs=1e-9)


def test_compare_logprob_gap_over_prefix() -> None:
    """Goal: token agreement is the coarse signal; the finer one is *how confidently*
    each engine emitted the shared tokens, so the comparator must surface the largest
    per-step logprob gap even when the tokens themselves all match.

    Expectation: identical tokens but a 0.05 logprob difference at the first position.
    Asserts ``max_logprob_gap`` recovers 0.05 — the secondary numeric signal that lets
    a reduced-precision engine be flagged even when its argmax never flips.
    """
    ref = [GenResult("p", [1, 2], "x", [-0.10, -0.20])]
    cand = [GenResult("p", [1, 2], "x", [-0.15, -0.20])]
    report = compare_generations(ref, cand)
    assert report.per_prompt[0].max_logprob_gap == pytest.approx(0.05, abs=1e-9)


def test_compare_prompt_mismatch_raises() -> None:
    """Goal: a differential is only meaningful if the two engines were asked the same
    question, so comparing generations from *different* prompts is a caller error that
    must fail loudly rather than produce a misleading agreement number.

    Expectation: reference and candidate whose prompts differ. Asserts a ``ValueError``
    naming the mismatch — misaligned inputs surface as an explicit error, never as a
    silently wrong score.
    """
    with pytest.raises(ValueError, match="prompt mismatch"):
        compare_generations([GenResult("a", [1], "", [])], [GenResult("b", [1], "", [])])
