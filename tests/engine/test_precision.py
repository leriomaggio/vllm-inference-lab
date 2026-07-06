"""Precision-axis logic and its engine-level check (``vllab.engine.precision``).

Pure tests exercise the dtype-derived logprob band and the verdict flags offline.
The gated test confirms that on a real engine fp16's divergence from the fp32
oracle stays within its own band but exceeds the fp32 schedule-noise floor.
"""

from __future__ import annotations

import pytest

from vllab.engine.differential import DifferentialReport, PromptDiff, compare_generations
from vllab.engine.precision import logprob_band, precision_result


def _diff_with_gap(gap: float) -> DifferentialReport:
    """A single-prompt DifferentialReport with all tokens matching and a given gap."""
    return DifferentialReport(
        [PromptDiff("p", prefix_match=16, length=16, first_divergence=-1, max_logprob_gap=gap)]
    )


def test_logprob_band_ordering_and_growth() -> None:
    """Goal: the precision axis is only meaningful if the tolerance band tracks the two
    things that actually drive rounding — the working precision and the reduction
    length — so a looser dtype and a longer reduction must both widen the band.

    Expectation: at a fixed ``K=768``, the fp16 band exceeds the fp32 band (fp16's unit
    roundoff is ~8200x larger); and at fixed fp32, ``K=4096`` gives a wider band than
    ``K=768``. Asserts both orderings — the band is derived from ``sqrt(K)*u``, not a
    hand-picked constant, so it stays honest as either lever changes.
    """
    b32 = logprob_band("fp32", 768)
    b16 = logprob_band("fp16", 768)
    assert 0 < b32 < b16  # fp16 unit roundoff is ~8200x fp32's
    # band grows like sqrt(K): a longer reduction earns more slack.
    assert logprob_band("fp32", 4096) > logprob_band("fp32", 768)


def test_precision_result_flags_fp16_effect() -> None:
    """Goal: the central claim of the axis — a reduced-precision engine can agree on
    every token yet still be *measurably* running lower precision — must be classified
    correctly: within its own dtype band, but clearly above the fp32 schedule-noise
    floor.

    Expectation: an all-tokens-match diff with a gap of 8e-3 (the measured fp16
    magnitude), classified at fp16. Asserts ``token_exact`` (tokens agree),
    ``logprob_within_band`` (fine for fp16), ``exceeds_oracle_band`` (well past the
    fp32 floor), and ``band_ratio > 100`` — the precision effect is real, quantified,
    and not mistaken for a correctness failure.
    """
    # gap typical of measured fp16 (~8e-3): within fp16 band, far above fp32 floor.
    res = precision_result(_diff_with_gap(8e-3), candidate_dtype="fp16", reduction_length=768)
    assert res.token_exact
    assert res.logprob_within_band
    assert res.exceeds_oracle_band
    assert res.band_ratio > 100


def test_precision_result_fp32_is_schedule_noise() -> None:
    """Goal: the counterpart to the fp16 case — an fp32 candidate differs from the fp32
    oracle only by kernel *schedule*, so its gap must classify as mere schedule noise
    and never trip the precision-effect flag.

    Expectation: an all-tokens-match diff with a gap of 1.1e-5 (the measured fp32
    magnitude), classified at fp32. Asserts ``token_exact``, ``logprob_within_band``,
    and crucially ``not exceeds_oracle_band`` — fp32 sits at/under its own floor, so
    the flag distinguishes precision effects from ordinary schedule reordering.
    """
    # gap typical of measured fp32 (~1.1e-5): within the fp32 band, not above it.
    res = precision_result(_diff_with_gap(1.1e-5), candidate_dtype="fp32", reduction_length=768)
    assert res.token_exact
    assert res.logprob_within_band
    assert not res.exceeds_oracle_band


@pytest.mark.engine
def test_vllm_fp16_precision_axis(small_model: str, prompts: list[str]) -> None:
    """Goal: confirm the precision axis holds on a *real* engine, not just synthetic
    gaps — that vLLM in fp16, compared against the fp32 HuggingFace oracle, drifts by
    an amount that is tolerable for fp16 yet unmistakably beyond fp32 schedule noise.

    Expectation: HF fp32 reference vs vLLM fp16 on the shared prompts, banded at the
    model's hidden size. Asserts ``logprob_within_band`` (the drift is legitimate for
    fp16) and ``exceeds_oracle_band`` (it is larger than fp32 schedule noise) — the
    measured behaviour matches the classification the pure tests pin down. Gated: loads
    two engines.
    """
    hf_mod = pytest.importorskip("vllab.engine.hf_reference")
    vl_mod = pytest.importorskip("vllab.engine.vllm_runner")
    if not (hf_mod.transformers_available() and vl_mod.vllm_available()):
        pytest.skip("transformers or vLLM not installed")

    hf = hf_mod.HFReference(small_model, dtype="fp32")
    hidden = int(hf.model.config.hidden_size)
    reference = hf.greedy_generate(prompts, max_new_tokens=8)
    vllm = vl_mod.VLLMRunner(small_model, dtype="fp16")
    diff = compare_generations(reference, vllm.greedy_generate(prompts, max_new_tokens=8))
    res = precision_result(diff, candidate_dtype="fp16", reduction_length=hidden)
    # fp16 drift stays within its (loose) band but exceeds the fp32 schedule-noise
    # floor: a measurable precision effect, not a correctness failure.
    assert res.logprob_within_band
    assert res.exceeds_oracle_band
