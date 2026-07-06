"""Precision axis for the L3 engine: a reduced-precision engine vs the fp32 oracle.

The differential harness (:mod:`vllab.engine.differential`) compares two engines'
greedy generations. This module puts a *tolerance* on that comparison, derived the
same way as everywhere else in the lab — from the working precision and the
reduction length (:func:`vllab.numerics.reduction_atol`).

The reduction behind every next-token logit is the final projection
``hidden_state @ lm_head.T``, a sum over the model's **hidden size**. That is the
``K`` that sets how far a candidate engine's per-step log-probabilities may drift
from the fp32 oracle before the drift is more than rounding.

Two regimes, one mechanism
--------------------------
* **fp32 candidate** (e.g. vLLM fp32) vs fp32 oracle: identical math up to kernel
  *schedule*. Token ids must match exactly (greedy argmax is unchanged); the
  logprob gap is tiny and must sit within the fp32 band.
* **fp16 candidate** vs fp32 oracle: reduced precision perturbs the logits by the
  fp16 unit roundoff. Token ids may still match when the argmax margin is wide, but
  the logprob gap grows by roughly the ``u_fp16 / u_fp32`` ratio — visible in the
  numbers even when the tokens agree. It must sit within the (much looser) fp16
  band, while *exceeding* the fp32 band: that excess is the precision effect made
  measurable, not a correctness failure.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..numerics import reduction_atol
from .differential import DifferentialReport
from .hf_reference import resolve_dtype


def logprob_band(
    dtype: str,
    reduction_length: int,
    *,
    scale: float = 1.0,
    safety: float = 8.0,
) -> float:
    """Tolerance for a per-step log-probability gap at a given candidate precision.

    Delegates to :func:`vllab.numerics.reduction_atol` with the candidate dtype and
    the logit reduction length, so the band is the same ``safety·scale·√K·u`` form
    used to validate the matmul and attention layers.

    Parameters
    ----------
    dtype : {"fp32", "fp16", "bf16"}
        The *candidate* engine's precision (the oracle is fp32).
    reduction_length : int
        ``K`` for the logit projection — the model's hidden size.
    scale : float, optional
        Representative magnitude regime for the compared quantity. Default ``1.0``
        (bounds the per-unit-magnitude logprob perturbation).
    safety : float, optional
        Slack multiplier; the lab-wide default is ``8.0``.

    Returns
    -------
    float
        The absolute tolerance the per-step logprob gap is checked against.
    """
    return reduction_atol(resolve_dtype(dtype), reduction_length, scale=scale, safety=safety)


@dataclass(frozen=True)
class PrecisionResult:
    """A candidate precision's agreement with the fp32 oracle, plus its tolerance.

    Attributes
    ----------
    candidate_dtype : str
        The candidate engine's precision (e.g. ``"fp16"``).
    oracle_dtype : str
        The oracle's precision (``"fp32"``).
    reduction_length : int
        ``K`` used to derive the bands (the model hidden size).
    band : float
        Logprob-gap tolerance at ``candidate_dtype``.
    oracle_band : float
        Logprob-gap tolerance at ``oracle_dtype`` — the fp32 schedule-noise floor,
        kept so the report can say how far the candidate's gap exceeds it.
    diff : DifferentialReport
        The token/logprob comparison of candidate vs oracle generations.
    """

    candidate_dtype: str
    oracle_dtype: str
    reduction_length: int
    band: float
    oracle_band: float
    diff: DifferentialReport

    @property
    def token_exact(self) -> bool:
        """Whether every prompt's tokens match the oracle exactly."""
        return self.diff.exact_match_fraction == 1.0

    @property
    def logprob_within_band(self) -> bool:
        """Whether the max per-step logprob gap is within the candidate band."""
        return self.diff.max_logprob_gap <= self.band

    @property
    def exceeds_oracle_band(self) -> bool:
        """Whether the gap exceeds the fp32 schedule-noise floor.

        For a reduced-precision candidate this is expected and *is* the precision
        signal: the divergence is larger than mere kernel-schedule noise. For an
        fp32 candidate it should be ``False``.
        """
        return self.diff.max_logprob_gap > self.oracle_band

    @property
    def band_ratio(self) -> float:
        """How many multiples of the fp32 band the observed gap is.

        ``max_logprob_gap / oracle_band`` — ``1.0`` means the gap sits right at the
        fp32 schedule-noise floor; large values quantify the precision effect.
        """
        if self.oracle_band <= 0:
            return float("inf")
        return self.diff.max_logprob_gap / self.oracle_band


def precision_result(
    diff: DifferentialReport,
    *,
    candidate_dtype: str,
    reduction_length: int,
    oracle_dtype: str = "fp32",
    scale: float = 1.0,
    safety: float = 8.0,
) -> PrecisionResult:
    """Attach dtype-derived tolerance bands to a candidate-vs-oracle comparison.

    Parameters
    ----------
    diff : DifferentialReport
        Result of comparing the candidate's generations against the fp32 oracle's.
    candidate_dtype : {"fp32", "fp16", "bf16"}
        The candidate engine's precision.
    reduction_length : int
        ``K`` for the logit projection (model hidden size).
    oracle_dtype : str, optional
        The oracle's precision. Default ``"fp32"``.
    scale, safety : float, optional
        Passed through to :func:`logprob_band`.

    Returns
    -------
    PrecisionResult
        The comparison bundled with its candidate and oracle bands.
    """
    return PrecisionResult(
        candidate_dtype=candidate_dtype,
        oracle_dtype=oracle_dtype,
        reduction_length=reduction_length,
        band=logprob_band(candidate_dtype, reduction_length, scale=scale, safety=safety),
        oracle_band=logprob_band(oracle_dtype, reduction_length, scale=scale, safety=safety),
        diff=diff,
    )
