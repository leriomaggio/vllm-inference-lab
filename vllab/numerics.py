"""Shared numerical-comparison utilities and tolerance derivation.

The whole lab rests on comparing a candidate schedule against an oracle *within a
tolerance justified by the working precision and the reduction length*, rather than
demanding bitwise equality. These helpers make that discipline explicit and
reusable across layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# The precisions this lab exercises. We take the unit roundoff from
# ``torch.finfo(dtype).eps`` (machine epsilon = 2**-mantissa_bits), used directly
# as a conservative per-rounding bound rather than the eps/2 convention.
_SUPPORTED_DTYPES = frozenset(
    {torch.float64, torch.float32, torch.float16, torch.bfloat16}
)


def unit_roundoff(dtype: torch.dtype) -> float:
    """Return the unit roundoff ``u`` for ``dtype``."""
    if dtype not in _SUPPORTED_DTYPES:
        raise KeyError(f"no unit roundoff registered for {dtype}")
    return torch.finfo(dtype).eps


def reduction_atol(
    dtype: torch.dtype,
    reduction_length: int,
    *,
    scale: float = 1.0,
    safety: float = 8.0,
) -> float:
    """An absolute tolerance for a length-``K`` reduction in ``dtype``.

    The forward error of a floating-point sum of ``K`` terms grows like
    ``~sqrt(K) * u`` for a well-conditioned reduction (and up to ``K * u`` in the
    worst case). We use the ``sqrt(K)`` model times a safety factor and a problem
    ``scale`` (typical magnitude of the accumulated result).

    Args:
        dtype: Working precision of the *candidate* (not the oracle).
        reduction_length: ``K``, the number of accumulated terms.
        scale: Representative magnitude of the result entries.
        safety: Multiplier absorbing constants and non-worst-case slack.
    """
    return safety * scale * math.sqrt(max(reduction_length, 1)) * unit_roundoff(dtype)


@dataclass(frozen=True)
class DiffReport:
    """Summary of the discrepancy between two tensors."""

    max_abs: float
    mean_abs: float
    max_rel: float
    within: bool
    atol: float

    def __str__(self) -> str:
        verdict = "within" if self.within else "OVER"
        return (
            f"max_abs={self.max_abs:.3e} mean_abs={self.mean_abs:.3e} "
            f"max_rel={self.max_rel:.3e} [{verdict} atol={self.atol:.3e}]"
        )


def compare(candidate: torch.Tensor, oracle: torch.Tensor, *, atol: float) -> DiffReport:
    """Compare ``candidate`` against ``oracle`` in fp64 and summarise the gap.

    The verdict is a single absolute check (``max_abs <= atol``): there is no
    ``rtol``. The relative dimension lives in :func:`reduction_atol`'s ``scale``
    argument, which sets the magnitude regime the tolerance is measured against —
    the right quantity for a reduction, where cancellation can make a result entry
    tiny while its summands are large (a per-element ``rtol * |oracle|`` would then
    under-tolerate a legitimately hard reduction). ``max_rel`` is reported for
    diagnostics only and does not gate the verdict.
    """
    c = candidate.to(torch.float64)
    o = oracle.to(torch.float64)
    if c.shape != o.shape:
        raise ValueError(f"shape mismatch: {tuple(c.shape)} vs {tuple(o.shape)}")
    diff = (c - o).abs()
    denom = o.abs().clamp_min(torch.finfo(torch.float64).tiny)
    max_abs = float(diff.max())
    return DiffReport(
        max_abs=max_abs,
        mean_abs=float(diff.mean()),
        max_rel=float((diff / denom).max()),
        within=max_abs <= atol,
        atol=atol,
    )
