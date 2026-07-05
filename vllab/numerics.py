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


def to_cpu_fp64(t: torch.Tensor) -> torch.Tensor:
    """Move ``t`` to the CPU and cast it to fp64 — the lab's canonical "truth" form.

    Every oracle value and every candidate/oracle comparison is materialised here so
    that the reference numbers live in one device-independent, fully-precise place.
    This is a *deliberate* normalisation, not incidental plumbing:

    * fp64 is the oracle's contract, but Apple's MPS backend has no fp64 at all
      (casting an MPS tensor to fp64 raises ``TypeError``), and a GPU reduction is
      order-nondeterministic — neither is a trustworthy source of truth;
    * the CPU move happens *before* the fp64 cast, because that cast is the step MPS
      rejects; ``t.cpu()`` first sidesteps it.

    The trade-off is explicit: a candidate produced on an accelerator is pulled back
    to the CPU for comparison. That host hop is accepted in exchange for a single,
    reproducible reference — correctness over avoiding a copy.

    Future extension: if a caller ever needs the comparison to stay on-device (e.g. a
    CUDA-only fast path where fp64 *is* available), add an opt-in ``device`` argument
    here rather than re-deriving the cast at each call site.
    """
    return t.cpu().to(torch.float64)


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

    Parameters
    ----------
    dtype : torch.dtype
        Working precision of the *candidate* (not the oracle). The oracle runs in
        fp64, whose unit roundoff is negligible against the candidate's.
    reduction_length : int
        ``K``, the number of terms accumulated to produce a single output entry
        (the axis that addition collapses). See Notes for what ``K`` is per
        operator.
    scale : float, optional
        Representative magnitude of the result entries. Sets the regime the
        absolute error is measured against; the tolerance scales linearly with it.
    safety : float, optional
        Multiplier absorbing hidden constants and non-worst-case slack.

    Returns
    -------
    float
        The absolute tolerance ``safety * scale * sqrt(K) * u``, where ``u`` is the
        unit roundoff of ``dtype``.

    Notes
    -----
    The forward error of a floating-point sum of ``K`` terms grows like
    ``~sqrt(K) * u`` for a well-conditioned reduction (a random-walk model), and up
    to ``K * u`` in the adversarial worst case. We use the ``sqrt(K)`` model times a
    ``safety`` factor and a problem ``scale`` (typical magnitude of the accumulated
    result).

    ``K`` is whichever axis is summed away -- it differs by operator:

    - **matmul** ``C = A @ B`` with ``A`` of shape ``(M, K)`` and ``B`` of shape
      ``(K, N)``: each ``C[i, j] = sum_k A[i, k] * B[k, j]`` is a sum of ``K``
      products, so ``reduction_length`` is the shared inner dimension. For
      ``A: (128, 512)``, ``B: (512, 64)`` in fp32 with ``scale=1.0``::

          atol = 8 * 1.0 * sqrt(512) * 2**-23  ~= 2.16e-5

      Growing ``K`` to 4096 loosens the tolerance by ``sqrt(4096/512) ~= 2.8x``,
      because the longer sum genuinely accumulates more rounding.
    - **attention** for one query over ``S`` key/value positions: each output row
      ``out[i] = sum_s w[i, s] * V[s]`` is a sum over ``S`` terms, so
      ``reduction_length`` is the sequence / KV-cache length. Longer contexts earn
      proportionally more slack.

    ``scale`` is a *separate* knob from ``K``: ``K`` counts additions (drives
    ``sqrt(K)``) while ``scale`` sets the result magnitude. Keeping them apart is
    why the verdict in :func:`compare` is a single absolute check rather than a
    per-element ``rtol`` -- under cancellation a result entry can be tiny while its
    summands are large, and a ``rtol * |oracle|`` gate would under-tolerate a
    legitimately hard reduction.
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
    c = to_cpu_fp64(candidate)
    o = to_cpu_fp64(oracle)
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
