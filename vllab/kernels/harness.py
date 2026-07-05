"""L1 validation harness: compare kernel schedules against the fp64 oracle.

Runs on any CPU. It always validates the pure-PyTorch tiled-reduction *model*
(:func:`vllab.reference.matmul.tiled_matmul`) and, when Triton is importable, also
the real :func:`vllab.kernels.matmul_triton.triton_matmul` kernel — both against
:func:`vllab.reference.matmul.reference_matmul`. It also quantifies the low-bit
divergence produced purely by changing the tile size (the schedule), which is the
reason cross-backend correctness is tolerance-defined rather than bitwise.

The layer's thesis in one place: a matmul has one mathematical definition but many
valid *schedules* (tile sizes, backends), and they disagree in the low bits because
floating-point addition is not associative. This harness makes that disagreement
measurable — every schedule is graded against the same fp64 oracle within a
tolerance derived from the reduction length (:func:`vllab.numerics.reduction_atol`),
and the raw schedule-to-schedule gap is reported alongside so the "why tolerance,
not equality" argument rests on a number rather than an assertion.

The output is a :class:`MatmulCheck` — a plain data record, not a pass/fail
side effect — so callers (a notebook, a CLI table, a regression test) decide how to
present or gate on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllab.numerics import compare, reduction_atol
from vllab.reference.matmul import reference_matmul, tiled_matmul

from .matmul_triton import triton_available, triton_matmul


@dataclass(frozen=True)
class KernelRow:
    """One validated schedule: a single (backend, config) graded against the oracle.

    Attributes
    ----------
    backend : str
        Which implementation produced the result — ``"torch-tiled"`` for the
        pure-PyTorch model, ``"triton"`` for the real kernel.
    config : str
        Human-readable schedule knob for this row, e.g. ``"block_k=32"`` or
        ``"BK=32"``. Kept as a label rather than structured fields because it is only
        ever displayed, never computed on.
    max_abs : float
        Largest absolute element-wise gap from the oracle (from :func:`compare`).
    atol : float
        The tolerance this row was graded against — the same value for every row of a
        given check, carried per-row so a row is self-contained when tabulated.
    within : bool
        Verdict: ``max_abs <= atol``. This is the pass/fail for the row.
    """

    backend: str
    config: str
    max_abs: float
    atol: float
    within: bool


@dataclass(frozen=True)
class MatmulCheck:
    """Result of a matmul kernel check across schedules for one problem shape.

    Attributes
    ----------
    shape : tuple[int, int, int]
        The ``(M, K, N)`` problem validated. ``K`` is the reduction length that sets
        the tolerance, so it is carried alongside the rows for context.
    rows : list[KernelRow]
        One graded row per (backend, tile size) exercised — the tiled-torch model
        always, the Triton kernel only when it ran.
    schedule_divergence : float
        The headline number of the layer: the max absolute gap between the *same*
        math run with the smallest vs. the largest tile size, measured in fp64. It is
        a property of the schedule alone (identical inputs, identical operator), so it
        isolates the low-bit effect of reordering the summation from any oracle error.
    triton_ran : bool
        Whether the Triton kernel was importable and therefore included. When
        ``False`` the check still stands on the pure-PyTorch model; this flag records
        that the real-kernel rows were skipped rather than passing vacuously.
    """

    shape: tuple[int, int, int]
    rows: list[KernelRow]
    schedule_divergence: float  # max |low-bit gap| between two tile sizes
    triton_ran: bool

    @property
    def all_within(self) -> bool:
        """True if every graded row is within tolerance — the check's overall pass."""
        return all(r.within for r in self.rows)


def run_matmul_check(
    m: int = 128,
    k: int = 256,
    n: int = 96,
    *,
    seed: int = 0,
    tile_ks: tuple[int, ...] = (16, 32, 64),
) -> MatmulCheck:
    """Validate matmul schedules for one ``(M, K, N)`` shape against the oracle.

    Builds one random problem, grades every available schedule against the fp64
    oracle at a shared tolerance, and measures the pure schedule-to-schedule gap.

    Parameters
    ----------
    m, k, n : int
        Problem dimensions: ``A`` is ``(m, k)``, ``B`` is ``(k, n)``. ``k`` is the
        reduction length and therefore the dominant driver of the tolerance.
    seed : int, optional
        Seed for the input RNG. Fixed by default so the whole check is reproducible —
        the same seed yields the same ``max_abs`` and divergence on every run.
    tile_ks : tuple[int, ...], optional
        Reduction tile sizes to exercise. Each becomes one row per backend; the
        smallest and largest also define the divergence probe below.

    Returns
    -------
    MatmulCheck
        The graded rows plus the schedule-divergence figure for this shape.

    Notes
    -----
    Inputs are drawn once and reused for every schedule so that all rows and the
    divergence probe differ *only* in the schedule, never in the data.
    """
    # One fixed problem, shared by every schedule so differences are attributable to
    # the schedule alone. Seeding the generator (not the global RNG) keeps the draw
    # reproducible without touching process-wide state.
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(m, k, generator=g)
    b = torch.randn(k, n, generator=g)
    oracle = reference_matmul(a, b)

    # Tolerance is measured against the magnitude the results actually occupy; the
    # +1.0 floor keeps atol sane when the mean entry is ~0 (near-total cancellation),
    # so a tiny-scale shape does not collapse the tolerance toward zero.
    scale = float(oracle.abs().mean()) + 1.0
    atol = reduction_atol(torch.float32, k, scale=scale)

    rows: list[KernelRow] = []
    # The pure-PyTorch tiled model — always available, carries the lesson on any CPU.
    for bk in tile_ks:
        rep = compare(tiled_matmul(a, b, block_k=bk, accum_dtype=torch.float32), oracle, atol=atol)
        rows.append(KernelRow("torch-tiled", f"block_k={bk}", rep.max_abs, atol, rep.within))

    # The real Triton kernel — only when importable; otherwise its rows are omitted
    # (not failed), and triton_ran records that the real-kernel evidence is absent.
    triton_ran = triton_available()
    if triton_ran:
        for bt in tile_ks:
            out = triton_matmul(a, b, block_m=32, block_n=32, block_k=bt)
            rep = compare(out, oracle, atol=atol)
            rows.append(KernelRow("triton", f"BK={bt}", rep.max_abs, atol, rep.within))

    # Schedule divergence: identical math and inputs, only the tile size differs, so
    # the gap is purely the reordered-summation effect. Compared in fp64 to read the
    # low bits directly rather than against the oracle — this is schedule-vs-schedule,
    # not schedule-vs-truth, and is the concrete number behind "tolerance, not bitwise".
    lo = tiled_matmul(a, b, block_k=min(tile_ks), accum_dtype=torch.float32)
    hi = tiled_matmul(a, b, block_k=max(tile_ks), accum_dtype=torch.float32)
    divergence = float((lo.to(torch.float64) - hi.to(torch.float64)).abs().max())

    return MatmulCheck((m, k, n), rows, divergence, triton_ran)
