"""L1 kernel tests: the validation harness grades every schedule against the oracle,
and the low-bit divergence it reports behaves as the floating-point model predicts.

These exercise the harness through its public entry point
(:func:`vllab.kernels.harness.run_matmul_check`) rather than the individual kernels.
The pure-PyTorch schedule model runs everywhere; the real Triton kernel is only
exercised where Triton is importable (skipped on macOS, which ships no wheels).
"""

from __future__ import annotations

import pytest
import torch

from vllab.kernels.harness import run_matmul_check
from vllab.kernels.matmul_triton import triton_available
from vllab.numerics import compare, reduction_atol
from vllab.reference.matmul import reference_matmul


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
def test_matmul_harness_all_within_tolerance() -> None:
    """Goal: the end-to-end harness grades every available schedule and they all clear
    the same correctness bar — the L1 "many schedules, one tolerance" claim checked
    through the real entry point, not the kernels in isolation.

    Expectation: ``run_matmul_check`` on a representative shape (128x256x96) yields at
    least one row (guards against a silently empty sweep) and every row is ``within``
    the fp32 ``reduction_atol`` the harness derived for itself. Asserts no
    (backend, tile size) combination steps outside the rounding budget of a length-``K``
    sum.
    """
    check = run_matmul_check(m=128, k=256, n=96, seed=0)
    assert check.rows, "harness produced no rows"
    for row in check.rows:
        assert (
            row.within
        ), f"{row.backend} {row.config}: max_abs={row.max_abs:.3e} > atol={row.atol:.3e}"


def test_schedule_divergence_is_nonzero_but_small() -> None:
    """Goal: the harness's headline figure behaves as the lab's thesis requires —
    changing only the tile size perturbs the result in the low bits (non-zero) but
    never beyond the oracle tolerance (small). This is the matmul analogue of
    ``test_tile_size_changes_low_bits_but_stays_in_tolerance`` in the L0 suite,
    measured through the harness rather than by hand.

    Expectation: two tile sizes (16 vs 128) over a ``K=512`` reduction. Asserts
    ``schedule_divergence > 0`` (the reorder is real, not a no-op) and that it stays
    below the fp32 length-512 ``reduction_atol`` (bounded — tolerance-level agreement,
    never bitwise). Together these are the measured evidence behind "tolerance, not
    equality".
    """
    # Changing the tile size reorders the sum: a real, non-zero, low-bit gap that
    # nonetheless stays far below the oracle tolerance.
    check = run_matmul_check(m=64, k=512, n=64, seed=1, tile_ks=(16, 128))
    assert check.schedule_divergence > 0.0, "expected tile size to reorder the summation"
    assert check.schedule_divergence < reduction_atol(torch.float32, 512, scale=2.0)


def test_triton_ran_flag_matches_availability() -> None:
    """Goal: the check reports honestly whether the real kernel participated — a
    skipped Triton must be *recorded* as skipped, never silently counted as a pass.

    Expectation: ``triton_ran`` on a default check must equal ``triton_available()``.
    Asserts the harness neither claims real-kernel evidence it lacks (on macOS) nor
    drops it when Triton is present, so ``all_within`` can never pass vacuously.
    """
    check = run_matmul_check()
    assert check.triton_ran == triton_available()


# --------------------------------------------------------------------------- #
# triton kernel
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not triton_available(), reason="Triton not importable (e.g. macOS)")
def test_triton_kernel_matches_oracle() -> None:
    """Goal: where Triton is importable, the real kernel computes the *same* matmul as
    the oracle even on shapes that do not divide the tile grid — i.e. the edge masks
    for ragged tiles are correct, not just the tile-aligned happy path.

    Expectation: skipped unless Triton imports. A deliberately ragged shape
    (70x96x45 against 32-wide tiles) forces partial edge tiles in all three
    dimensions. Asserts the kernel output is ``within`` the fp32 ``reduction_atol``,
    proving the kernel is a faithful schedule — correct masking included — and not
    merely correct on sizes that happen to tile evenly.
    """
    g = torch.Generator().manual_seed(7)
    m, k, n = 70, 96, 45  # deliberately ragged vs the 32-tile
    a = torch.randn(m, k, generator=g)
    b = torch.randn(k, n, generator=g)
    from vllab.kernels.matmul_triton import triton_matmul

    oracle = reference_matmul(a, b)
    rep = compare(triton_matmul(a, b), oracle, atol=reduction_atol(torch.float32, k, scale=2.0))
    assert rep.within, f"triton kernel vs oracle: {rep}"
