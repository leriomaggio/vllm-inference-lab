"""L1 kernel tests.

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


def test_matmul_harness_all_within_tolerance() -> None:
    check = run_matmul_check(m=128, k=256, n=96, seed=0)
    assert check.rows, "harness produced no rows"
    for row in check.rows:
        assert row.within, (
            f"{row.backend} {row.config}: max_abs={row.max_abs:.3e} > atol={row.atol:.3e}"
        )


def test_schedule_divergence_is_nonzero_but_small() -> None:
    # Changing the tile size reorders the sum: a real, non-zero, low-bit gap that
    # nonetheless stays far below the oracle tolerance.
    check = run_matmul_check(m=64, k=512, n=64, seed=1, tile_ks=(16, 128))
    assert check.schedule_divergence > 0.0, "expected tile size to reorder the summation"
    assert check.schedule_divergence < reduction_atol(torch.float32, 512, scale=2.0)


def test_triton_ran_flag_matches_availability() -> None:
    check = run_matmul_check()
    assert check.triton_ran == triton_available()


@pytest.mark.skipif(not triton_available(), reason="Triton not importable (e.g. macOS)")
def test_triton_kernel_matches_oracle() -> None:
    g = torch.Generator().manual_seed(7)
    m, k, n = 70, 96, 45  # deliberately ragged vs the 32-tile
    a = torch.randn(m, k, generator=g)
    b = torch.randn(k, n, generator=g)
    from vllab.kernels.matmul_triton import triton_matmul

    oracle = reference_matmul(a, b)
    rep = compare(triton_matmul(a, b), oracle, atol=reduction_atol(torch.float32, k, scale=2.0))
    assert rep.within, f"triton kernel vs oracle: {rep}"
