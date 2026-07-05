"""L1 validation harness: compare kernel schedules against the fp64 oracle.

Runs on any CPU. It always validates the pure-PyTorch tiled-reduction *model*
(:func:`vllab.reference.matmul.tiled_matmul`) and, when Triton is importable, also
the real :func:`vllab.kernels.matmul_triton.triton_matmul` kernel — both against
:func:`vllab.reference.matmul.reference_matmul`. It also quantifies the low-bit
divergence produced purely by changing the tile size (the schedule), which is the
reason cross-backend correctness is tolerance-defined rather than bitwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllab.numerics import compare, reduction_atol
from vllab.reference.matmul import reference_matmul, tiled_matmul

from .matmul_triton import triton_available, triton_matmul


@dataclass(frozen=True)
class KernelRow:
    """One validated schedule."""

    backend: str
    config: str
    max_abs: float
    atol: float
    within: bool


@dataclass(frozen=True)
class MatmulCheck:
    """Result of a matmul kernel check across schedules."""

    shape: tuple[int, int, int]
    rows: list[KernelRow]
    schedule_divergence: float  # max |low-bit gap| between two tile sizes
    triton_ran: bool

    @property
    def all_within(self) -> bool:
        return all(r.within for r in self.rows)


def run_matmul_check(
    m: int = 128,
    k: int = 256,
    n: int = 96,
    *,
    seed: int = 0,
    tile_ks: tuple[int, ...] = (16, 32, 64),
) -> MatmulCheck:
    """Validate matmul schedules for one ``(M, K, N)`` shape against the oracle."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(m, k, generator=g)
    b = torch.randn(k, n, generator=g)
    oracle = reference_matmul(a, b)
    scale = float(oracle.abs().mean()) + 1.0
    atol = reduction_atol(torch.float32, k, scale=scale)

    rows: list[KernelRow] = []
    for bk in tile_ks:
        rep = compare(tiled_matmul(a, b, block_k=bk, accum_dtype=torch.float32), oracle, atol=atol)
        rows.append(KernelRow("torch-tiled", f"block_k={bk}", rep.max_abs, atol, rep.within))

    triton_ran = triton_available()
    if triton_ran:
        for bt in tile_ks:
            out = triton_matmul(a, b, block_m=32, block_n=32, block_k=bt)
            rep = compare(out, oracle, atol=atol)
            rows.append(KernelRow("triton", f"BK={bt}", rep.max_abs, atol, rep.within))

    # Schedule divergence: same math, two different tile sizes.
    lo = tiled_matmul(a, b, block_k=min(tile_ks), accum_dtype=torch.float32)
    hi = tiled_matmul(a, b, block_k=max(tile_ks), accum_dtype=torch.float32)
    divergence = float((lo.to(torch.float64) - hi.to(torch.float64)).abs().max())

    return MatmulCheck((m, k, n), rows, divergence, triton_ran)
