"""Tiled Triton matmul kernel — the "matmul spine" as a real hardware kernel.

Triton ships wheels only for Linux (and CUDA hosts); platforms without them —
macOS/Apple-Silicon among them — cannot import it. The module tolerates this: when
the import fails, :func:`triton_available` returns ``False``, the kernel skips
cleanly, and the L1 *numerical* lesson is carried instead by the pure-PyTorch
:func:`vllab.reference.matmul.tiled_matmul` model, which reproduces the same
tile-wise reduction on any CPU. Where Triton *is* importable, the same kernel runs
in **interpreter mode** (``TRITON_INTERPRET=1``, set below before the import) so it
executes on CPU with no GPU required.

The kernel is the known-good tiled ``tl.dot`` matmul: a 2-D grid of programs, one
per ``BM x BN`` output tile, each walking K in steps of ``BK`` and accumulating
into a private **fp32** accumulator (explicit accumulator width is the point),
with edge masks for ragged tiles.
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_INTERPRET", "1")  # CPU interpretation; must precede import

import torch

try:  # Triton is optional and absent on macOS.
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - exercised only where triton is missing
    _HAS_TRITON = False


def triton_available() -> bool:
    """True if Triton could be imported (interpreter mode included)."""
    return _HAS_TRITON


if _HAS_TRITON:

    @triton.jit
    def _matmul_kernel(  # noqa: PLR0913 - explicit pointer/stride/tile args
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        sam,
        sak,
        sbk,
        sbn,
        scm,
        scn,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(
                a_ptr + rm[:, None] * sam + (k0 + rk)[None, :] * sak,
                mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr + (k0 + rk)[:, None] * sbk + rn[None, :] * sbn,
                mask=((k0 + rk)[:, None] < K) & (rn[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b)
        tl.store(
            c_ptr + rm[:, None] * scm + rn[None, :] * scn,
            acc,
            mask=(rm[:, None] < M) & (rn[None, :] < N),
        )


def triton_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_m: int = 32,
    block_n: int = 32,
    block_k: int = 32,
) -> torch.Tensor:
    """Compute ``a @ b`` with the tiled Triton kernel.

    Parameters
    ----------
    a : torch.Tensor
        ``(M, K)`` fp32 tensor (strides are passed through, so transposed or
        sliced inputs work unchanged).
    b : torch.Tensor
        ``(K, N)`` fp32 tensor. Must be on the same device as ``a`` — unlike the
        device-independent reference oracle, this is a hardware kernel and all
        operands are co-located.
    block_m, block_n, block_k : int, optional
        Output-tile and K-step sizes. Changing them changes the summation
        order — outputs then differ in the low bits.

    Returns
    -------
    torch.Tensor
        ``(M, N)`` fp32 result, allocated on the inputs' device.

    Raises
    ------
    ValueError
        If the inputs are not 2-D, their inner dimensions disagree, or they live
        on different devices.
    """
    if not _HAS_TRITON:
        raise RuntimeError("Triton is not available in this environment")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("triton_matmul expects 2-D inputs")
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"inner dimensions do not match: {(m, k)} @ {(k2, n)}")
    if a.device != b.device:
        raise ValueError(f"inputs must be on the same device: {a.device} vs {b.device}")

    a = a.to(torch.float32).contiguous()
    b = b.to(torch.float32).contiguous()
    c = torch.empty((m, n), dtype=torch.float32, device=a.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BM=block_m,
        BN=block_n,
        BK=block_k,
    )
    return c
