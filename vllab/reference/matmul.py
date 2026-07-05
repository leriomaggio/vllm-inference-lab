"""Reference matmul oracle.

The oracle for ``C = A @ B`` is simply the product evaluated in wide precision
(fp64) with a single, unambiguous reduction. It is deliberately the plainest
possible implementation: its job is to be *obviously correct*, not fast, so that
every faster schedule (a tiled reduction, a Triton kernel, a fused attention) can
be measured against it within a justified tolerance.

Why fp64 is a fair oracle: the rounding error of a length-``K`` dot product in a
working precision with unit roundoff ``u`` grows like ``~sqrt(K) * u`` (for a
well-conditioned sum). Evaluating the same reduction in fp64 (``u ~ 1.1e-16``)
makes the oracle's own error negligible relative to an fp32/fp16 candidate, so the
measured gap is dominated by the candidate's schedule, not the oracle's.
"""

from __future__ import annotations

import torch

from vllab.numerics import to_cpu_fp64


def reference_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute ``a @ b`` in fp64 on the CPU and return the fp64 result.

    Args:
        a: Tensor of shape ``(..., M, K)``.
        b: Tensor of shape ``(..., K, N)`` broadcastable against ``a``.

    Returns:
        The product in ``torch.float64`` on the CPU. Callers cast to the precision
        they are validating and compare with a tolerance derived from that precision.

    The reduction is always evaluated on the CPU, regardless of where the inputs
    live. This keeps the oracle a single, device-independent source of truth — a GPU
    reduction is order-nondeterministic, and Apple's MPS backend has no fp64 at all —
    and it lets ``a`` and ``b`` originate on different devices. The normalisation
    (CPU move *before* the fp64 cast, since that cast is what MPS rejects) is shared
    with the comparison path via :func:`vllab.numerics.to_cpu_fp64`, so the lab's
    "truth form" is defined in exactly one place.
    """
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"inner dimensions do not match: {tuple(a.shape)} @ {tuple(b.shape)}")
    return torch.matmul(to_cpu_fp64(a), to_cpu_fp64(b))


def tiled_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_k: int = 32,
    accum_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute ``a @ b`` as an explicit, tiled reduction over the K dimension.

    This is a faithful *model* of how a hardware kernel (Triton ``tl.dot``, a BLAS
    micro-kernel) actually evaluates a matmul: the K reduction is split into blocks
    of ``block_k`` and accumulated into an accumulator of dtype ``accum_dtype``.

    It exists so the two numerical levers a real kernel exposes are directly
    controllable and testable on any CPU, without a GPU or a Triton install:

    * **schedule / tile size** — changing ``block_k`` changes the *order* of the
      floating-point summation, so results differ in the low bits even though the
      mathematical definition is unchanged;
    * **accumulator width** — a narrow ``accum_dtype`` (e.g. fp16) loses precision
      that a wide one (fp32) keeps, and the error grows with ``K``.

    Args:
        a: Tensor of shape ``(M, K)``.
        b: Tensor of shape ``(K, N)``.
        block_k: Reduction tile size along K.
        accum_dtype: Dtype of the running accumulator.

    Returns:
        The product in ``accum_dtype``. The *inputs* are read in their own dtype;
        each block product is computed and then added into the accumulator, which
        is the behaviour that makes accumulator width observable.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("tiled_matmul expects 2-D inputs")
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"inner dimensions do not match: {(m, k)} @ {(k2, n)}")
    if block_k <= 0:
        raise ValueError("block_k must be positive")

    acc = torch.zeros((m, n), dtype=accum_dtype)
    for k0 in range(0, k, block_k):
        k1 = min(k0 + block_k, k)
        # Block product is formed in the accumulator dtype, then reduced into acc.
        partial = torch.matmul(a[:, k0:k1].to(accum_dtype), b[k0:k1, :].to(accum_dtype))
        acc = acc + partial
    return acc
