"""Reference attention oracles: plain (stable) softmax attention and the tiled
*online-softmax* schedule that flash-attention and paged kernels actually use.

Both compute scaled dot-product attention

    out = softmax( (Q Kᵀ) * scale  [+ causal mask] ) V

but by different schedules. ``softmax_attention`` materialises the full score
matrix and is the trusted oracle (evaluated in fp64).
``online_softmax_attention`` never materialises the full row: it streams over
blocks of keys/values, maintaining a running max and normaliser, which is the
memory trick that makes long-context and paged KV attention feasible. The two
must agree within a tolerance set by the working precision.

Shape convention: ``Q`` is ``(..., Sq, D)``, ``K``/``V`` are ``(..., Sk, D)``, and
any leading dims (batch, heads) broadcast through the matmuls. Causal masking
aligns the queries to the *end* of the key sequence, so query row ``i`` sits at
absolute position ``Sk - Sq + i`` — this makes both the full (``Sq == Sk``) and the
incremental-decode (``Sq == 1``) cases correct.
"""

from __future__ import annotations

import math

import torch


def _causal_keep_mask(sq: int, sk: int, device: torch.device) -> torch.Tensor:
    """Boolean ``(Sq, Sk)`` mask, True where a query may attend to a key."""
    q_pos = torch.arange(sk - sq, sk, device=device).unsqueeze(1)  # (Sq, 1)
    k_pos = torch.arange(sk, device=device).unsqueeze(0)  # (1, Sk)
    return k_pos <= q_pos


def softmax_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Oracle attention: full score matrix, numerically stable softmax, in fp64.

    Returns the output in ``torch.float64``; callers cast to the precision under
    test and compare with a precision-appropriate tolerance.
    """
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    qd, kd, vd = q.to(torch.float64), k.to(torch.float64), v.to(torch.float64)

    scores = torch.matmul(qd, kd.transpose(-1, -2)) * scale  # (..., Sq, Sk)
    if causal:
        keep = _causal_keep_mask(q.shape[-2], k.shape[-2], scores.device)
        scores = scores.masked_fill(~keep, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, vd)


def online_softmax_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int = 16,
    causal: bool = False,
    scale: float | None = None,
    accum_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Flash-style attention: stream over key/value blocks with a running softmax.

    For each block of keys it updates a running row-max ``m`` and normaliser ``l``
    and rescales the partial output ``o`` by ``exp(m_old - m_new)`` before adding
    the block's contribution — mathematically identical to the full softmax, but
    with ``O(block_size)`` working memory instead of ``O(Sk)``.

    Args:
        block_size: Number of keys processed per step (the KV tile).
        accum_dtype: Precision of the running statistics and output accumulator.

    Returns:
        Output in ``accum_dtype``. Changing ``block_size`` changes the summation
        order (results differ in the low bits); narrowing ``accum_dtype`` widens
        the gap to the oracle.
    """
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    sq, sk = q.shape[-2], k.shape[-2]
    dv = v.shape[-1]
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    qd = q.to(accum_dtype)
    lead = q.shape[:-2]
    neg = torch.finfo(accum_dtype).min

    m = torch.full((*lead, sq, 1), neg, dtype=accum_dtype, device=q.device)
    ell = torch.zeros((*lead, sq, 1), dtype=accum_dtype, device=q.device)
    o = torch.zeros((*lead, sq, dv), dtype=accum_dtype, device=q.device)

    for j0 in range(0, sk, block_size):
        j1 = min(j0 + block_size, sk)
        kb = k[..., j0:j1, :].to(accum_dtype)
        vb = v[..., j0:j1, :].to(accum_dtype)

        s = torch.matmul(qd, kb.transpose(-1, -2)) * scale  # (..., Sq, jb)
        if causal:
            q_pos = torch.arange(sk - sq, sk, device=q.device).unsqueeze(1)
            k_pos = torch.arange(j0, j1, device=q.device).unsqueeze(0)
            s = s.masked_fill(k_pos > q_pos, neg)

        block_max = s.max(dim=-1, keepdim=True).values  # (..., Sq, 1)
        m_new = torch.maximum(m, block_max)
        alpha = torch.exp(m - m_new)  # rescale prior stats
        p = torch.exp(s - m_new)  # (..., Sq, jb)

        ell = ell * alpha + p.sum(dim=-1, keepdim=True)
        o = o * alpha + torch.matmul(p, vb)
        m = m_new

    return o / ell
