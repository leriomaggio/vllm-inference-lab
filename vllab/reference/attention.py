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

    Intuition
    ---------
    Plain softmax needs *all* the scores at once, because the normaliser is a sum
    over the whole row and every term is shifted by the row's maximum for numerical
    stability::

        softmax(s)_j = exp(s_j - max(s)) / Σ_k exp(s_k - max(s))

    Flash attention refuses to hold the whole row. It walks the keys in tiles and
    keeps just three running numbers per query: the max ``m`` seen so far, the
    running sum of weights ``l``, and the running weighted-value output ``o``. The
    catch is that ``m`` can *grow* when a later tile contains a bigger score — and
    every weight already accumulated was computed against the *old*, smaller max.
    The fix is one rescale: multiply the old ``l`` and ``o`` by
    ``alpha = exp(m_old - m_new)``, which re-expresses them as if they'd been
    computed against ``m_new`` all along, then add the new tile.

    Concrete example — scores ``s = [1, 3, 2]`` over two tiles of one key each,
    with the exact answer ``softmax([1,3,2]) = [0.0900, 0.6652, 0.2447]``.

    Tile 1 sees ``s = 1``::

        m = 1,  l = exp(1-1) = 1,  o = 1 · V0

    Tile 2 sees ``s = 3`` — a new, larger max, so rescale before adding::

        m_new = 3,  alpha = exp(1 - 3) = 0.1353
        l = 1·alpha + exp(3-3) = 0.1353 + 1        = 1.1353
        o = (1·V0)·alpha + exp(3-3)·V1 = 0.1353·V0 + 1·V1

    Tile 3 sees ``s = 2`` — smaller than the current max, so ``alpha = 1`` (no
    rescale), just accumulate::

        l = 1.1353 + exp(2-3)   = 1.1353 + 0.3679  = 1.5032
        o = o + exp(2-3)·V2     = o + 0.3679·V2

    Final normalisation ``o / l`` gives weights
    ``[0.1353, 1, 0.3679] / 1.5032 = [0.0900, 0.6652, 0.2447]`` — exactly the plain
    softmax, but we never held more than one score at a time. Note the numerator's
    ``exp(-m)`` factor is common to every term and cancels against the same factor
    in ``l``, which is why the deferred divide recovers the true softmax regardless
    of what the running max ended up being.

    Why it matters in practice: the KV cache for a long context (or a paged /
    multi-request server) is far too large to score in one matmul. Tiling lets the
    kernel keep only a ``block_size``-wide slice of K/V in fast memory (SRAM /
    registers) at a time, which is the whole reason flash-attention and paged
    attention are feasible on real hardware.

    Parameters
    ----------
    block_size : int, optional
        Number of keys processed per step (the KV tile).
    accum_dtype : torch.dtype, optional
        Precision of the running statistics and output accumulator.

    Returns
    -------
    torch.Tensor
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
    lead = q.shape[:-2]  # batch/head dims that broadcast through every matmul
    # Smallest representable value of the accumulator dtype: seeds the running max
    # (so the first real score always wins) and doubles as the causal-mask fill,
    # since exp(neg - m_new) underflows to 0 and contributes nothing.
    neg = torch.finfo(accum_dtype).min

    # Per-query running statistics, one row per query, carried across all blocks:
    #   m   — running row-max of the scores seen so far (for numerical stability)
    #   ell — running normaliser: Σ exp(score - m) over keys seen so far
    #   o   — running *unnormalised* output: Σ exp(score - m) · V, divided by ell
    #         only at the very end. Keeping it unnormalised lets us rescale it
    #         cheaply whenever m changes.
    m = torch.full((*lead, sq, 1), neg, dtype=accum_dtype, device=q.device)
    ell = torch.zeros((*lead, sq, 1), dtype=accum_dtype, device=q.device)
    o = torch.zeros((*lead, sq, dv), dtype=accum_dtype, device=q.device)

    # Stream over the key/value sequence one KV tile at a time. Only one block of
    # K/V is resident at a time — this is the O(block_size) working-memory win.
    for j0 in range(0, sk, block_size):
        j1 = min(j0 + block_size, sk)
        kb = k[..., j0:j1, :].to(accum_dtype)  # this tile's keys   (..., jb, D)
        vb = v[..., j0:j1, :].to(accum_dtype)  # this tile's values (..., jb, Dv)

        # Scaled scores of every query against just this tile of keys.
        s = torch.matmul(qd, kb.transpose(-1, -2)) * scale  # (..., Sq, jb)
        if causal:
            # Query row i sits at absolute position (sk - sq + i); mask out any
            # key in this tile that lies strictly after it. Masked entries get
            # `neg`, which becomes a ~0 probability weight below.
            q_pos = torch.arange(sk - sq, sk, device=q.device).unsqueeze(1)
            k_pos = torch.arange(j0, j1, device=q.device).unsqueeze(0)
            s = s.masked_fill(k_pos > q_pos, neg)

        # Fold this tile into the running softmax. The new max may exceed the old
        # one, so both the prior stats and this tile must share a common max.
        block_max = s.max(dim=-1, keepdim=True).values  # (..., Sq, 1)
        m_new = torch.maximum(m, block_max)  # updated running max
        # alpha corrects everything accumulated under the *old* max m to the new
        # max m_new: multiplying by exp(m_old - m_new) is exactly the shift needed
        # so old and new contributions are expressed on the same exponent scale.
        alpha = torch.exp(m - m_new)  # rescale prior stats (<= 1)
        p = torch.exp(s - m_new)  # this tile's softmax weights, common max (..., Sq, jb)

        # Rescale the carried normaliser/output, then add this tile's share.
        ell = ell * alpha + p.sum(dim=-1, keepdim=True)  # running Σ of weights
        o = o * alpha + torch.matmul(p, vb)  # running Σ of weights · V
        m = m_new  # advance the running max for the next tile

    # Deferred normalisation: dividing the accumulated (weight · V) sum by the
    # accumulated weight sum yields exactly softmax(scores) · V — the missing
    # exp(-m) factors cancel between numerator and denominator.
    return o / ell
