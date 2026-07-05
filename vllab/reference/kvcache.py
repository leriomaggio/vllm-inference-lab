"""Reference (non-paged) KV-cache incremental decode.

Autoregressive decoding caches the keys and values of every token seen so far and,
at each step, attends the single new query against the *entire* cache. This module
is the plain, contiguous-tensor version of that cache. Its defining property — the
one the paged implementation (L2) must reproduce — is:

    stepping token-by-token, appending K/V and attending to all cached keys,
    yields exactly the same output as one-shot causal attention over the whole
    sequence.

Keeping this equivalence explicit is what lets a paged cache, which scatters the
same K/V across non-contiguous blocks, be validated: correct paging is invisible
in the output, so any divergence is a bug (e.g. a mis-resolved block).
"""

from __future__ import annotations

import torch

from .attention import softmax_attention


class NonPagedKVCache:
    """A contiguous KV cache: keys/values are stored as growing dense tensors."""

    def __init__(self, *, scale: float | None = None) -> None:
        self._keys: torch.Tensor | None = None
        self._values: torch.Tensor | None = None
        self._scale = scale

    @property
    def length(self) -> int:
        """Number of tokens currently cached."""
        return 0 if self._keys is None else self._keys.shape[-2]

    def append(self, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        """Append one or more tokens' keys/values.

        Parameters
        ----------
        k_t : torch.Tensor
            ``(..., n, D)`` — a length axis is always required (use ``n == 1``
            for a single decode step). Requiring it removes the ambiguity
            between a single token and a chunk when leading dims are present.
        v_t : torch.Tensor
            Matching values, same rank as ``k_t``.
        """
        if k_t.shape[-1] != v_t.shape[-1] or k_t.ndim != v_t.ndim:
            raise ValueError("key/value shapes are inconsistent")
        if self._keys is None:
            self._keys, self._values = k_t, v_t
        else:
            self._keys = torch.cat([self._keys, k_t], dim=-2)
            self._values = torch.cat([self._values, v_t], dim=-2)

    def attend(self, q_t: torch.Tensor) -> torch.Tensor:
        """Attend a query against the full cache (all cached keys are visible).

        Parameters
        ----------
        q_t : torch.Tensor
            ``(..., Sq, D)``.

        Returns
        -------
        torch.Tensor
            ``(..., Sq, Dv)`` attention output (fp64, from the oracle softmax).
        """
        if self._keys is None:
            raise RuntimeError("cache is empty; append before attend")
        return softmax_attention(q_t, self._keys, self._values, causal=False, scale=self._scale)


def incremental_decode(
    q_seq: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Decode a full sequence one token at a time through a ``NonPagedKVCache``.

    Parameters
    ----------
    q_seq, k_seq, v_seq : torch.Tensor
        ``(..., T, D)`` sequences.
    scale : float, optional
        Attention scale (defaults to ``1/sqrt(D)`` inside the oracle).

    Returns
    -------
    torch.Tensor
        ``(..., T, Dv)`` — equal to one-shot causal attention over the sequence.
    """
    t = q_seq.shape[-2]
    cache = NonPagedKVCache(scale=scale)
    outs = []
    for i in range(t):
        cache.append(k_seq[..., i : i + 1, :], v_seq[..., i : i + 1, :])
        outs.append(cache.attend(q_seq[..., i : i + 1, :]))
    return torch.cat(outs, dim=-2)
