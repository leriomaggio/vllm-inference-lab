# L0 — Reference oracles

The lab validates fast, hardware-shaped implementations against slow, obviously
correct ones. This layer defines those **oracles** and the tolerance vocabulary
every later layer reuses.

## What an oracle is (and why fp64)

An oracle is an implementation whose correctness is self-evident, used as the
ground truth in a differential test. Ours evaluate the operator in **fp64** with a
single, unambiguous reduction order.

Floating-point addition is *not associative*: `(a + b) + c` and `a + (b + c)`
can differ in the low bits. So "the same" matmul, evaluated with a different
summation order, gives a *different* bit pattern. The forward error of a
length-`K` reduction in a precision with unit roundoff `u` grows like

```
error ~ sqrt(K) * u          (well-conditioned; up to K*u worst case)
```

Evaluating the oracle in fp64 (`u ≈ 2.2e-16`) makes *its own* error negligible
against an fp32 (`u ≈ 1.2e-7`) or fp16 (`u ≈ 9.8e-4`) candidate, so the measured
gap is attributable to the candidate's schedule, not the oracle.

## Modules

| File | Oracle | Companion schedule |
|------|--------|--------------------|
| `vllab/reference/matmul.py` | `reference_matmul` (fp64) | `tiled_matmul` — explicit block-wise K reduction |
| `vllab/reference/attention.py` | `softmax_attention` (fp64, full score matrix) | `online_softmax_attention` — flash-style streaming |
| `vllab/reference/kvcache.py` | one-shot causal attention | `incremental_decode` via `NonPagedKVCache` |

### The two numerical levers (`tiled_matmul`)

`tiled_matmul` is a faithful *model* of how a hardware kernel evaluates a matmul,
exposing the two knobs that a real kernel also has:

- **schedule / tile size** (`block_k`) — changes the *order* of the summation, so
  results move in the low bits though the math is unchanged;
- **accumulator width** (`accum_dtype`) — a narrow accumulator (fp16) loses
  precision a wide one (fp32) keeps, and the loss grows with `K`.

These are the same two levers the L1 Triton kernel exposes; modelling them in
PyTorch means the lesson runs on any CPU, with no GPU or Triton install.

### Online softmax (`online_softmax_attention`)

Computes attention **without materialising the full score row**: it streams over
key/value blocks, maintaining a running row-max `m` and normaliser `l`, and
rescales the partial output by `exp(m_old - m_new)` before adding each block. This
is the memory trick behind flash-attention and paged KV attention. It is
mathematically identical to the full softmax and must agree with the oracle within
tolerance.

### KV-cache equivalence (`incremental_decode`)

Autoregressive decoding caches K/V for every past token and attends the new query
against the whole cache. The defining invariant — which the paged cache in L2 must
reproduce — is:

> stepping token-by-token equals one-shot causal attention over the full sequence.

## Tolerances (`vllab/numerics.py`)

`reduction_atol(dtype, K, scale, safety)` returns `safety * scale * sqrt(K) * u`.
Tests derive their `atol` from it rather than hardcoding a constant — the tolerance
is *justified* by precision and reduction length. `compare()` returns a
`DiffReport` (`max_abs`, `mean_abs`, `max_rel`, `within`) computed in fp64.

## How to validate

```bash
pytest tests/test_reference.py -q
```

The tests assert:

- every tile size lands within tolerance of the fp64 oracle (matmul + attention);
- two different tile sizes are **not** bitwise equal (schedule dependence) yet
  both stay within tolerance;
- an fp16 accumulator's error is >10× an fp32 accumulator's (accumulator width
  dominates);
- incremental decode equals one-shot causal attention.

## Takeaway

Correctness across schedules is **tolerance-defined, not bitwise**. Bitwise
reproducibility would require pinning the schedule (tile sizes, accumulator dtype,
reduction order) — which the later layers make concrete all the way up to a real
inference engine.
