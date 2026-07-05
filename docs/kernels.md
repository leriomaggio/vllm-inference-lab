# L1 — The matmul spine as a real kernel

Matmul is the operator almost every model spends its time in: dense projections
and feed-forward layers are matmuls; a convolution becomes a matmul after
`im2col`; attention is a chain of matmuls around a softmax. So the kernel worth
understanding first is the tiled matmul — and the discipline worth building first
is validating one.

## One definition, many schedules

`C = A @ B` has a single mathematical definition and many valid *schedules*: a
triple loop, an `einsum`, a BLAS micro-kernel, a Triton `tl.dot` kernel. They
compute the same thing, but — because floating-point addition is not associative —
they generally produce **different low-order bits**. Correctness is therefore
defined by a tolerance, not by bit-equality (see
[reference-oracles.md](reference-oracles.md) for the error model).

## The kernel (`vllab/kernels/matmul_triton.py`)

The Triton kernel is the canonical tiled `tl.dot` matmul:

- a **2-D grid** of program instances, one per `BM × BN` output tile;
- each program walks `K` in steps of `BK`, loading a `BM × BK` block of `A` and a
  `BK × BN` block of `B`;
- it multiplies with `tl.dot` into a **private fp32 accumulator** — the
  accumulation dtype is explicit and is *wider than fp16/bf16 operands* on purpose;
- **edge masks** (`other=0.0`) zero-pad ragged tiles when `M/N/K` are not
  multiples of the tile, so ragged shapes are a mandatory test case;
- **strides are passed in**, not assumed — so transposed or sliced inputs work
  unchanged (a pointer to any element is `base + row*row_stride + col*col_stride`).

Two config choices change the numerics:

- **tile sizes** (`BM/BN/BK`) change the summation order → low-bit differences;
- the **accumulator dtype** changes how much precision the reduction retains.

Neither is visible by inspecting the output tensor's `dtype`. That is the crux:
**a compute-mode change (tile config, accumulator width, a fast-math flag, tf32)
cannot be detected by `dtype` inspection — only by measuring outputs.**

### Running it without a GPU (and without Triton on macOS)

Triton publishes no macOS/Apple-Silicon wheels, so on macOS the module imports
without Triton and `triton_available()` returns `False`. Two consequences, both
handled:

1. The module still imports and the CLI/tests still run — the real kernel is
   simply skipped where it cannot load.
2. The **numerical lesson still runs on CPU**, because the pure-PyTorch
   `tiled_matmul` model (L0) reproduces the same tile-wise reduction with the same
   two levers.

Where Triton *is* importable (Linux, or a CUDA host), the kernel runs in
**interpreter mode** (`TRITON_INTERPRET=1`, set in the module before import) so it
executes on CPU with no GPU required, or natively on GPU otherwise.

## Validation (`vllab/kernels/harness.py`)

`run_matmul_check()` compares every available schedule against the fp64 oracle and
quantifies the divergence caused purely by changing the tile size.

```bash
vllab kernel-check
pytest tests/test_kernels.py -q
```

### Measured (M, K, N) = (128, 256, 96), seed 0, on macOS CPU

| backend      | config       |   max_abs |      atol | verdict |
|--------------|--------------|-----------|-----------|---------|
| torch-tiled  | block_k=16   | 1.138e-05 | 2.111e-04 | within  |
| torch-tiled  | block_k=32   | 1.062e-05 | 2.111e-04 | within  |
| torch-tiled  | block_k=64   | 1.252e-05 | 2.111e-04 | within  |

*(A `triton` row is added on machines where Triton is importable.)*

**Schedule divergence (tile-size low-bit gap): `1.907e-05`.**

Read that together: every schedule sits ~20× inside the `2.1e-4` tolerance, yet
two tile sizes differ from each other by `1.9e-5` — a real, non-zero gap. Same
math, different order, different bits, all correct. Bitwise reproducibility would
require pinning the tile configuration.

## What the tests assert

- `test_matmul_harness_all_within_tolerance` — every schedule within oracle tolerance;
- `test_schedule_divergence_is_nonzero_but_small` — tile size reorders the sum
  (gap > 0) but stays below tolerance;
- `test_triton_ran_flag_matches_availability` — the harness reports honestly
  whether the real kernel ran;
- `test_triton_kernel_matches_oracle` — the Triton kernel matches the oracle on a
  deliberately ragged shape; **skipped where Triton is unavailable** (e.g. macOS).

## Takeaway

The operator that dominates model runtime is the operator whose validation you
must get right, and getting it right means a *justified tolerance*, ragged-tile
coverage, and awareness that the dangerous changes (accumulator width, compute
mode) are invisible to `dtype` and show up only in the numbers.
