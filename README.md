# vllm-inference-lab

A hands-on lab that climbs the inference stack **from a hand-written matmul up to a
real vLLM engine**, validating every layer against a reference oracle with an
explicit *tolerance* discipline rather than assuming bitwise equality.

The organising idea:

> **One operator has a single mathematical definition but many valid execution
> schedules.** A loop, an `einsum`, an `im2col` GEMM, a tiled Triton kernel, and a
> paged-attention decode all compute "the same thing" — yet floating-point
> arithmetic is not associative, so they agree only *up to a tolerance* that must
> be justified by dtype, reduction length, and accumulator width. Verifying that
> is the job.

## The ladder

| Layer | What it is | Runs on |
|-------|------------|---------|
| **L0** | Reference oracles: matmul, attention (online softmax), KV-cache decode | CPU (any) |
| **L1** | Tiled **Triton** matmul + attention kernels, validated in interpreter mode | CPU (`TRITON_INTERPRET=1`) |
| **L2** | From-scratch **PagedAttention** (pages + block table) with fault injection | CPU (any) |
| **L3** | Real **vLLM** engine, differential vs a HuggingFace reference, fp32-vs-fp16 | CPU (macOS build) |
| **L4** | Multi-backend correctness/latency **matrix** (HF, vLLM-CPU, optional Metal) | CPU + optional Apple GPU |

Layers **L0–L2 are self-contained** and need no vLLM install. L3/L4 add the engine.

## Quick start

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[kernels,hf,dev]"

# L1 kernels (no GPU required)
TRITON_INTERPRET=1 pytest tests/test_kernels.py

# CLI
vllab --help
```

Building the vLLM CPU backend on macOS and the optional Apple-GPU backend are
covered in [docs/setup.md](docs/setup.md).

## CLI

```
vllab kernel-check   # L1: Triton kernels vs reference oracles (interpreter mode)
vllab paged-demo     # L2: paged == reference; block-table fault injection
vllab bench          # L3: single-engine latency / metrics
vllab validate       # L3: vLLM vs HuggingFace differential + precision axis
vllab matrix         # L4: multi-backend correctness/latency matrix -> artifact
vllab report         # render a markdown report from an artifact
```

## Code organisation

The package is `vllab`, one sub-package per layer. Every module has a single
responsibility; the fast/hardware-shaped code always has a slow, obvious oracle to
be measured against.

```
vllab/
├── numerics.py              cross-cutting: tolerances and diff reporting
├── reference/   (L0)        the trusted oracles
│   ├── matmul.py            fp64 matmul oracle + tiled-reduction model (tile size, accumulator dtype)
│   ├── attention.py         fp64 softmax-attention oracle + flash-style online-softmax schedule
│   └── kvcache.py           non-paged incremental KV-cache decode (the paging correctness target)
├── kernels/     (L1)        the matmul spine as a real kernel
│   ├── matmul_triton.py     tiled Triton tl.dot kernel; lazy-imported, interpreter mode, skips w/o Triton
│   └── harness.py           validate PyTorch model + Triton kernel vs oracle; quantify schedule divergence
├── paged/       (L2)        PagedAttention from scratch
│   ├── block_table.py       logical→physical page mapping over a free pool (+ fault injection)
│   └── paged_attention.py   paged KV pool, gather/attend, paged_decode, silent-corruption demo
├── engine/      (L3)        the real inference engine + its reference
│   ├── types.py             GenResult: one prompt's generation (ids, text, per-step logprobs)
│   ├── hf_reference.py      HuggingFace transformers oracle (greedy + next-token logprobs)
│   ├── vllm_runner.py       vLLM offline wrapper, same surface as the HF reference; lazy-imported
│   └── differential.py      compare two engines' generations (prefix match, logprob gap)
└── cli.py                   Typer entry point (vllab): one subcommand per layer
```

| Module | Scope / reason to be |
|--------|----------------------|
| `numerics.py` | One home for the tolerance discipline: `reduction_atol` (the `sqrt(K)·u` model) and `compare`/`DiffReport`. Every layer derives tolerances here instead of hardcoding magic numbers. |
| `reference/*` | The oracles. Slow, fp64, obviously correct — the definition of "right" that all faster schedules are checked against. |
| `kernels/*` | The GEMM as a hardware kernel, plus the harness that proves same-math-different-schedule diverges only in the low bits. The lesson runs on CPU even where Triton has no wheels. |
| `paged/*` | The KV-cache data structure that makes LLM serving efficient, and the demonstration that a mis-mapped page corrupts output *silently* — the class of bug a latency benchmark cannot see. |
| `engine/*` | Where the real engine (vLLM) meets a trusted reference (HuggingFace). Both wrappers share one surface so their outputs are directly comparable; `differential.py` is pure and testable without either installed. |
| `cli.py` | A thin, dependency-light command layer; heavy imports are deferred into each command so `vllab --help` and the light layers work without vLLM/Triton. |

**Design rule throughout:** heavy/optional dependencies (Triton, vLLM,
transformers) are imported lazily and guarded, so any module imports and its tests
run — skipping cleanly — regardless of what is installed on a given machine.

**Docstring style:** all docstrings follow the
[NumPy documentation style](https://numpydoc.readthedocs.io/en/latest/format.html)
(`Parameters`/`Returns`/`Attributes`/`Notes` sections with underlined headings),
used consistently across the package rather than the Google style. This is a
deliberate choice: the numerical arguments carry prose-heavy, multi-paragraph
rationale (tolerance derivations, the `sqrt(K)·u` error model, device
normalisation), and NumPy's section layout — with its dedicated `Notes` block —
reads better for that kind of explanatory documentation and matches the
scientific-Python (NumPy/SciPy) ecosystem this lab builds on.

## Documentation

Each layer ships a companion note explaining the concept, how to validate it, and
the actual measured numbers:

- [docs/reference-oracles.md](docs/reference-oracles.md) — L0: oracles, the fp64
  rationale, and the tolerance vocabulary.
- [docs/kernels.md](docs/kernels.md) — L1: the matmul spine, schedule vs bitwise
  correctness, and running Triton in interpreter mode.
- [docs/paged-attention.md](docs/paged-attention.md) — L2: paged KV cache, the
  block table, and the silent-corruption fault demo.
- [docs/setup.md](docs/setup.md) — environments (base, vLLM CPU, `vllm-metal`, cloud GPU).

## Why tolerance, not equality

The whole lab rests on comparing a candidate schedule against an oracle *within a
tolerance justified by precision and reduction length*, rather than demanding
bitwise equality. A silent precision substitution — a path labelled fp32 that
computes in reduced precision — is invisible to `dtype` inspection and to latency
benchmarking, and shows up only when you measure outputs against a reference. The
per-layer docs above make that concrete from a hand-written matmul up to a real
engine.
