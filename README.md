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
