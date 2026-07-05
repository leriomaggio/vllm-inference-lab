# Setup

## Base environment (L0-L2)

Layers L0 (reference oracles), L1 (Triton kernels, interpreter mode) and L2
(PagedAttention) need only a Python 3.12 environment with PyTorch and Triton.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[kernels,hf,dev]"
```

Triton is used exclusively in **interpreter mode** for the kernel layer, so no GPU
is required:

```bash
TRITON_INTERPRET=1 pytest tests/test_kernels.py
```

## vLLM CPU backend (L3) — macOS Apple Silicon

vLLM has no Apple GPU (Metal/MPS) backend in-tree; on macOS it builds as a
**CPU-only** engine. There are no prebuilt wheels for Apple Silicon, so it is a
source build. On macOS `VLLM_TARGET_DEVICE` is auto-set to `cpu`, and the CPU
implementation currently supports **fp32 and fp16 only** (bf16/fp8 belong to the
documented cloud-GPU path).

```bash
# Xcode Command Line Tools required (provides clang++):
xcode-select --install    # if not already installed

git clone https://github.com/vllm-project/vllm.git
cd vllm
# Pin to a known-good tag (recorded here once verified during milestone 5).
uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
uv pip install -e .
```

Reference: <https://docs.vllm.ai/en/latest/getting_started/installation/cpu/?device=apple>

## Optional: Apple-GPU engine via `vllm-metal` (L4)

`vllm-metal` is a community out-of-tree **hardware plugin** that runs vLLM on
Apple-Silicon GPUs through MLX + prebuilt Metal kernels. It is the same *shape* as
any other out-of-tree accelerator backend and gives a second, GPU-backed engine on
the same laptop for the L4 differential matrix. It requires native arm64 Python
3.12.

Reference: <https://github.com/vllm-project/vllm-metal>

The L4 matrix degrades gracefully: if `vllm-metal` is absent it runs with the
HuggingFace reference and the CPU vLLM engine alone.

## Cloud-GPU path (optional, for real perf and bf16/fp8)

CPU + tiny models cannot produce representative throughput/TTFT numbers, and
bf16/fp8 are unavailable on macOS CPU. To obtain real numbers, reproduce L3/L4 on
a CUDA host (e.g. a rented GPU, Modal, or RunPod) with a 1-7B model. The harness is
device-agnostic; only the engine construction differs.
