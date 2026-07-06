# L3 — Real vLLM engine

L0–L2 built the inference stack from first principles: a matmul, an attention
oracle, a from-scratch PagedAttention cache. L3 swaps in the real thing — vLLM's
offline `LLM` engine — and holds it to the *same* discipline: its output is only
"correct" insofar as it agrees with a trusted reference within a justified
tolerance.

The layer has two halves:

1. **Benchmark + KV-cache introspection** (this document) — run one engine, time
   it honestly, and read out the KV-cache layout it allocated.
2. **Differential validation + precision axis** (added next) — vLLM greedy vs a
   HuggingFace `transformers` fp32 oracle, token-for-token, plus the fp32-vs-fp16
   divergence measurement.

## The pieces

| File | Role |
|------|------|
| `vllab/engine/hf_reference.py` | `HFReference`: eager `transformers` greedy decode + per-step logprobs — the trusted oracle. |
| `vllab/engine/vllm_runner.py` | `VLLMRunner`: same surface over an offline `vllm.LLM`; plus `kv_cache_report` for KV introspection. |
| `vllab/engine/bench.py` | `run_bench` + `BenchResult`: warm-up, timed full runs, and a prefill probe. |
| `vllab/engine/types.py` | `GenResult`, `KVCacheReport`, `PromptFootprint`. |
| `vllab/engine/differential.py` | `compare_generations` (used by the validation half). |

Any object exposing `greedy_generate(prompts, max_new_tokens=...)` is benchable, so
the same harness times either engine.

## What `bench` measures

Autoregressive inference has two regimes with very different costs:

- **Prefill** — one parallel forward pass over the whole prompt, filling the KV
  cache. Its cost scales with prompt length and is paid once.
- **Decode** — one forward pass *per generated token*, each reading the whole KV
  cache. Its cost scales with the number of tokens generated.

`bench` separates them with two timings of the *same* greedy generation:

- **full** — `max_new_tokens = N`, timed `repeats` times after a warm-up.
- **prefill** — `max_new_tokens = 1`, timed once (one decode step, so wall time is
  dominated by prefill — a time-to-first-token proxy).

Subtracting isolates an **estimated decode throughput**:

```
decode_tok/s ≈ (full_tokens − prefill_tokens) / (full_wall − prefill_wall)
```

This is a first-order split, not a per-token trace, but it exposes the structure.
The warm-up run matters: the first generation pays lazy allocation and one-time
setup that would otherwise contaminate the first timed sample.

## KV-cache introspection

`VLLMRunner.kv_cache_report(prompts)` reads the engine's live cache config
(`llm_engine.vllm_config.cache_config` on the v1 engine) and reports:

- **block size** — tokens per KV page;
- **device KV blocks** — total blocks allocated. On the CPU backend the device
  pool is exposed through the engine's *`num_gpu_blocks`* slot (`num_cpu_blocks`
  is `None`) — vLLM's unified device abstraction still calls it "gpu blocks" even
  when the device is the CPU;
- **capacity** — `block_size × num_blocks` KV token-slots;
- **reserved KV pool** — the bytes set aside (`VLLM_CPU_KVCACHE_SPACE`);
- **per-prompt footprint** — each prompt's prefill length in tokens and the
  logical blocks it occupies, `ceil(tokens / block_size)`.

## Validation

```bash
# build/load the engine first — see docs/setup.md
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CPU_KVCACHE_SPACE=4 \
    vllab bench --engine vllm --dtype fp32 --max-new-tokens 32 --repeats 3
```

### Measured — `facebook/opt-125m`, fp32, 4 prompts × 32 tokens, 3 runs (CPU)

```
median wall              799.9 ms
wall min-max             785.2 – 837.8 ms
end-to-end throughput    160.0 tok/s
prefill (TTFT proxy)     29.8 ms
decode throughput (est.) 161.0 tok/s

KV-cache layout
  block size (tokens/page)   128
  device KV blocks           455
  capacity                   58,240 tokens
  cache dtype                auto
  reserved KV pool           4.00 GiB

per-prompt prefill footprint
  "The capital of France is"      6 tokens   1 block
  "Water boils at"                4 tokens   1 block
  "In the beginning"              4 tokens   1 block
  "The three primary colors are"  6 tokens   1 block
```

What these numbers say:

1. **Decode dominates here.** Prefill for the whole batch is ~30 ms; the full
   32-token decode is ~800 ms. So end-to-end throughput (160 tok/s) and the
   decode-only estimate (161 tok/s) nearly coincide — with short prompts and a
   real decode budget, prefill is in the noise. Push the prompt length up (or the
   decode budget down) and the two separate; that separation is the point of
   measuring both.
2. **The cache is enormous relative to the workload.** 455 blocks × 128 tokens =
   58,240 KV slots reserved, while each prompt's prefill uses a *single* 128-token
   block. On CPU the KV pool is sized by a fixed byte budget, not the workload, so
   a tiny model over short prompts leaves it almost entirely idle — the block
   accounting only starts to bite with long contexts or many concurrent sequences.
3. **These are illustrative, not representative.** CPU + a 125M model cannot
   produce honest throughput curves; the value here is seeing the prefill/decode
   split and the KV-cache accounting, not the absolute tok/s. Real numbers are the
   documented cloud-GPU path.

## What comes next in L3

The **differential validation** half compares vLLM's greedy output against the
HuggingFace fp32 oracle token-for-token (greedy is deterministic, so a correct
engine must match exactly), and measures the **fp32-vs-fp16** divergence — the
recurring precision axis, now at the engine altitude. That is the `vllab validate`
command and `docs/engine.md`'s second section.
