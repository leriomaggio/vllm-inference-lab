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

## Differential validation + precision axis

`vllab validate` holds the real engine to the oracle. It has two parts, both
driven by the same differential harness (`compare_generations`), which reports the
identical-token prefix length and the largest per-step log-probability gap over
that prefix.

### The correctness gate — vLLM fp32 vs HF fp32

Greedy decoding is deterministic: given identical logits, `argmax` is identical.
vLLM fp32 and HF fp32 compute the *same* math in a *different kernel schedule*, so
the tokens must match **exactly** and the logprobs must agree to within fp32
schedule noise. This is a hard gate — any token mismatch is a bug in the engine
integration, not a tolerance question.

### The precision axis — vLLM fp16 vs the fp32 oracle

fp16 rounds the logits at ~2⁻¹⁰ instead of ~2⁻²³. The tolerance on the per-step
logprob gap comes from the same `reduction_atol(dtype, K)` model used at L1/L2,
with `K` = the model **hidden size** (the reduction behind each logit is
`hidden_state @ lm_head.T`, a sum over hidden). fp16's band is ~8200× looser than
fp32's because its unit roundoff is 8200× larger.

The finding: fp16 may still match tokens (greedy is robust when the argmax margin
is wide), but its logprob gap is **orders of magnitude above the fp32
schedule-noise floor**. `validate` reports that ratio — the precision effect made
measurable, invisible to a token-id check.

### Validation

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CPU_KVCACHE_SPACE=4 \
    vllab validate --model facebook/opt-125m --max-new-tokens 16
```

#### Measured — `facebook/opt-125m`, K = 768, 5 prompts × 16 tokens (CPU)

```
correctness gate — vLLM fp32 vs HF fp32
  token exact match          100%                     PASS
  logprob gap vs fp32 band   1.10e-05 / 2.64e-05      within

precision axis — vLLM fp16 vs HF fp32 oracle
  token agreement            100.00%                  exact
  logprob gap vs fp16 band   8.33e-03 / 2.17e-01      within
  gap vs fp32 schedule floor 315x                     precision effect
```

Reading these three rows top to bottom:

1. **fp32 is a hard match.** Same tokens, and the logprob gap (1.1e-5) sits at
   ~0.4× the fp32 band — the residual is pure kernel-schedule noise between vLLM's
   fused kernels and HF's eager path.
2. **fp16 keeps the tokens but not the numbers.** On these prompts every token
   still agrees (no argmax flipped within 16 steps), yet the logprob gap is
   8.3e-3 — **~750× the fp32 gap**, and comfortably within its own (much looser)
   fp16 band.
3. **The precision effect is real and quantified.** That fp16 gap is **~315× the
   fp32 schedule-noise floor**: if you (wrongly) judged the fp16 engine by the
   fp32 tolerance, you would flag it — correctly — as running reduced precision.
   The token-id check alone would have called both engines identical. This is the
   lab's recurring point at engine altitude: *the compute precision is invisible to
   equality checks and only shows up when you measure against a trusted oracle
   within a justified tolerance.*

Token agreement being 100% here is prompt-dependent, not guaranteed: a prompt whose
top-two logits fall within the fp16 noise floor would flip a token. The harness
reports the first-divergence position per prompt so such a flip is localised rather
than hidden in an aggregate.

## What the engine tests assert

- `test_vllm_matches_hf_reference_fp32` (gated) — vLLM fp32 tokens == HF fp32,
  exactly;
- `test_vllm_fp16_precision_axis` (gated) — fp16's logprob gap stays within the
  fp16 band but exceeds the fp32 floor (`exceeds_oracle_band`);
- `test_logprob_band_ordering_and_growth`, `test_precision_result_*` — the band is
  larger for fp16 than fp32 and grows like √K; the verdict flags classify the two
  regimes correctly;
- plus the bench and KV-footprint pure tests above, and the always-on differential
  logic tests.
