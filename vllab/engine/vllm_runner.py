"""vLLM offline engine wrapper.

Presents the same surface as :class:`vllab.engine.hf_reference.HFReference`
(greedy generation with per-step logprobs) so the two can be compared directly.
vLLM is an optional, from-source dependency on macOS; the module imports without it
and :func:`vllm_available` reports its presence.

On macOS vLLM runs CPU-only and supports fp32/fp16. ``enforce_eager=True`` keeps
startup fast and avoids graph compilation, which is what we want for a correctness
lab rather than a throughput benchmark.
"""

from __future__ import annotations

import os

# CPU/macOS defaults, set before importing vLLM:
#  - run the engine in-process (the multiprocess executor fails to spawn workers
#    on macOS); this also surfaces real errors inline.
#  - reserve a small CPU KV-cache space (GiB); the CPU backend requires it set.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "4")

from .types import GenResult  # noqa: E402

try:
    import vllm  # noqa: F401

    _HAS_VLLM = True
except Exception:  # pragma: no cover - exercised where vLLM is absent
    _HAS_VLLM = False


def vllm_available() -> bool:
    return _HAS_VLLM


_DTYPE_NAMES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


class VLLMRunner:
    """Greedy decoding from an offline ``vllm.LLM`` with per-step logprobs."""

    def __init__(
        self,
        model_id: str,
        *,
        dtype: str = "fp32",
        seed: int = 0,
        max_model_len: int | None = 2048,
    ) -> None:
        if not _HAS_VLLM:
            raise RuntimeError("vLLM is not installed (see docs/setup.md for the CPU build)")
        if dtype not in _DTYPE_NAMES:
            raise ValueError(f"unknown dtype {dtype!r}; choose from {sorted(_DTYPE_NAMES)}")
        from vllm import LLM

        self.model_id = model_id
        self.dtype = dtype
        self._llm = LLM(
            model=model_id,
            dtype=_DTYPE_NAMES[dtype],
            seed=seed,
            enforce_eager=True,
            max_model_len=max_model_len,
            disable_log_stats=True,
        )

    def greedy_generate(self, prompts: list[str], *, max_new_tokens: int = 16) -> list[GenResult]:
        """Greedy-decode each prompt (temperature 0), returning ids, text, logprobs."""
        from vllm import SamplingParams

        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, logprobs=1)
        outputs = self._llm.generate(prompts, sp)

        results: list[GenResult] = []
        for prompt, out in zip(prompts, outputs, strict=True):
            o = out.outputs[0]
            token_ids = list(o.token_ids)
            step_logprobs: list[float] = []
            if o.logprobs is not None:
                for i, tok in enumerate(token_ids):
                    # vLLM always includes the sampled token in the per-step dict.
                    entry = o.logprobs[i].get(tok)
                    lp = float(entry.logprob) if entry is not None else float("nan")
                    step_logprobs.append(lp)
            results.append(GenResult(prompt, token_ids, o.text, step_logprobs))
        return results

    def cache_info(self) -> dict[str, int]:
        """Best-effort KV-cache configuration (block size and number of blocks).

        Reads the engine's cache config through non-public attributes, so it is
        wrapped defensively; returns an empty dict if the layout changes.
        """
        info: dict[str, int] = {}
        try:  # pragma: no cover - depends on vLLM internals
            cache_cfg = self._llm.llm_engine.cache_config
            info["block_size"] = int(cache_cfg.block_size)
            if getattr(cache_cfg, "num_cpu_blocks", None):
                info["num_cpu_blocks"] = int(cache_cfg.num_cpu_blocks)
            if getattr(cache_cfg, "num_gpu_blocks", None):
                info["num_gpu_blocks"] = int(cache_cfg.num_gpu_blocks)
        except Exception:
            pass
        return info
