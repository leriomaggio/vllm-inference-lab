"""L3 engine tests.

The differential comparison is pure and always runs. The engine integration tests
(HuggingFace, vLLM) download a model and are gated behind ``VLLAB_RUN_ENGINE_TESTS``
so the default suite stays fast and offline.
"""

from __future__ import annotations

import os

import pytest

from vllab.engine.differential import compare_generations
from vllab.engine.types import GenResult

_RUN_ENGINE = bool(os.environ.get("VLLAB_RUN_ENGINE_TESTS"))
_SMALL_MODEL = os.environ.get("VLLAB_TEST_MODEL", "facebook/opt-125m")
_PROMPTS = ["The capital of France is", "Water boils at"]


# --------------------------------------------------------------------------- #
# pure differential logic
# --------------------------------------------------------------------------- #
def test_compare_identical_generations() -> None:
    ref = [GenResult("p", [1, 2, 3], "x", [-0.1, -0.2, -0.3])]
    cand = [GenResult("p", [1, 2, 3], "x", [-0.1, -0.2, -0.3])]
    report = compare_generations(ref, cand)
    assert report.exact_match_fraction == 1.0
    assert report.token_agreement_rate == 1.0
    assert report.max_logprob_gap == 0.0


def test_compare_detects_first_divergence() -> None:
    ref = [GenResult("p", [1, 2, 3, 4], "x", [-0.1, -0.2, -0.3, -0.4])]
    cand = [GenResult("p", [1, 2, 9, 4], "y", [-0.1, -0.2, -0.9, -0.4])]
    report = compare_generations(ref, cand)
    d = report.per_prompt[0]
    assert d.prefix_match == 2
    assert d.first_divergence == 2
    assert not d.identical
    # logprob gap is measured only over the shared prefix (positions 0..1).
    assert d.max_logprob_gap == pytest.approx(0.0, abs=1e-9)


def test_compare_logprob_gap_over_prefix() -> None:
    ref = [GenResult("p", [1, 2], "x", [-0.10, -0.20])]
    cand = [GenResult("p", [1, 2], "x", [-0.15, -0.20])]
    report = compare_generations(ref, cand)
    assert report.per_prompt[0].max_logprob_gap == pytest.approx(0.05, abs=1e-9)


def test_compare_prompt_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="prompt mismatch"):
        compare_generations([GenResult("a", [1], "", [])], [GenResult("b", [1], "", [])])


# --------------------------------------------------------------------------- #
# pure benchmark logic (no engine)
# --------------------------------------------------------------------------- #
class _FakeEngine:
    """Deterministic stand-in: emits ``min(max_new_tokens, cap)`` tokens per prompt."""

    def __init__(self, cap: int = 100) -> None:
        self.cap = cap
        self.calls: list[int] = []

    def greedy_generate(self, prompts, *, max_new_tokens: int = 16):
        self.calls.append(max_new_tokens)
        n = min(max_new_tokens, self.cap)
        return [GenResult(p, list(range(n)), "", []) for p in prompts]


def test_bench_result_derived_stats() -> None:
    from vllab.engine.bench import BenchResult, LatencySample

    res = BenchResult(
        engine="fake",
        model_id="m",
        dtype="fp32",
        num_prompts=2,
        max_new_tokens=10,
        samples=[LatencySample(1.0, 20), LatencySample(2.0, 20), LatencySample(3.0, 20)],
        prefill=LatencySample(0.5, 2),
    )
    assert res.median_wall_s == 2.0
    assert res.wall_range == (1.0, 3.0)
    # median-wall sample is the 2.0s / 20-token run -> 10 tok/s
    assert res.median_tokens_per_s == pytest.approx(10.0)
    # decode split: (20-2) tokens / (2.0-0.5) s = 12 tok/s
    assert res.decode_tokens_per_s == pytest.approx(12.0)


def test_bench_decode_split_none_without_prefill() -> None:
    from vllab.engine.bench import BenchResult, LatencySample

    res = BenchResult("f", "m", "fp32", 1, 4, [LatencySample(1.0, 4)], prefill=None)
    assert res.decode_tokens_per_s is None


def test_run_bench_calls_and_counts() -> None:
    from vllab.engine.bench import run_bench

    eng = _FakeEngine()
    res = run_bench(
        eng, ["a", "b"], engine_label="fake", model_id="m", dtype="fp32",
        max_new_tokens=8, repeats=3, warmup=1, measure_prefill=True,
    )
    # 1 warm-up + 3 timed full runs, all at 8 tokens; then 1 prefill warm-up + 1 timed at 1.
    assert eng.calls == [8, 8, 8, 8, 1, 1]
    assert len(res.samples) == 3
    assert all(s.new_tokens == 2 * 8 for s in res.samples)  # 2 prompts x 8 tokens
    assert res.prefill is not None and res.prefill.new_tokens == 2 * 1


def test_run_bench_rejects_empty_prompts() -> None:
    from vllab.engine.bench import run_bench

    with pytest.raises(ValueError, match="non-empty"):
        run_bench(_FakeEngine(), [], engine_label="f", model_id="m", dtype="fp32")


# --------------------------------------------------------------------------- #
# pure KV-cache footprint math (no engine)
# --------------------------------------------------------------------------- #
def test_kv_blocks_for_rounds_up() -> None:
    from vllab.engine.types import KVCacheReport

    assert KVCacheReport.blocks_for(0, 128) == 0
    assert KVCacheReport.blocks_for(1, 128) == 1
    assert KVCacheReport.blocks_for(128, 128) == 1
    assert KVCacheReport.blocks_for(129, 128) == 2
    with pytest.raises(ValueError, match="block_size"):
        KVCacheReport.blocks_for(10, 0)


def test_kv_capacity_tokens() -> None:
    from vllab.engine.types import KVCacheReport

    rep = KVCacheReport(block_size=128, num_blocks=455, cache_dtype="auto", kv_bytes=2**32)
    assert rep.capacity_tokens == 128 * 455


# --------------------------------------------------------------------------- #
# gated integration tests (download a model)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _RUN_ENGINE, reason="set VLLAB_RUN_ENGINE_TESTS=1 to run engine tests")
def test_hf_reference_is_deterministic() -> None:
    hf = pytest.importorskip("vllab.engine.hf_reference")
    if not hf.transformers_available():
        pytest.skip("transformers not installed")
    ref = hf.HFReference(_SMALL_MODEL, dtype="fp32")
    a = ref.greedy_generate(_PROMPTS, max_new_tokens=8)
    b = ref.greedy_generate(_PROMPTS, max_new_tokens=8)
    assert [r.token_ids for r in a] == [r.token_ids for r in b]


@pytest.mark.skipif(not _RUN_ENGINE, reason="set VLLAB_RUN_ENGINE_TESTS=1 to run engine tests")
def test_vllm_matches_hf_reference_fp32() -> None:
    hf_mod = pytest.importorskip("vllab.engine.hf_reference")
    vl_mod = pytest.importorskip("vllab.engine.vllm_runner")
    if not (hf_mod.transformers_available() and vl_mod.vllm_available()):
        pytest.skip("transformers or vLLM not installed")

    hf = hf_mod.HFReference(_SMALL_MODEL, dtype="fp32")
    vllm = vl_mod.VLLMRunner(_SMALL_MODEL, dtype="fp32")
    reference = hf.greedy_generate(_PROMPTS, max_new_tokens=8)
    candidate = vllm.greedy_generate(_PROMPTS, max_new_tokens=8)

    report = compare_generations(reference, candidate)
    # Greedy fp32 vs fp32: token sequences should match exactly.
    assert report.exact_match_fraction == 1.0, [d.first_divergence for d in report.per_prompt]
