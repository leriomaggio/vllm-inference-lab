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
