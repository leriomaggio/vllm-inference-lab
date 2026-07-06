"""Gated integration test for the vLLM engine wrapper.

Loads both engines, so it runs only under ``VLLAB_RUN_ENGINE_TESTS`` (the ``engine``
marker gates it — see ``conftest.py``).
"""

from __future__ import annotations

import pytest

from vllab.engine.differential import compare_generations


@pytest.mark.engine
def test_vllm_matches_hf_reference_fp32(small_model: str, prompts: list[str]) -> None:
    """Goal: the correctness gate of the whole engine layer — at equal precision the
    real engine must reproduce the oracle *exactly*. Greedy decoding is deterministic,
    so vLLM fp32 and HF fp32 compute the same math in a different kernel schedule and
    the argmax must be identical token for token.

    Expectation: HF fp32 reference vs vLLM fp32 on the shared prompts. Asserts
    ``exact_match_fraction == 1.0`` (reporting the first-divergence positions on
    failure) — a hard equality gate, not a tolerance question: any token mismatch is a
    bug in the engine integration. Gated: loads two engines.
    """
    hf_mod = pytest.importorskip("vllab.engine.hf_reference")
    vl_mod = pytest.importorskip("vllab.engine.vllm_runner")
    if not (hf_mod.transformers_available() and vl_mod.vllm_available()):
        pytest.skip("transformers or vLLM not installed")

    hf = hf_mod.HFReference(small_model, dtype="fp32")
    vllm = vl_mod.VLLMRunner(small_model, dtype="fp32")
    reference = hf.greedy_generate(prompts, max_new_tokens=8)
    candidate = vllm.greedy_generate(prompts, max_new_tokens=8)

    report = compare_generations(reference, candidate)
    # Greedy fp32 vs fp32: token sequences should match exactly.
    assert report.exact_match_fraction == 1.0, [d.first_divergence for d in report.per_prompt]
