"""Gated integration test for the HuggingFace reference engine.

Loads a model, so it runs only under ``VLLAB_RUN_ENGINE_TESTS`` (the ``engine``
marker gates it — see ``conftest.py``).
"""

from __future__ import annotations

import pytest


@pytest.mark.engine
def test_hf_reference_is_deterministic(small_model: str, prompts: list[str]) -> None:
    """Goal: the HuggingFace engine is the *oracle* every higher layer is compared
    against, so it must be reproducible — greedy decoding twice on the same prompts has
    to yield byte-identical token ids, or the reference itself is untrustworthy.

    Expectation: two ``greedy_generate`` calls on the same prompts. Asserts the two
    token-id sequences are identical — the oracle is deterministic, a precondition for
    treating any downstream divergence as a property of the engine under test rather
    than of the reference. Gated: loads a model.
    """
    hf = pytest.importorskip("vllab.engine.hf_reference")
    if not hf.transformers_available():
        pytest.skip("transformers not installed")
    ref = hf.HFReference(small_model, dtype="fp32")
    a = ref.greedy_generate(prompts, max_new_tokens=8)
    b = ref.greedy_generate(prompts, max_new_tokens=8)
    assert [r.token_ids for r in a] == [r.token_ids for r in b]
