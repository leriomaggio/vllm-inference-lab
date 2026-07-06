"""Shared fixtures and gating for the L3 engine test package.

Pure-logic tests (differential, bench, precision, KV-cache math) run offline in the
default suite. Integration tests that download and load a model are marked
``@pytest.mark.engine`` and skipped unless ``VLLAB_RUN_ENGINE_TESTS`` is set, so the
default run stays fast and offline.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "engine: integration test that loads a model (needs VLLAB_RUN_ENGINE_TESTS=1)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``engine``-marked tests unless the opt-in env var is set."""
    if os.environ.get("VLLAB_RUN_ENGINE_TESTS"):
        return
    skip = pytest.mark.skip(reason="set VLLAB_RUN_ENGINE_TESTS=1 to run engine tests")
    for item in items:
        if item.get_closest_marker("engine") is not None:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def small_model() -> str:
    """Model id used by the gated integration tests."""
    return os.environ.get("VLLAB_TEST_MODEL", "facebook/opt-125m")


@pytest.fixture(scope="session")
def prompts() -> list[str]:
    """Deterministic prompt set shared across integration tests."""
    return ["The capital of France is", "Water boils at"]
