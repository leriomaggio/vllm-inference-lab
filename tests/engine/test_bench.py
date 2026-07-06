"""Pure benchmark logic (``vllab.engine.bench``), exercised with a fake engine."""

from __future__ import annotations

import pytest

from vllab.engine.bench import BenchResult, LatencySample, run_bench
from vllab.engine.types import GenResult


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
    """Goal: the timing report's headline numbers are *derived* from the raw samples,
    so the median, spread, and the prefill/decode split must be computed exactly as
    documented — the honesty of the benchmark depends on these reductions being right.

    Expectation: three full samples (1/2/3 s, 20 tokens each) plus a 0.5 s / 2-token
    prefill probe. Asserts median wall 2.0 s, range (1.0, 3.0), throughput taken from
    the actual median-wall run (10 tok/s, not a ratio-of-medians), and the decode
    split ``(20-2)/(2.0-0.5) = 12`` tok/s — the full-minus-prefill isolation of the
    decode regime.
    """
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
    """Goal: the decode-only throughput is an *estimate* that requires the prefill
    probe to subtract; without that probe the number is undefined and must not be
    fabricated from thin air.

    Expectation: a result built with ``prefill=None``. Asserts ``decode_tokens_per_s``
    is ``None`` — the benchmark declines to report a split it cannot compute rather
    than emitting a misleading figure.
    """
    res = BenchResult("f", "m", "fp32", 1, 4, [LatencySample(1.0, 4)], prefill=None)
    assert res.decode_tokens_per_s is None


def test_run_bench_calls_and_counts() -> None:
    """Goal: the harness must drive the engine in exactly the documented shape —
    untimed warm-ups first (so lazy setup never contaminates a timed sample), then the
    timed full runs, then a warmed prefill probe — and must count *actual* emitted
    tokens, not the requested budget.

    Expectation: a fake engine recording its ``max_new_tokens`` calls under
    ``warmup=1, repeats=3, measure_prefill=True``. Asserts the call sequence is
    ``[8, 8, 8, 8, 1, 1]`` (warm-up + 3 full, then prefill warm-up + 1 probe), exactly
    3 samples are kept, and each counts ``2 prompts x 8`` real tokens — the timing
    protocol is honoured and nothing off-by-one slips in.
    """
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
    """Goal: benchmarking an empty prompt batch is meaningless (zero tokens, a
    divide-by-nothing throughput), so the harness must reject it up front instead of
    returning a degenerate all-zero result.

    Expectation: ``run_bench`` called with no prompts. Asserts a ``ValueError`` about
    the non-empty requirement — the guard fails fast rather than producing a
    nonsensical measurement.
    """
    with pytest.raises(ValueError, match="non-empty"):
        run_bench(_FakeEngine(), [], engine_label="f", model_id="m", dtype="fp32")
