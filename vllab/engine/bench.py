"""Single-engine latency / throughput benchmark for the L3 engine layer.

Honesty note
------------
On CPU with a tiny model (``facebook/opt-125m``) these numbers are *illustrative*
of the prefill-vs-decode structure of autoregressive inference, **not**
representative of production throughput. The lab's weight is on correctness; the
documented path to honest throughput/latency curves is a cloud GPU with a larger
model. Every reported figure is a wall-clock measurement over real generations,
reported with its spread so the reader can see the noise.

What is measured
----------------
Two configurations of the *same* greedy generation over a fixed prompt batch:

* **full** — ``max_new_tokens = N`` decodes, timed ``repeats`` times after a
  warm-up. Yields end-to-end wall time and tokens/s.
* **prefill** — ``max_new_tokens = 1``, timed once after a warm-up. One decode
  step, so its wall time is dominated by prompt *prefill* (the TTFT proxy).

Subtracting the two isolates an **estimated decode throughput**:
``(full_tokens - prefill_tokens) / (full_wall - prefill_wall)``. This is a
first-order split, not a per-token trace, but it exposes the two regimes that
dominate LLM serving.

Any object exposing ``greedy_generate(prompts, max_new_tokens=...) -> list[GenResult]``
can be benched (both :class:`~vllab.engine.hf_reference.HFReference` and
:class:`~vllab.engine.vllm_runner.VLLMRunner` qualify).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import median
from typing import Protocol

from .types import GenResult


class _Engine(Protocol):
    """Structural type for anything benchable: a greedy generator."""

    def greedy_generate(
        self, prompts: list[str], *, max_new_tokens: int = ...
    ) -> list[GenResult]: ...


@dataclass(frozen=True)
class LatencySample:
    """One timed run over the whole prompt batch.

    Attributes
    ----------
    wall_s : float
        Wall-clock seconds for the batch generation.
    new_tokens : int
        Total continuation tokens produced across the batch (actual, so shorter
        when a sequence hits EOS early).
    """

    wall_s: float
    new_tokens: int

    @property
    def tokens_per_s(self) -> float:
        """Batch decode rate, ``new_tokens / wall_s`` (``0.0`` if no time elapsed)."""
        return self.new_tokens / self.wall_s if self.wall_s > 0 else 0.0


@dataclass(frozen=True)
class BenchResult:
    """Aggregate timing over ``repeats`` full runs, plus a prefill probe.

    Attributes
    ----------
    engine : str
        Label for the engine that was benched (e.g. ``"vllm"``, ``"hf"``).
    model_id : str
        Model identifier the engine loaded.
    dtype : str
        Short dtype name the engine ran in.
    num_prompts : int
        Number of prompts in the batch.
    max_new_tokens : int
        Decode budget per prompt for the ``full`` configuration.
    samples : list[LatencySample]
        One :class:`LatencySample` per timed full run, in run order.
    prefill : LatencySample or None
        A single ``max_new_tokens=1`` timing (TTFT proxy), or ``None`` if the
        prefill probe was disabled.
    """

    engine: str
    model_id: str
    dtype: str
    num_prompts: int
    max_new_tokens: int
    samples: list[LatencySample]
    prefill: LatencySample | None = None

    @property
    def median_wall_s(self) -> float:
        """Median full-run wall time across samples (``0.0`` if none)."""
        return median(s.wall_s for s in self.samples) if self.samples else 0.0

    @property
    def wall_range(self) -> tuple[float, float]:
        """``(min, max)`` full-run wall time across samples (``(0, 0)`` if none)."""
        if not self.samples:
            return (0.0, 0.0)
        walls = [s.wall_s for s in self.samples]
        return (min(walls), max(walls))

    @property
    def median_tokens_per_s(self) -> float:
        """Median end-to-end batch throughput across full runs.

        Taken as the throughput of the median-wall sample rather than the median
        of the ratios, so it corresponds to an actual observed run.
        """
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples, key=lambda s: s.wall_s)
        return ordered[len(ordered) // 2].tokens_per_s

    @property
    def decode_tokens_per_s(self) -> float | None:
        """Estimated decode-only throughput from the full − prefill split.

        Returns
        -------
        float or None
            ``(full_tokens - prefill_tokens) / (full_wall - prefill_wall)`` using
            the median full run, or ``None`` when no prefill probe was taken or
            the subtraction is degenerate (non-positive time or token delta).
        """
        if self.prefill is None or not self.samples:
            return None
        ordered = sorted(self.samples, key=lambda s: s.wall_s)
        full = ordered[len(ordered) // 2]
        d_tokens = full.new_tokens - self.prefill.new_tokens
        d_wall = full.wall_s - self.prefill.wall_s
        if d_wall <= 0 or d_tokens <= 0:
            return None
        return d_tokens / d_wall


def _time_once(engine: _Engine, prompts: list[str], max_new_tokens: int) -> LatencySample:
    """Time a single batch generation and count the tokens it produced."""
    start = time.perf_counter()
    outs = engine.greedy_generate(prompts, max_new_tokens=max_new_tokens)
    wall = time.perf_counter() - start
    new_tokens = sum(len(o.token_ids) for o in outs)
    return LatencySample(wall, new_tokens)


def run_bench(
    engine: _Engine,
    prompts: list[str],
    *,
    engine_label: str,
    model_id: str,
    dtype: str,
    max_new_tokens: int = 32,
    repeats: int = 3,
    warmup: int = 1,
    measure_prefill: bool = True,
) -> BenchResult:
    """Benchmark greedy generation of ``prompts`` on ``engine``.

    Runs ``warmup`` untimed generations (to absorb lazy allocation / first-call
    graph setup), then ``repeats`` timed full runs, then an optional single
    prefill probe (``max_new_tokens=1``, also warmed up once).

    Parameters
    ----------
    engine : _Engine
        Any object exposing ``greedy_generate``.
    prompts : list[str]
        The prompt batch, timed as a unit.
    engine_label, model_id, dtype : str
        Descriptive metadata recorded on the result (not used for timing).
    max_new_tokens : int, optional
        Decode budget for the full configuration. Default ``32``.
    repeats : int, optional
        Number of timed full runs. Default ``3``. Must be >= 1.
    warmup : int, optional
        Untimed full runs before timing. Default ``1``.
    measure_prefill : bool, optional
        Whether to take the ``max_new_tokens=1`` prefill probe. Default ``True``.

    Returns
    -------
    BenchResult
        The timed samples, prefill probe, and descriptive metadata.

    Raises
    ------
    ValueError
        If ``prompts`` is empty or ``repeats`` < 1.
    """
    if not prompts:
        raise ValueError("prompts must be non-empty")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    for _ in range(warmup):
        engine.greedy_generate(prompts, max_new_tokens=max_new_tokens)

    samples = [_time_once(engine, prompts, max_new_tokens) for _ in range(repeats)]

    prefill: LatencySample | None = None
    if measure_prefill:
        engine.greedy_generate(prompts, max_new_tokens=1)  # warm the prefill path
        prefill = _time_once(engine, prompts, 1)

    return BenchResult(
        engine=engine_label,
        model_id=model_id,
        dtype=dtype,
        num_prompts=len(prompts),
        max_new_tokens=max_new_tokens,
        samples=samples,
        prefill=prefill,
    )
