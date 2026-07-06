"""Typer command-line interface for vllm-inference-lab.

Commands are thin wrappers over the library layers (L0-L4). Each subcommand is
kept importable without its heavy optional dependency so that ``vllab --help`` and
the lightweight layers work even when vLLM / Triton are not installed.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Climb the inference stack from a hand-written matmul to a real vLLM engine, "
    "validating each layer against a reference oracle within a justified tolerance.",
)


@app.command("kernel-check")
def kernel_check(
    m: int = typer.Option(128, help="Rows of A / C."),
    k: int = typer.Option(256, help="Shared (reduction) dimension."),
    n: int = typer.Option(96, help="Cols of B / C."),
    seed: int = typer.Option(0, help="RNG seed for reproducible inputs."),
) -> None:
    """L1: validate matmul schedules against the fp64 oracle (interpreter mode)."""
    from rich.console import Console
    from rich.table import Table

    from vllab.kernels.harness import run_matmul_check

    check = run_matmul_check(m=m, k=k, n=n, seed=seed)
    console = Console()

    table = Table(title=f"matmul schedules vs fp64 oracle  (M,K,N)={check.shape}")
    table.add_column("backend")
    table.add_column("config")
    table.add_column("max_abs", justify="right")
    table.add_column("atol", justify="right")
    table.add_column("verdict", justify="center")
    for row in check.rows:
        table.add_row(
            row.backend,
            row.config,
            f"{row.max_abs:.3e}",
            f"{row.atol:.3e}",
            "[green]within[/]" if row.within else "[red]OVER[/]",
        )
    console.print(table)

    console.print(
        f"schedule divergence (tile-size low-bit gap): [bold]{check.schedule_divergence:.3e}[/] "
        "— non-zero, yet within tolerance: correctness here is tolerance-defined, not bitwise."
    )
    if not check.triton_ran:
        console.print(
            "[yellow]triton not importable here[/] — real kernel skipped; "
            "the PyTorch tiled model carries the schedule/accumulator lesson on CPU."
        )
    raise typer.Exit(code=0 if check.all_within else 1)


@app.command("paged-demo")
def paged_demo(
    heads: int = typer.Option(3, help="Number of attention heads."),
    seqlen: int = typer.Option(17, help="Sequence length (tokens)."),
    dim: int = typer.Option(16, help="Head dimension."),
    block_size: int = typer.Option(4, help="Tokens per KV block (page)."),
    seed: int = typer.Option(0, help="RNG seed."),
) -> None:
    """L2: verify paged decode == reference, then inject a block-table fault."""
    import torch
    from rich.console import Console

    from vllab.numerics import compare
    from vllab.paged.faults import block_table_fault_demo
    from vllab.paged.paged_attention import paged_decode
    from vllab.reference.kvcache import incremental_decode

    console = Console()
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(heads, seqlen, dim, generator=g)
    k = torch.randn(heads, seqlen, dim, generator=g)
    v = torch.randn(heads, seqlen, dim, generator=g)

    paged = paged_decode(q, k, v, block_size=block_size)
    reference = incremental_decode(q, k, v)
    rep = compare(paged, reference, atol=1e-10)
    console.print(
        f"paged vs non-paged reference: max_abs={rep.max_abs:.3e} "
        + ("[green]equal within fp64 noise[/]" if rep.within else "[red]DIVERGED[/]")
    )

    fault = block_table_fault_demo(q, k, v, block_size=block_size, corrupt_logical=0)
    console.print(
        f"block-table fault (logical 0: phys {fault.from_physical} -> {fault.to_physical}): "
        f"max_abs_diff=[bold]{fault.max_abs_diff:.3e}[/], "
        f"shape/dtype unchanged={fault.output_shape_unchanged}"
    )
    console.print(
        "[yellow]Same shape, same dtype, wrong numbers[/] — a mis-mapped page is "
        "silent to a latency benchmark and only a differential check catches it."
    )
    raise typer.Exit(code=0 if rep.within else 1)


_BENCH_PROMPTS = [
    "The capital of France is",
    "Water boils at",
    "In the beginning",
    "The three primary colors are",
]


@app.command("bench")
def bench(
    model: str = typer.Option("facebook/opt-125m", help="Model id."),
    engine: str = typer.Option("vllm", help="Engine to bench: 'vllm' or 'hf'."),
    dtype: str = typer.Option("fp32", help="Parameter dtype: fp32 or fp16."),
    max_new_tokens: int = typer.Option(32, help="Decode budget per prompt (full run)."),
    repeats: int = typer.Option(3, help="Number of timed full runs."),
    warmup: int = typer.Option(1, help="Untimed warm-up runs before timing."),
) -> None:
    """L3: single-engine latency / throughput + KV-cache introspection.

    Numbers are illustrative on CPU with a tiny model, not production throughput;
    the prefill/decode split shows the two regimes of autoregressive inference.
    """
    from rich.console import Console
    from rich.table import Table

    from vllab.engine.bench import run_bench

    console = Console()

    if engine == "vllm":
        from vllab.engine.vllm_runner import VLLMRunner, vllm_available

        if not vllm_available():
            console.print("[red]vLLM is not installed[/] — see docs/setup.md for the CPU build.")
            raise typer.Exit(code=2)
        eng = VLLMRunner(model, dtype=dtype)
    elif engine == "hf":
        from vllab.engine.hf_reference import HFReference, transformers_available

        if not transformers_available():
            console.print("[red]transformers is not installed[/] — pip install '.[hf]'.")
            raise typer.Exit(code=2)
        eng = HFReference(model, dtype=dtype)
    else:
        console.print(f"[red]unknown engine {engine!r}[/] — choose 'vllm' or 'hf'.")
        raise typer.Exit(code=2)

    result = run_bench(
        eng,
        _BENCH_PROMPTS,
        engine_label=engine,
        model_id=model,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        repeats=repeats,
        warmup=warmup,
    )

    lo, hi = result.wall_range
    table = Table(
        title=f"{engine} {dtype} latency  ({result.num_prompts} prompts x "
        f"{result.max_new_tokens} tok, {repeats} runs)"
    )
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("median wall", f"{result.median_wall_s * 1e3:.1f} ms")
    table.add_row("wall min-max", f"{lo * 1e3:.1f} - {hi * 1e3:.1f} ms")
    table.add_row("end-to-end throughput", f"{result.median_tokens_per_s:.1f} tok/s")
    if result.prefill is not None:
        table.add_row("prefill (TTFT proxy)", f"{result.prefill.wall_s * 1e3:.1f} ms")
    decode = result.decode_tokens_per_s
    table.add_row("decode throughput (est.)", f"{decode:.1f} tok/s" if decode else "n/a")
    console.print(table)

    # KV-cache introspection (vLLM only).
    if engine == "vllm":
        try:
            report = eng.kv_cache_report(_BENCH_PROMPTS)
        except RuntimeError as exc:
            console.print(f"[yellow]KV-cache introspection unavailable[/]: {exc}")
        else:
            kv = Table(title="KV-cache layout")
            kv.add_column("property")
            kv.add_column("value", justify="right")
            kv.add_row("block size (tokens/page)", str(report.block_size))
            kv.add_row("device KV blocks", str(report.num_blocks))
            kv.add_row("capacity", f"{report.capacity_tokens:,} tokens")
            kv.add_row("cache dtype", report.cache_dtype)
            kv.add_row("reserved KV pool", f"{report.kv_bytes / 2**30:.2f} GiB")
            console.print(kv)

            foot = Table(title="per-prompt prefill footprint")
            foot.add_column("prompt")
            foot.add_column("tokens", justify="right")
            foot.add_column("blocks", justify="right")
            for f in report.per_prompt:
                shown = f.prompt if len(f.prompt) <= 30 else f.prompt[:27] + "..."
                foot.add_row(shown, str(f.tokens), str(f.blocks))
            console.print(foot)

    console.print(
        "[dim]CPU + tiny model: figures are illustrative of the prefill/decode "
        "structure, not representative throughput.[/]"
    )
    raise typer.Exit(code=0)


_VALIDATE_PROMPTS = [
    "The capital of France is",
    "Water boils at",
    "In the beginning",
    "The three primary colors are",
    "Once upon a time",
]


@app.command("validate")
def validate(
    model: str = typer.Option("facebook/opt-125m", help="Model id."),
    max_new_tokens: int = typer.Option(16, help="Greedy tokens per prompt."),
    skip_fp16: bool = typer.Option(False, help="Skip the fp16 precision axis."),
) -> None:
    """L3: differential correctness (vLLM fp32 vs HF fp32) + fp32/fp16 precision axis.

    The fp32-vs-fp32 token match is the correctness gate (greedy is deterministic,
    so a correct engine matches the oracle exactly). The fp16 axis measures how far
    reduced precision drifts from the fp32 oracle, classified against a dtype-derived
    logprob band.
    """
    from rich.console import Console
    from rich.table import Table

    from vllab.engine.differential import compare_generations
    from vllab.engine.hf_reference import HFReference, transformers_available
    from vllab.engine.precision import precision_result
    from vllab.engine.vllm_runner import VLLMRunner, vllm_available

    console = Console()
    if not transformers_available():
        console.print("[red]transformers is not installed[/] — pip install '.[hf]'.")
        raise typer.Exit(code=2)
    if not vllm_available():
        console.print("[red]vLLM is not installed[/] — see docs/setup.md for the CPU build.")
        raise typer.Exit(code=2)

    prompts = _VALIDATE_PROMPTS
    hf = HFReference(model, dtype="fp32")
    hidden = int(hf.model.config.hidden_size)
    reference = hf.greedy_generate(prompts, max_new_tokens=max_new_tokens)

    # --- correctness gate: vLLM fp32 vs HF fp32 (exact token match expected) ---
    vllm_fp32 = VLLMRunner(model, dtype="fp32")
    fp32_cand = vllm_fp32.greedy_generate(prompts, max_new_tokens=max_new_tokens)
    fp32_diff = compare_generations(reference, fp32_cand)
    fp32 = precision_result(fp32_diff, candidate_dtype="fp32", reduction_length=hidden)

    gate = Table(title=f"correctness gate — vLLM fp32 vs HF fp32  ({model}, K={hidden})")
    gate.add_column("check")
    gate.add_column("value", justify="right")
    gate.add_column("verdict", justify="center")
    gate.add_row(
        "token exact match",
        f"{fp32_diff.exact_match_fraction:.0%}",
        "[green]PASS[/]" if fp32.token_exact else "[red]FAIL[/]",
    )
    gate.add_row(
        "logprob gap vs fp32 band",
        f"{fp32_diff.max_logprob_gap:.2e} / {fp32.band:.2e}",
        "[green]within[/]" if fp32.logprob_within_band else "[red]OVER[/]",
    )
    console.print(gate)

    ok = fp32.token_exact and fp32.logprob_within_band

    # --- precision axis: vLLM fp16 vs HF fp32 oracle ---
    if not skip_fp16:
        vllm_fp16 = VLLMRunner(model, dtype="fp16")
        fp16_cand = vllm_fp16.greedy_generate(prompts, max_new_tokens=max_new_tokens)
        fp16_diff = compare_generations(reference, fp16_cand)
        fp16 = precision_result(fp16_diff, candidate_dtype="fp16", reduction_length=hidden)

        axis = Table(title="precision axis — vLLM fp16 vs HF fp32 oracle")
        axis.add_column("check")
        axis.add_column("value", justify="right")
        axis.add_column("verdict", justify="center")
        axis.add_row(
            "token agreement",
            f"{fp16_diff.token_agreement_rate:.2%}",
            "[green]exact[/]" if fp16.token_exact else "[yellow]flips[/]",
        )
        axis.add_row(
            "logprob gap vs fp16 band",
            f"{fp16_diff.max_logprob_gap:.2e} / {fp16.band:.2e}",
            "[green]within[/]" if fp16.logprob_within_band else "[red]OVER[/]",
        )
        axis.add_row(
            "gap vs fp32 schedule floor",
            f"{fp16.band_ratio:.0f}x",
            "[yellow]precision effect[/]" if fp16.exceeds_oracle_band else "[green]schedule[/]",
        )
        console.print(axis)
        console.print(
            f"[dim]fp16 diverges from the fp32 oracle ~{fp16.band_ratio:.0f}x the fp32 "
            "schedule-noise floor — invisible to token-id equality, plain in the logprobs.[/]"
        )
        ok = ok and fp16.logprob_within_band

    raise typer.Exit(code=0 if ok else 1)


@app.command("matrix")
def matrix(
    out: str = typer.Option("artifacts/run.json", help="Artifact output path."),
) -> None:
    """L4: run the multi-backend correctness/latency matrix and write an artifact."""
    typer.echo("matrix: not implemented yet (milestone 7).")
    raise typer.Exit(code=0)


@app.command("report")
def report(
    artifact: str = typer.Argument(..., help="Path to a run artifact JSON."),
) -> None:
    """Render a markdown report from a run artifact."""
    typer.echo("report: not implemented yet (milestone 7).")
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
