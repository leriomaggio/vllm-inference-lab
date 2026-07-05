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
    corrupt: bool = typer.Option(False, help="Inject a block-table fault to show corruption."),
) -> None:
    """L2: verify paged decode == reference; optionally inject a block-table fault."""
    typer.echo("paged-demo: not implemented yet (milestone 4).")
    raise typer.Exit(code=0)


@app.command("bench")
def bench(
    model: str = typer.Option("facebook/opt-125m", help="Model id."),
) -> None:
    """L3: measure single-engine latency / throughput metrics (illustrative on CPU)."""
    typer.echo("bench: not implemented yet (milestone 5-6).")
    raise typer.Exit(code=0)


@app.command("validate")
def validate(
    model: str = typer.Option("Qwen/Qwen2.5-0.5B-Instruct", help="Model id."),
) -> None:
    """L3: differential correctness vLLM vs HuggingFace + fp32/fp16 precision axis."""
    typer.echo("validate: not implemented yet (milestone 6).")
    raise typer.Exit(code=0)


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
