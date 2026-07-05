"""Scaffold smoke tests: the package imports and the CLI is wired up."""

from __future__ import annotations

from typer.testing import CliRunner

from vllab import __version__
from vllab.cli import app

runner = CliRunner()


def test_version_is_set() -> None:
    assert __version__


def test_cli_help_runs() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "kernel-check" in result.output


def test_subcommands_registered() -> None:
    for cmd in ["kernel-check", "paged-demo", "bench", "validate", "matrix", "report"]:
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0, f"{cmd} --help failed: {result.output}"
