# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provision the two userspace venvs `nemo agent-hardener setup` needs.

agent-hardener's own venv (installed via uv), and the separate garak venv its agent_breaker spawns
(delegated to agent-hardener's own ``setup``, which owns the garak version pin).
"""

from __future__ import annotations

import os
import shutil
import subprocess

import typer
from nemo_agent_hardener_plugin.cli.checks import redact_index_url
from nemo_agent_hardener_plugin.config import GARAK_PYTHON_ENVVAR, AgentHardenerConfig

# `uv pip install` pulls torch-sized wheels; generous enough for a cold cache on a slow link, but
# bounded so a hung download fails with a message instead of blocking setup forever.
SUBPROCESS_TIMEOUT_SECONDS = 900


def run_subprocess(
    cmd: list[str],
    action: str,
    env: dict[str, str] | None = None,
    *,
    cwd: str | None = None,
    timeout: int | None = SUBPROCESS_TIMEOUT_SECONDS,
) -> None:
    """Run *cmd* with its output streamed to the terminal, exiting with *action* context on failure.

    Output is inherited rather than captured: these are multi-minute installs, and swallowing uv's
    progress makes setup look hung. It also means both stdout and stderr reach the operator — uv
    reports some failures on stdout. Inherited stdio also lets an interactive child prompt the
    operator, which is why *timeout* accepts ``None`` (a person deciding is not a hung command).
    """
    try:
        proc = subprocess.run(cmd, check=False, env=env, cwd=cwd, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        typer.secho(f"Timed out after {timeout}s trying to {action} — the command made no progress.", fg="red")
        raise typer.Exit(code=1) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        typer.secho(f"Failed to {action}: {exc}", fg="red")
        raise typer.Exit(code=1) from exc
    if proc.returncode != 0:
        typer.secho(f"Failed to {action} (exit {proc.returncode}) — see the output above.", fg="red")
        raise typer.Exit(code=1)


def provision_venv(config: AgentHardenerConfig, *, force: bool) -> None:
    """Create agent-hardener's dedicated venv and install agent-hardener into it via uv."""
    if shutil.which("uv") is None:
        typer.secho("uv not found — install it (https://docs.astral.sh/uv/) then re-run setup.", fg="red")
        raise typer.Exit(code=1)

    if config.agent_hardener_bin.exists() and not force:
        typer.echo(f"agent-hardener venv already present at {config.venv_path} (use --force to recreate).")
        return

    config.venv_path.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Creating agent-hardener venv at {config.venv_path} ...")
    run_subprocess(["uv", "venv", "--python", "3.12", str(config.venv_path)], "create venv")

    typer.echo(f"Installing {config.spec} into the venv ...")
    install_cmd = ["uv", "pip", "install", "--python", str(config.venv_path / "bin" / "python")]
    # Both flags are passed on this command only, so the platform's own environment is never resolved
    # against the extra index. Credentials are deliberately not handled here — uv picks them up from
    # ~/.netrc or UV_INDEX_<NAME>_* in the inherited environment.
    if config.index_url:
        typer.echo(f"  using extra index {redact_index_url(config.index_url)}")
        install_cmd += ["--index", config.index_url]
    if config.index_strategy:
        install_cmd += ["--index-strategy", config.index_strategy]
    run_subprocess([*install_cmd, config.spec], "install agent-hardener")
    if not config.agent_hardener_bin.exists():
        typer.secho(
            f"Install finished but {config.agent_hardener_bin} is missing — check the package spec ({config.spec}).",
            fg="red",
        )
        raise typer.Exit(code=1)
    typer.secho(f"agent-hardener installed: {config.agent_hardener_bin}", fg="green")


def run_agent_hardener_setup(config: AgentHardenerConfig, *, force: bool) -> None:
    """Run ``agent-hardener setup`` (idempotent): provisions the garak venv and registers the gateway.

    Not gated on the garak venv, so the OpenShell gateway is re-ensured on every setup (agent-hardener
    fast-returns the existing venv). Needs agent-hardener installed first (``provision_venv``). The
    plugin points garak provisioning at its managed location via ``AGENT_HARDENER_GARAK_PYTHON``.
    """
    cmd = [str(config.agent_hardener_bin), "setup"]
    if force:
        cmd.append("--force")
    typer.echo("Running `agent-hardener setup` (garak venv + OpenShell gateway) ...")
    run_subprocess(cmd, "run agent-hardener setup", {**os.environ, GARAK_PYTHON_ENVVAR: str(config.garak_python)})
    if not config.garak_python.exists():
        typer.secho(f"agent-hardener setup finished but {config.garak_python} is missing.", fg="red")
        raise typer.Exit(code=1)
    typer.secho(f"garak venv ready: {config.garak_venv_path}", fg="green")
