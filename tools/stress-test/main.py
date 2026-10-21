# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stress-test controller. Run from the repository root with ``uv run``."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn

import typer
from bundle import build_zip
from cluster import ClusterWriteError
from config import ConfigError, Sweep, load_sweep, override_run_key, select_step
from controller import clean_jobs, clean_results, execute_steps, run_job_prefix, selected_steps
from helix import ContextError, HelixCluster, open_cluster
from job_profile import ProfileError, ProfileFacts
from names import NameError, resource_name
from records import write_json
from report import ReportError, report_files, write_report

app = typer.Typer(add_completion=False, no_args_is_help=True)
TOOL = Path(__file__).resolve().parent
DEFAULT_CONFIG = TOOL / "sweep.yaml"


def _sweep(config: Path, run_key: str | None, step: str | None) -> Sweep:
    path = config if config.is_file() else DEFAULT_CONFIG
    sweep = override_run_key(load_sweep(path), run_key)
    select_step(sweep, step)
    return sweep


def _fail(exc: Exception) -> NoReturn:
    print(exc, file=sys.stderr)
    raise typer.Exit(1) from exc


def _version() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=TOOL,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _dirs(run_key: str) -> tuple[Path, Path]:
    return TOOL / "runs" / run_key, TOOL / "output" / run_key


@app.command()
def publish(
    context: str | None = typer.Option(None, "--context", help="Saved CLI context."),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config"),
    run_key: str | None = typer.Option(None, "--run-key"),
    job_profile: str = typer.Option("default", "--job-profile"),
) -> None:
    """Upload the worker zip to the run fileset."""
    try:
        sweep = _sweep(config, run_key, None)
        _publish(sweep, context, job_profile, force=True)
    except (ConfigError, NameError, ProfileError, ContextError, OSError) as exc:
        _fail(exc)


@app.command()
def run(
    context: str | None = typer.Option(None, "--context", help="Saved CLI context."),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config"),
    run_key: str | None = typer.Option(None, "--run-key"),
    step: str | None = typer.Option(None, "--step", help="Replace one step id from the sweep."),
    job_profile: str = typer.Option("default", "--job-profile"),
) -> None:
    """Run unfinished steps, or replace the step named by ``--step``."""
    try:
        sweep = _sweep(config, run_key, step)
    except (ConfigError, NameError) as exc:
        _fail(exc)
    run_dir, _output = _dirs(sweep.run_key)
    chosen = selected_steps(sweep, run_dir, step)
    if not chosen:
        return
    try:
        cluster, facts, base_url, worker_digest = _publish(sweep, context, job_profile, force=True)
        write_json(
            run_dir / "meta.json",
            {
                "context": context or "",
                "base_url": base_url,
                "profile": facts.name,
                "ttl_seconds_before_active": facts.ttl_seconds_before_active,
                "ttl_seconds_active": facts.ttl_seconds_active,
                "worker_digest": worker_digest,
                "controller_version": _version(),
                "pods": {
                    kind: {
                        "cpu_request": pod.cpu_request,
                        "memory_request": pod.memory_request,
                        "cpu_limit": pod.cpu_limit,
                        "memory_limit": pod.memory_limit,
                    }
                    for kind, pod in sweep.pods.items()
                },
            },
        )
        code = execute_steps(
            sweep,
            run_dir,
            chosen,
            cluster,
            profile_name=facts.name,
            fileset=resource_name(sweep.run_key),
            ttl_before=facts.ttl_seconds_before_active,
            ttl_active=facts.ttl_seconds_active,
            clock=time.monotonic,
            sleep=time.sleep,
        )
    except (ConfigError, NameError, ProfileError, ContextError, OSError) as exc:
        _fail(exc)
    if code != 0:
        raise typer.Exit(code)


@app.command()
def report(
    context: str | None = typer.Option(None, "--context", help="Saved CLI context."),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config"),
    run_key: str | None = typer.Option(None, "--run-key"),
    job_profile: str = typer.Option("default", "--job-profile"),
) -> None:
    """Build the local report, then copy it to the run fileset."""
    try:
        sweep = _sweep(config, run_key, None)
        run_dir, output_dir = _dirs(sweep.run_key)
        write_report(sweep, run_dir, output_dir)
    except (ConfigError, NameError, ReportError, OSError) as exc:
        _fail(exc)
    try:
        cluster, _facts, _base_url, _digest = _publish(sweep, context, job_profile, force=False)
        for path in report_files(output_dir):
            cluster.write_file(f"reports/{path.name}", path.read_bytes())
    except (ProfileError, ContextError, ClusterWriteError, OSError) as exc:
        print(exc, file=sys.stderr)
        raise typer.Exit(1) from exc


@app.command("clean-results")
def clean_results_command(
    context: str | None = typer.Option(None, "--context", help="Saved CLI context."),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config"),
    run_key: str | None = typer.Option(None, "--run-key"),
    job_profile: str = typer.Option("default", "--job-profile"),
) -> None:
    """Delete this run's fileset objects, local bundle, and report directory."""
    try:
        sweep = _sweep(config, run_key, None)
        cluster, _facts, _base_url, _digest = _publish(sweep, context, job_profile, force=False)
        run_dir, output_dir = _dirs(sweep.run_key)
        clean_results(cluster, run_dir, output_dir)
    except (ConfigError, NameError, ProfileError, ContextError, OSError) as exc:
        _fail(exc)


@app.command("clean-jobs")
def clean_jobs_command(
    context: str | None = typer.Option(None, "--context", help="Saved CLI context."),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config"),
    run_key: str | None = typer.Option(None, "--run-key"),
    job_profile: str = typer.Option("default", "--job-profile"),
) -> None:
    """Cancel and delete every job for this run key. This is a reset, not a measurement."""
    try:
        sweep = _sweep(config, run_key, None)
        cluster, facts, _base_url, _digest = _publish(sweep, context, job_profile, force=False)
        finished = clean_jobs(
            cluster,
            run_job_prefix(sweep.run_key),
            clock=time.monotonic,
            sleep=time.sleep,
            budget=float(facts.ttl_seconds_before_active + facts.ttl_seconds_active),
        )
    except (ConfigError, NameError, ProfileError, ContextError, OSError) as exc:
        _fail(exc)
    if not finished:
        raise typer.Exit(1)


def _publish(
    sweep: Sweep,
    context: str | None,
    job_profile: str,
    *,
    force: bool,
) -> tuple[HelixCluster, ProfileFacts, str, str]:
    cluster, facts, base_url = open_cluster(
        context=context,
        workspace=resource_name(sweep.run_key),
        profile_name=job_profile,
        steps=list(sweep.steps),
    )
    payload = build_zip(TOOL / "worker")
    worker_digest = cluster.sync_worker(payload) if force else ""
    return cluster, facts, base_url, worker_digest


if __name__ == "__main__":
    app()
