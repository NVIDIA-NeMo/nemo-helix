# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stress-test controller. Run from the repository root with ``uv run``."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import NoReturn, Protocol

import typer
from bundle import build_zip
from cluster import ClusterWriteError
from config import ConfigError, Step, Sweep, format_catalog, load_sweep, override_run_key, select_step, sweep_files
from controller import clean_jobs, clean_results, execute_steps, run_job_prefix, selected_steps, step_summary
from helix import ContextError, HelixCluster, open_cluster
from job_profile import ProfileError, ProfileFacts, runnable_steps, validate_profile
from names import NameError, resource_name, union_workspace_names
from records import marker_matches, write_json
from report import ReportError, report_files, write_report


class _ReportStore(Protocol):
    def write_file(self, path: str, data: bytes) -> None:
        """Copy one object into the run fileset."""


_DEBUG = "Log each HTTP request and each job status change."
_HTTP_LOGGERS = ("httpx", "httpcore", "urllib3")
_LOG = logging.getLogger("stress")
_STDERR = logging.StreamHandler(sys.stderr)
_STDERR.setFormatter(logging.Formatter("%(message)s"))


def configure_logging(debug: bool) -> None:
    """Show job start, stop, error, and the report path. ``--debug`` adds HTTP and status lines."""
    if _STDERR not in _LOG.handlers:
        _LOG.addHandler(_STDERR)
    _LOG.setLevel(logging.DEBUG if debug else logging.INFO)
    for name in _HTTP_LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG if debug else logging.WARNING)
        if debug and _STDERR not in logger.handlers:
            logger.addHandler(_STDERR)
        if not debug and _STDERR in logger.handlers:
            logger.removeHandler(_STDERR)


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
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
    job_profile: str = typer.Option(
        "default", "--job-profile", help="Profile name or provider/profile, such as cpu/gpu."
    ),
    debug: bool = typer.Option(False, "--debug", help=_DEBUG),
) -> None:
    """Upload the worker zip to the run fileset."""
    configure_logging(debug)
    try:
        sweep = _sweep(config, run_key, None)
        _publish(sweep, context, job_profile, force=True)
    except (ConfigError, NameError, ProfileError, ContextError, OSError) as exc:
        _fail(exc)


@app.command("list")
def list_sweeps(
    config: Path | None = typer.Option(None, "--config", help="List one sweep file. Omit this to list every sweep."),
) -> None:
    """List sweep files and their steps. This does not contact a cluster."""
    try:
        paths = [config] if config is not None else sweep_files(TOOL)
        if not paths:
            raise ConfigError(f"no sweep files in {TOOL}")
        blocks: list[str] = []
        for path in paths:
            if not path.is_file():
                raise ConfigError(f"sweep file {path} was not found")
            label = path.name if path.parent.resolve() == TOOL else str(path)
            blocks.append(format_catalog(label, load_sweep(path)))
        print("\n\n".join(blocks))
    except (ConfigError, NameError, OSError) as exc:
        _fail(exc)


@app.command(no_args_is_help=True)
def run(
    context: str = typer.Option(..., "--context", help="Saved CLI context. Required."),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config"),
    run_key: str | None = typer.Option(None, "--run-key"),
    step: str | None = typer.Option(None, "--step", help="Replace one step id from the sweep."),
    job_profile: str = typer.Option(
        "default", "--job-profile", help="Profile name or provider/profile, such as cpu/gpu."
    ),
    debug: bool = typer.Option(False, "--debug", help=_DEBUG),
    image: str | None = typer.Option(None, "--image", help="Override the sweep image for this run."),
    force: bool = typer.Option(False, "--force", help="Rerun finished steps. --step still limits the run to one step."),
    workspaces: int | None = typer.Option(
        None,
        "--workspaces",
        help="Override every step's workspace count for this run. Omit this to use the sweep.",
    ),
) -> None:
    """Run unfinished steps, or replace the step named by ``--step``."""
    configure_logging(debug)
    try:
        context_name = _required_context(context)
        sweep = _sweep(config, run_key, step)
        image_name = _optional_image(image) or sweep.image
        if workspaces is not None:
            sweep = _override_workspaces(sweep, _workspace_count(workspaces))
    except (ConfigError, NameError, ContextError, ProfileError) as exc:
        _fail(exc)
    run_dir, _output = _dirs(sweep.run_key)
    chosen = selected_steps(sweep, run_dir, step, force=force)
    if not chosen:
        print(
            f"every step in {sweep.run_key} is already complete; use --force to rerun the sweep, or --step to rerun one",
            file=sys.stderr,
        )
        if run_dir.is_dir():
            _finish_or_fail(sweep, run_dir, _output, ran=[], skipped=[], only=None, cluster=None)
        return
    try:
        primary_name = resource_name(sweep.run_key)
        cluster, facts, base_url, auth_enabled = _open_workspace(context_name, job_profile, primary_name)
        kept, skipped = runnable_steps(chosen, auth_enabled=auth_enabled, only=step)
        validate_profile(facts, [item for item, _force in kept])
        if image_name is None and facts.task_image is None:
            raise ProfileError(
                "the sweep has no image and the profile has no default task image; set image in the sweep or pass --image"
            )
        spaces = union_workspace_names(sweep.run_key, [item.workspaces for item, _force in kept])
        opened: dict[str, HelixCluster] = {primary_name: cluster}
        for space in spaces:
            if space not in opened:
                shard, _facts, _url, _auth = _open_workspace(context_name, job_profile, space)
                opened[space] = shard
        _announce(facts, auth_enabled, image_name, spaces or None)
        if skipped:
            print(f"auth is disabled; skipping {', '.join(skipped)}", file=sys.stderr)
        worker_digest = ""
        if any(item.kind in {"hammer", "barrier"} for item, _force in kept):
            payload = build_zip(TOOL / "worker")
            targets = [opened[name] for name in spaces]
            worker_digest = targets[0].sync_worker(payload)
            for shard in targets[1:]:
                shard.sync_worker(payload)
        _write_meta(run_dir, sweep, context_name, base_url, facts, auth_enabled, worker_digest, image_name)
        code = 0
        if kept:
            code = execute_steps(
                sweep,
                run_dir,
                kept,
                cluster,
                profile_name=facts.name,
                provider=facts.provider,
                fileset=primary_name,
                ttl_before=facts.ttl_seconds_before_active,
                ttl_active=facts.ttl_seconds_active,
                clock=time.monotonic,
                sleep=time.sleep,
                image=image_name,
                workspace_clusters=opened,
            )
    except (ConfigError, NameError, ProfileError, ContextError, OSError) as exc:
        _fail(exc)
    _finish_or_fail(
        sweep,
        run_dir,
        _output,
        ran=[item.id for item, _force in kept],
        skipped=skipped,
        only=step,
        cluster=cluster,
    )
    if code != 0:
        raise typer.Exit(code)


@app.command()
def report(
    context: str | None = typer.Option(None, "--context", help="Saved CLI context."),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config"),
    run_key: str | None = typer.Option(None, "--run-key"),
    job_profile: str = typer.Option(
        "default", "--job-profile", help="Profile name or provider/profile, such as cpu/gpu."
    ),
    debug: bool = typer.Option(False, "--debug", help=_DEBUG),
) -> None:
    """Build the local report, then copy it to the run fileset."""
    configure_logging(debug)
    try:
        sweep = _sweep(config, run_key, None)
        run_dir, output_dir = _dirs(sweep.run_key)
        write_report(sweep, run_dir, output_dir)
        _report_written(output_dir / "report.md")
    except (ConfigError, NameError, ReportError, OSError) as exc:
        _fail(exc)
    try:
        cluster, _facts, _base_url, _auth, _digest = _publish(sweep, context, job_profile, force=False)
        _copy_report(output_dir, cluster)
        print(f"report copied to the {resource_name(sweep.run_key)} fileset", file=sys.stderr)
    except (ProfileError, ContextError, ClusterWriteError, OSError) as exc:
        print(exc, file=sys.stderr)
        raise typer.Exit(1) from exc


@app.command("clean-results")
def clean_results_command(
    context: str | None = typer.Option(None, "--context", help="Saved CLI context."),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config"),
    run_key: str | None = typer.Option(None, "--run-key"),
    job_profile: str = typer.Option(
        "default", "--job-profile", help="Profile name or provider/profile, such as cpu/gpu."
    ),
    debug: bool = typer.Option(False, "--debug", help=_DEBUG),
) -> None:
    """Delete this run's fileset objects, local bundle, and report directory."""
    configure_logging(debug)
    try:
        sweep = _sweep(config, run_key, None)
        cluster, _facts, _base_url, _auth, _digest = _publish(sweep, context, job_profile, force=False)
        run_dir, output_dir = _dirs(sweep.run_key)
        clean_results(cluster, run_dir, output_dir)
    except (ConfigError, NameError, ProfileError, ContextError, OSError) as exc:
        _fail(exc)


@app.command("clean-jobs")
def clean_jobs_command(
    context: str | None = typer.Option(None, "--context", help="Saved CLI context."),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config"),
    run_key: str | None = typer.Option(None, "--run-key"),
    job_profile: str = typer.Option(
        "default", "--job-profile", help="Profile name or provider/profile, such as cpu/gpu."
    ),
    debug: bool = typer.Option(False, "--debug", help=_DEBUG),
    workspaces: int | None = typer.Option(
        None, "--workspaces", help="Also clean this many workspaces. Omit this to clean every workspace the sweep uses."
    ),
) -> None:
    """Cancel and delete every job for this run key. This is a reset, not a measurement."""
    configure_logging(debug)
    try:
        sweep = _sweep(config, run_key, None)
        counts = [item.workspaces for item in sweep.steps]
        if workspaces is not None:
            counts.append(_workspace_count(workspaces))
        finished = True
        for space in union_workspace_names(sweep.run_key, counts):
            cluster, facts, _base_url, _auth = _open_workspace(context, job_profile, space)
            _announce(facts, _auth, spaces=[space])
            finished = (
                clean_jobs(
                    cluster,
                    run_job_prefix(sweep.run_key),
                    clock=time.monotonic,
                    sleep=time.sleep,
                    budget=float(facts.ttl_seconds_before_active + facts.ttl_seconds_active),
                )
                and finished
            )
    except (ConfigError, NameError, ProfileError, ContextError, OSError) as exc:
        _fail(exc)
    if not finished:
        raise typer.Exit(1)


def _open(sweep: Sweep, context: str | None, job_profile: str) -> tuple[HelixCluster, ProfileFacts, str, bool]:
    return _open_workspace(context, job_profile, resource_name(sweep.run_key))


def _open_workspace(
    context: str | None, job_profile: str, workspace: str
) -> tuple[HelixCluster, ProfileFacts, str, bool]:
    return open_cluster(context=context, workspace=workspace, profile_name=job_profile)


def _override_workspaces(sweep: Sweep, count: int) -> Sweep:
    return replace(sweep, steps=tuple(replace(step, workspaces=count) for step in sweep.steps))


def _workspace_count(value: int) -> int:
    if value < 1:
        raise NameError("--workspaces must be a positive integer")
    return value


def _publish(
    sweep: Sweep,
    context: str | None,
    job_profile: str,
    *,
    force: bool,
) -> tuple[HelixCluster, ProfileFacts, str, bool, str]:
    cluster, facts, base_url, auth_enabled = _open(sweep, context, job_profile)
    _announce(facts, auth_enabled)
    payload = build_zip(TOOL / "worker")
    worker_digest = cluster.sync_worker(payload) if force else ""
    return cluster, facts, base_url, auth_enabled, worker_digest


def _required_context(context: str) -> str:
    name = context.strip()
    if not name:
        raise ContextError("--context is required")
    return name


def _finish_or_fail(
    sweep: Sweep,
    run_dir: Path,
    output_dir: Path,
    *,
    ran: list[str],
    skipped: list[str],
    only: str | None,
    cluster: _ReportStore | None,
) -> None:
    try:
        _finish(sweep, run_dir, output_dir, ran=ran, skipped=skipped, only=only, cluster=cluster)
    except (ReportError, ClusterWriteError, OSError) as exc:
        _fail(exc)


def _finish(
    sweep: Sweep,
    run_dir: Path,
    output_dir: Path,
    *,
    ran: list[str],
    skipped: list[str],
    only: str | None,
    cluster: _ReportStore | None,
) -> None:
    """Write the report, say where it is, and copy it when a cluster is open."""
    ran_ids = set(ran)
    skipped_ids = set(skipped)
    for step in sweep.steps:
        _LOG.debug("  %s: %s", step.id, _outcome(run_dir, step, ran_ids, skipped_ids, only))
    _LOG.debug("records: %s", run_dir)
    write_report(sweep, run_dir, output_dir)
    report_path = output_dir / "report.md"
    _report_written(report_path)
    if cluster is not None:
        fileset = resource_name(sweep.run_key)
        _copy_report(output_dir, cluster)
        print(f"report copied to the {fileset} fileset", file=sys.stderr)


def _outcome(run_dir: Path, step: Step, ran: set[str], skipped: set[str], only: str | None) -> str:
    if step.id in skipped:
        return "skipped because auth is disabled"
    if step.id in ran:
        return step_summary(run_dir, step.id)
    if only is not None:
        return "not selected"
    if marker_matches(run_dir, step):
        return "already complete"
    return "not run"


def _report_written(path: Path) -> None:
    print(f"report: {path}", file=sys.stderr)


def _copy_report(output_dir: Path, cluster: _ReportStore) -> None:
    for path in report_files(output_dir):
        cluster.write_file(f"reports/{path.name}", path.read_bytes())


def _optional_image(image: str | None) -> str | None:
    if image is None:
        return None
    name = image.strip()
    if not name:
        raise ProfileError("--image is empty")
    return name


def _announce(
    facts: ProfileFacts, auth_enabled: bool, image: str | None = None, spaces: list[str] | None = None
) -> None:
    state = "enabled" if auth_enabled else "disabled"
    chosen = f"; image {image}" if image else ""
    print(f"using profile {facts.provider}/{facts.name} ({facts.backend}){chosen}; auth {state}", file=sys.stderr)
    if spaces is not None:
        label = "workspace" if len(spaces) == 1 else "workspaces"
        print(f"{label}: {', '.join(spaces)}", file=sys.stderr)


def _write_meta(
    run_dir: Path,
    sweep: Sweep,
    context: str | None,
    base_url: str,
    facts: ProfileFacts,
    auth_enabled: bool,
    worker_digest: str,
    image: str | None = None,
) -> None:
    write_json(
        run_dir / "meta.json",
        {
            "context": context or "",
            "base_url": base_url,
            "profile": facts.name,
            "backend": facts.backend,
            "auth": "enabled" if auth_enabled else "disabled",
            "ttl_seconds_before_active": facts.ttl_seconds_before_active,
            "ttl_seconds_active": facts.ttl_seconds_active,
            "image": image or "",
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


if __name__ == "__main__":
    app()
