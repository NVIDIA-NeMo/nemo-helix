# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo agents usage`` — token-usage reports from nat_runner outputs.

One subcommand, ``nemo agents usage show <ref>``, registered onto the
parent ``agents`` Typer app under a ``usage`` group.

``<ref>`` accepts a ``Union[LocalDir, FilesetRef]`` per the
``nemo_platform_plugin.refs`` convention — path-shaped values are read locally,
bare names download from a fileset.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import typer
from nemo_agents_plugin.cli_context import BaseUrlOption, resolve_base_url, resolve_context_headers
from nemo_agents_plugin.usage import compute, render
from nemo_agents_plugin.usage import parser as parser_module
from nemo_agents_plugin.usage.models import (
    BatchUsageReport,
    TaskUsage,
    UsageReport,
)
from nemo_agents_plugin.usage.sources.fileset import FilesetDownloadError, FilesetRefError, fileset_path
from nemo_agents_plugin.usage.sources.local import UsageSourceError, local_path
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.refs import FilesetRef, LocalDir, classify_output_target

logger = logging.getLogger(__name__)

_DEFAULT_WORKSPACE = "default"


def _validate_total_params(value: Optional[float]) -> Optional[float]:
    if value is None:
        return value
    if not math.isfinite(value):
        raise typer.BadParameter(f"--total-params must be a finite positive number, got {value}")
    if value <= 0:
        raise typer.BadParameter(f"--total-params must be > 0, got {value}")
    return value


def register_usage_commands(app: typer.Typer) -> None:
    """Register ``usage`` subcommands onto the parent ``agents`` Typer *app*."""
    usage_app = typer.Typer(
        name="usage",
        help="Token-usage reports from nat_runner.py outputs.",
        no_args_is_help=True,
    )
    app.add_typer(usage_app, rich_help_panel="Local commands")

    @usage_app.command(name="show")
    def show_cmd(
        ref: str = typer.Argument(
            ...,
            metavar="<PATH | FILESET_REF>",
            help="Local path to a result.json / run dir / nat-jobs dir, or a NeMo Platform fileset reference.",
        ),
        total_params: Optional[float] = typer.Option(
            None,
            "--total-params",
            callback=_validate_total_params,
            help="Model's total parameter count, in billions (e.g. 8.0 for Llama-3.1-8B, "
            "70.0 for Llama-3.1-70B).  When set with non-null tokens, "
            "compute_units = total_tokens × total_params.  Closed-source models have no "
            "public number — leave unset and compute_units stays null.",
        ),
        workspace: str = typer.Option(_DEFAULT_WORKSPACE, "--workspace", "-w"),
        base_url: BaseUrlOption = None,
    ) -> None:
        """Show a usage report for *ref*."""
        _show(
            ref,
            total_params=total_params,
            workspace=workspace,
            base_url=base_url,
        )


# ---------------------------------------------------------------------------
# Internal pipeline: resolve → parse → score → render
# ---------------------------------------------------------------------------


def _show(
    ref: str,
    *,
    total_params: float | None,
    workspace: str,
    base_url: str | None,
) -> None:
    try:
        report = _resolve_and_score(
            ref,
            total_params=total_params,
            workspace=workspace,
            base_url=base_url,
        )
    except (parser_module.UsageParseError, UsageSourceError, FilesetRefError, FilesetDownloadError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(render.render_json(report))


def _resolve_and_score(
    ref: str,
    *,
    total_params: float | None,
    workspace: str,
    base_url: str | None,
) -> UsageReport | BatchUsageReport:
    """End-to-end pipeline: source → parse → (rewrite if fileset) → score.

    Precedence (intentional, inverts the canonical ``classify_output_target``
    rule for *output* targets): if *ref* resolves to an existing local
    file or directory, the local path wins even if it would otherwise
    classify as a fileset name.  Path-shaped refs (``./foo``, ``/abs``)
    that don't exist locally fail fast rather than silently attempting a
    fileset download.  Bare names that don't exist locally fall through
    to fileset resolution.

    For local refs the parser's ``source_dir`` is already a stable local
    path.  For fileset refs the parser sees a tempdir that gets cleaned
    up when this function returns, so we rewrite ``source_dir`` to a
    synthetic ``<ref>/<rel>`` form *before* the tempdir disappears.
    """
    candidate = Path(ref).expanduser()
    if candidate.exists():
        with local_path(LocalDir(str(candidate))) as path:
            report = parser_module.parse_path(path)
        return _score_report(report, total_params=total_params)

    cls = classify_output_target(ref)
    if cls is LocalDir:
        # Path-shaped but missing — clearer error than a fileset 404.
        raise UsageSourceError(f"local path does not exist: {candidate}")

    # Only fileset refs contact the platform, so resolve/announce the target
    # (and attach auth) here rather than for purely-local reads above.
    sdk = _build_sdk(base_url=resolve_base_url(base_url))
    with fileset_path(FilesetRef(ref), sdk=sdk, workspace=workspace) as path:
        report = parser_module.parse_path(path)
        report = _rewrite_source_dirs(report, original_ref=ref, staged_root=path)
    return _score_report(report, total_params=total_params)


def _rewrite_source_dirs(
    report: UsageReport | BatchUsageReport,
    *,
    original_ref: str,
    staged_root: Path,
) -> UsageReport | BatchUsageReport:
    """Replace tempdir source_dirs with synthetic ``<ref>/<rel>`` paths.

    Called only on fileset-resolved reports — the staged tempdir is about
    to be cleaned up, so a literal local path would dangle.  The
    synthetic form keeps the per-task subdirectory hint visible to
    downstream consumers without lying about local-disk presence.
    """
    if isinstance(report, UsageReport):
        return report.model_copy(
            update={"task": _rewrite_task_source(report.task, original_ref=original_ref, staged_root=staged_root)}
        )
    new_runs = [_rewrite_task_source(t, original_ref=original_ref, staged_root=staged_root) for t in report.runs]
    return report.model_copy(update={"runs": new_runs})


def _rewrite_task_source(
    task: TaskUsage,
    *,
    original_ref: str,
    staged_root: Path,
) -> TaskUsage:
    try:
        rel = Path(task.source_dir).resolve().relative_to(staged_root.resolve())
    except ValueError:
        # source_dir resolved outside staged_root (e.g., a symlink in the
        # fileset pointing elsewhere).  Drop the per-task hint rather than
        # crash; the bare ref is still useful.
        logger.warning("source_dir %s is not under staged root %s; using bare ref", task.source_dir, staged_root)
        return task.model_copy(update={"source_dir": original_ref.rstrip("/")})
    new_src = original_ref.rstrip("/") if str(rel) == "." else f"{original_ref.rstrip('/')}/{rel}"
    return task.model_copy(update={"source_dir": new_src})


def _build_sdk(*, base_url: str) -> NemoClient:
    """Construct a typed platform client for fileset downloads.

    *base_url* is the value already resolved by ``resolve_base_url`` (flag /
    ``NEMO_BASE_URL`` > shared CLI config / ``NMP_BASE_URL`` > localhost), so
    don't re-read the env here — that would invert precedence.

    Attaches the CLI auth token from the shared context (the same
    ``Authorization: Bearer`` header the rest of the CLI sends) so fileset
    downloads succeed against a secured cluster.  Falls back to an
    unauthenticated client when no token is configured.
    """
    headers = resolve_context_headers()
    if headers:
        return NemoClient(base_url=base_url, default_headers=headers)
    return NemoClient(base_url=base_url)


def _score_report(
    report: UsageReport | BatchUsageReport,
    *,
    total_params: float | None,
) -> UsageReport | BatchUsageReport:
    """Populate ``compute_units`` on each task; return a new frozen report.

    Pydantic models are frozen, so we rebuild via ``model_copy(update=...)``.

    ``compute_units_total`` follows the same "totals are ``None`` if any
    run has missing usage" rule that ``parser.parse_path`` applies to
    token totals — partial sums are misleading, and the runs list still
    carries per-task values for any consumer that wants to inspect them.
    """
    if isinstance(report, UsageReport):
        scored_task = _scored_task(report.task, total_params=total_params)
        return report.model_copy(update={"task": scored_task})

    scored_runs = [_scored_task(t, total_params=total_params) for t in report.runs]
    # ``total_tokens_total`` is the parser's canonical "is this batch
    # complete?" signal — it's already nulled when any run has missing
    # usage, any sibling child was skipped, or any result.json was
    # unparseable.  Mirror that here so compute_units_total never gets
    # populated over a degraded batch (and never diverges from token totals).
    compute_values = [r.compute_units for r in scored_runs if r.compute_units is not None]
    compute_total: int | None
    if report.total_tokens_total is None or len(compute_values) != len(scored_runs):
        compute_total = None
    else:
        compute_total = sum(compute_values)
    return report.model_copy(
        update={
            "runs": scored_runs,
            "compute_units_total": compute_total,
        }
    )


def _scored_task(task: TaskUsage, *, total_params: float | None) -> TaskUsage:
    """Populate ``compute_units`` from the user-supplied total-params."""
    units = compute.compute_units_for(total_params, task.total_tokens)
    if units is None:
        return task
    return task.model_copy(update={"compute_units": units})
