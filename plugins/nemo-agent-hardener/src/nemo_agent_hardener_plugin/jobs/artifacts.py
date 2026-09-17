# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job-result artifacts and fileset round-trips for the war-game.

Saves the run's Studio-facing results (mitigations, validation scorecard, composed workflow) and moves
suites/hitlogs through platform filesets (download an uploaded replay hitlog or benign suite; persist the
run's produced hitlog for later replay). All best-effort — capturing an artifact never fails the run.
"""

from __future__ import annotations

import logging
from typing import Any

from nemo_agent_hardener_plugin.api.v2.events import _events_path
from nemo_agent_hardener_plugin.filesets import download_fileset, upload_file_to_fileset
from nemo_agent_hardener_plugin.jobs.errors import CATEGORY_FILESET, AgentHardenerRunError
from nemo_platform_plugin.job_context import JobContext

logger = logging.getLogger(__name__)


def _download_fileset(sdk: Any, ref: str, dest: Any, *, what: str) -> Any:
    """Download a fileset, classifying any transport/download failure as a :class:`fileset <AgentHardenerRunError>`."""
    try:
        return download_fileset(sdk, ref, dest)
    except AgentHardenerRunError:
        raise
    except Exception as exc:
        raise AgentHardenerRunError(CATEGORY_FILESET, f"could not download the {what} fileset {ref!r}: {exc}") from exc


def _replay_args(replay_hitlog_fileset: str | None, sdk: Any, ctx: JobContext) -> list[str]:
    """Resolve replay mode to `agent-hardener run` args: download the hitlog fileset and point `--replay` at it.

    Returns ``[]`` when not replaying. agent-hardener's ``--replay <path>`` skips the live garak attack and
    replays the recorded hits against the (defended) victim.
    """
    if not replay_hitlog_fileset:
        return []
    dest = _download_fileset(sdk, replay_hitlog_fileset, ctx.storage.persistent / "replay-hitlog", what="replay hitlog")
    hitlog = next((p for p in sorted(dest.rglob("*")) if p.is_file()), None)
    if hitlog is None:
        raise AgentHardenerRunError(
            CATEGORY_FILESET, f"Replay hitlog fileset {replay_hitlog_fileset!r} contained no file."
        )
    return ["--replay", str(hitlog)]


def _uploaded_benign_suite(benign_suite_fileset: str | None, sdk: Any, ctx: JobContext) -> str | None:
    """Download an uploaded benign-suite fileset and return its local CSV path, or ``None`` if not set."""
    if not benign_suite_fileset:
        return None
    dest = _download_fileset(
        sdk, benign_suite_fileset, ctx.storage.persistent / "benign-suite-upload", what="benign suite"
    )
    csv_file = next((p for p in sorted(dest.rglob("*")) if p.is_file()), None)
    if csv_file is None:
        raise AgentHardenerRunError(
            CATEGORY_FILESET, f"Benign suite fileset {benign_suite_fileset!r} contained no file."
        )
    return str(csv_file)


def _save_mitigations(ctx: JobContext) -> None:
    """Save the run's ``mitigations.json`` (before/after policy + workflow) as a job result for Studio.

    agent-hardener writes it under ``.agent-hardener/run-logs/<run_id>/`` at the end of a hardening run; the Studio
    Mitigations view fetches it via the results API. Best-effort — never fail the run over it.
    """
    run_logs = ctx.storage.persistent / ".agent-hardener" / "run-logs"
    try:
        candidates = sorted(
            run_logs.glob("*/mitigations.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            ctx.results.save("mitigations", candidates[0])
            logger.info("saved mitigations result from %s", candidates[0])
        else:
            # Not saving is indistinguishable from having nothing to save once the job's temp storage is
            # reclaimed, and the Harden tab simply never appears — so say which directory came up empty.
            logger.warning(
                "no mitigations.json under %s; the Harden tab will be hidden for this run (run-logs present: %s)",
                run_logs,
                sorted(p.name for p in run_logs.glob("*")) if run_logs.is_dir() else "<missing>",
            )
    except Exception:  # capturing the artifact is best-effort, not part of the war-game
        logger.warning("failed to save mitigations result", exc_info=True)


def _save_validation(ctx: JobContext) -> None:
    """Save the run's ``validation.json`` (per-item attack/benign results) as a job result for Studio.

    agent-hardener writes it under ``.agent-hardener/run-logs/<run_id>/`` for any run that ran validators — including
    the frozen validate-only sanity check. Drives the Studio scorecard. Best-effort — never fail the run.
    """
    run_logs = ctx.storage.persistent / ".agent-hardener" / "run-logs"
    try:
        candidates = sorted(
            run_logs.glob("*/validation.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            ctx.results.save("validation", candidates[0])
            logger.info("saved validation result from %s", candidates[0])
        else:
            logger.warning("no validation.json under %s; the run's scorecard will be unavailable", run_logs)
    except Exception:  # capturing the artifact is best-effort, not part of the war-game
        logger.warning("failed to save validation result", exc_info=True)


def _save_composed_guardrails(ctx: JobContext, defense_guardrails: str | None) -> None:
    """Persist the validated composed plugins.toml as a ``composed-guardrails`` job result (best-effort).

    Lets the Harden tab recover the exact guardrail set a sanity check validated after a page reload, so
    "Apply to Agent" stays available without re-running the check.
    """
    if not defense_guardrails:
        return
    try:
        path = ctx.storage.persistent / ".agent-hardener" / "composed-plugins.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(defense_guardrails, encoding="utf-8")
        ctx.results.save("composed-guardrails", path)
    except Exception:  # capturing the artifact is best-effort, not part of the war-game
        logger.warning("failed to save composed guardrails result", exc_info=True)


def _save_events_fileset(sdk: Any, *, workspace: str, run_name: str) -> str:
    """Upload the run's events.jsonl to a fileset; return its ref or '' on any failure."""
    path = _events_path(workspace, run_name)
    if not path.exists():
        return ""
    try:
        return upload_file_to_fileset(sdk, path, workspace=workspace)
    except Exception:
        logger.warning(
            "Failed to upload events.jsonl for run %r; history will not survive pod restart", run_name, exc_info=True
        )
        return ""


def _save_hitlog_fileset(sdk: Any, ctx: JobContext, workspace: str) -> str:
    """Upload the run's produced garak hitlog to a fileset so a later run can replay it; return its ref.

    agent-hardener's attacker writes ``*.hitlog.jsonl`` run-scoped under ``.agent-hardener/run-logs/<run_id>/…/garak/``.
    Persistent job storage is per-job, so we persist the newest hitlog as a fileset and record its ref on the
    run entity. Best-effort — returns ``""`` on any failure (a run with no attack has no hitlog to save).
    """
    if sdk is None:
        return ""
    try:
        hitlogs = sorted(
            (ctx.storage.persistent / ".agent-hardener" / "run-logs").rglob("*.hitlog.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not hitlogs:
            return ""
        return upload_file_to_fileset(sdk, hitlogs[0], workspace=workspace)
    except Exception:  # persisting the hitlog is best-effort, not part of the war-game
        logger.warning("failed to save hitlog fileset", exc_info=True)
        return ""
