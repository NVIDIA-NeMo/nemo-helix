# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job-result artifacts and fileset round-trips for the war-game.

Saves the run's Studio-facing results (mitigations, validation scorecard, composed workflow) and moves
suites/hitlogs through platform filesets (download an uploaded replay hitlog or benign suite; persist the
run's produced hitlog for later replay). All best-effort — capturing an artifact never fails the run.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from nemo_agent_hardener_plugin.api.v2.events import _events_path
from nemo_agent_hardener_plugin.filesets import download_fileset, upload_file_to_fileset
from nemo_agent_hardener_plugin.jobs.errors import CATEGORY_FILESET, AgentHardenerRunError
from nemo_helix_plugin.job_context import JobContext

logger = logging.getLogger(__name__)


@dataclass
class ArtifactCapture:
    """Exact output paths assigned to one native Agent Hardener invocation.

    The invocation uses a private manifest copy with a unique native
    ``storage.run_id``. An explicit shared ``garak.report_dir`` also receives a
    unique report prefix. This gives Platform exact ownership of the resulting
    files even when two jobs use the same local project at the same time.
    """

    run_dir: Path | None = None
    hitlog_dir: Path | None = None
    hitlog_prefix: str | None = None
    invocation_manifest: Path | None = None


def _storage_scope(data: dict[str, Any], manifest_path: Path, run_id: str) -> Path:
    """Pin native storage to *run_id* and return that exact run directory."""
    if "agent" in data:
        overrides = data.setdefault("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"Agent Hardener overrides must be a mapping: {manifest_path}")
        storage = overrides.get("storage")
        if isinstance(storage, str):
            storage = {"root_dir": storage}
        elif storage is None:
            storage = {}
        elif not isinstance(storage, dict):
            raise ValueError(f"Agent Hardener storage override must be a mapping: {manifest_path}")
        root = storage.get("root_dir") or ".agent-hardener"
        storage["run_id"] = run_id
        overrides["storage"] = storage
    else:
        storage = data.get("storage")
        if isinstance(storage, str):
            storage = {"root_dir": storage}
        elif storage is None and data.get("storage_dir"):
            storage = {"root_dir": data.pop("storage_dir")}
        elif not isinstance(storage, dict):
            raise ValueError(f"Agent Hardener config has no storage root: {manifest_path}")
        root = storage.get("root_dir")
        if not root:
            raise ValueError(f"Agent Hardener config has no storage root: {manifest_path}")
        storage["run_id"] = run_id
        data["storage"] = storage

    root_path = Path(str(root))
    if not root_path.is_absolute():
        root_path = manifest_path.parent / root_path
    return root_path / "run-logs" / run_id


def _garak_scope(data: dict[str, Any], run_dir: Path, run_id: str, manifest_path: Path) -> tuple[Path, str | None]:
    """Return the exact hitlog scope and isolate an explicit shared report directory."""
    garak = data.get("garak")
    if garak is None:
        return run_dir, None
    if not isinstance(garak, dict):
        raise ValueError(f"Agent Hardener garak config must be a mapping: {manifest_path}")
    report_dir = garak.get("report_dir")
    if not report_dir:
        return run_dir, None
    prefix = run_id
    garak["report_prefix"] = prefix
    return Path(str(report_dir)).resolve(), prefix


def _prepare_artifact_capture(capture: ArtifactCapture, manifest: str) -> str:
    """Create a run-scoped manifest and arm exact artifact capture.

    The copy stays beside the source manifest so every relative project path
    keeps its native meaning. Capture is best effort: if the copy cannot be
    prepared, the original manifest still runs and publication fails closed.
    """
    _cleanup_artifact_capture(capture)
    manifest_path = Path(manifest)
    scoped_path: Path | None = None
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Agent Hardener config root must be a mapping: {manifest_path}")
        run_id = f"nhx-agent-hardener-{uuid.uuid4().hex}"
        run_dir = _storage_scope(data, manifest_path, run_id)
        hitlog_dir, hitlog_prefix = _garak_scope(data, run_dir, run_id, manifest_path)
        scoped_path = manifest_path.with_name(f".{manifest_path.stem}.{run_id}.yaml")
        descriptor = os.open(scoped_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)
    except Exception:
        if scoped_path is not None:
            scoped_path.unlink(missing_ok=True)
        logger.warning("failed to scope Agent Hardener artifacts to this invocation", exc_info=True)
        return manifest
    capture.run_dir = run_dir
    capture.hitlog_dir = hitlog_dir
    capture.hitlog_prefix = hitlog_prefix
    capture.invocation_manifest = scoped_path
    return str(scoped_path)


def _cleanup_artifact_capture(capture: ArtifactCapture) -> None:
    """Remove the private invocation manifest and disarm capture."""
    if capture.invocation_manifest is not None:
        try:
            capture.invocation_manifest.unlink(missing_ok=True)
        except OSError:
            logger.warning("failed to remove scoped Agent Hardener manifest", exc_info=True)
    capture.run_dir = None
    capture.hitlog_dir = None
    capture.hitlog_prefix = None
    capture.invocation_manifest = None


def _hitlogs(capture: ArtifactCapture) -> list[Path]:
    """Return hitlogs owned by this invocation, newest first."""
    if capture.hitlog_dir is None or not capture.hitlog_dir.is_dir():
        return []
    if capture.hitlog_prefix is None:
        candidates = capture.hitlog_dir.rglob("*.hitlog.jsonl")
    else:
        candidates = (
            path
            for path in capture.hitlog_dir.iterdir()
            if path.is_file() and path.name.startswith(capture.hitlog_prefix) and path.name.endswith(".hitlog.jsonl")
        )
    return sorted(candidates, key=lambda path: path.stat().st_mtime_ns, reverse=True)


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


def _save_mitigations(ctx: JobContext, capture: ArtifactCapture) -> None:
    """Save the run's ``mitigations.json`` (before/after policy + workflow) as a job result for Studio.

    agent-hardener writes it under ``.agent-hardener/run-logs/<run_id>/`` at the end of a hardening run; the Studio
    Mitigations view fetches it via the results API. Best-effort — never fail the run over it.
    """
    try:
        path = capture.run_dir / "mitigations.json" if capture.run_dir is not None else None
        if path is not None and path.is_file():
            ctx.results.save("mitigations", path)
            logger.info("saved mitigations result from %s", path)
        else:
            logger.warning(
                "no mitigations.json in the invocation run directory %s; the Harden tab will be hidden",
                capture.run_dir,
            )
    except Exception:  # capturing the artifact is best-effort, not part of the war-game
        logger.warning("failed to save mitigations result", exc_info=True)


def _save_validation(ctx: JobContext, capture: ArtifactCapture) -> None:
    """Save the run's ``validation.json`` (per-item attack/benign results) as a job result for Studio.

    agent-hardener writes it under ``.agent-hardener/run-logs/<run_id>/`` for any run that ran validators — including
    the frozen validate-only sanity check. Drives the Studio scorecard. Best-effort — never fail the run.
    """
    try:
        path = capture.run_dir / "validation.json" if capture.run_dir is not None else None
        if path is not None and path.is_file():
            ctx.results.save("validation", path)
            logger.info("saved validation result from %s", path)
        else:
            logger.warning(
                "no validation.json in the invocation run directory %s; the run's scorecard will be unavailable",
                capture.run_dir,
            )
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


def _save_hitlog_fileset(sdk: Any, workspace: str, capture: ArtifactCapture) -> str:
    """Upload the run's produced garak hitlog to a fileset so a later run can replay it; return its ref.

    agent-hardener's attacker writes ``*.hitlog.jsonl`` run-scoped under ``.agent-hardener/run-logs/<run_id>/…/garak/``.
    Persist the newest hitlog in this invocation's exact run/prefix scope as a fileset and record its ref on the run
    entity. Best-effort — returns ``""`` on any failure (a run with no attack has no hitlog to save).
    """
    if sdk is None:
        return ""
    try:
        hitlogs = _hitlogs(capture)
        if not hitlogs:
            return ""
        return upload_file_to_fileset(sdk, hitlogs[0], workspace=workspace)
    except Exception:  # persisting the hitlog is best-effort, not part of the war-game
        logger.warning("failed to save hitlog fileset", exc_info=True)
        return ""
