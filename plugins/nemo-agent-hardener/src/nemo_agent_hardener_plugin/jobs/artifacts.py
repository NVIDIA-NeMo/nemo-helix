# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job-result artifacts and fileset round-trips for the war-game.

Saves the run's Studio-facing results (mitigations, validation scorecard, composed workflow) and moves
suites/hitlogs through platform filesets (download an uploaded replay hitlog or benign suite; persist the
run's produced hitlog for later replay). All best-effort — capturing an artifact never fails the run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from nemo_agent_hardener_plugin.api.v2.events import _events_path
from nemo_agent_hardener_plugin.filesets import download_fileset, upload_file_to_fileset
from nemo_agent_hardener_plugin.jobs.errors import CATEGORY_FILESET, AgentHardenerRunError
from nemo_helix_plugin.job_context import JobContext

logger = logging.getLogger(__name__)

_FileState = tuple[int, int, int, int, int]


@dataclass
class ArtifactCapture:
    """Artifact state captured immediately before one native war-game invocation.

    ``active`` remains false until the manifest is fully materialized and its
    artifact roots have been snapshotted. This makes every earlier failure
    fail closed instead of publishing files unpacked during setup or left by a
    previous local-config job.
    """

    active: bool = False
    run_logs: Path | None = None
    explicit_hitlog_dir: Path | None = None
    before: dict[Path, _FileState] = field(default_factory=dict)


def _file_state(path: Path) -> _FileState:
    """Return enough stat identity to detect replacement and in-place updates."""
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _hitlog_files(run_logs: Path, explicit_hitlog_dir: Path | None) -> set[Path]:
    """Enumerate hitlogs with native AgentBreaker's explicit-root precedence and depth."""
    if explicit_hitlog_dir is not None:
        return set(explicit_hitlog_dir.glob("*.hitlog.jsonl"))
    return set(run_logs.rglob("*.hitlog.jsonl"))


def _artifact_files(run_logs: Path, explicit_hitlog_dir: Path | None) -> set[Path]:
    """Enumerate artifacts relevant to later Platform publication."""
    return {
        *run_logs.glob("*/mitigations.json"),
        *run_logs.glob("*/validation.json"),
        *_hitlog_files(run_logs, explicit_hitlog_dir),
    }


def _snapshot(run_logs: Path, explicit_hitlog_dir: Path | None) -> dict[Path, _FileState]:
    """Capture file identities, failing closed if the tree changes during the scan."""
    return {path: _file_state(path) for path in _artifact_files(run_logs, explicit_hitlog_dir)}


def _explicit_hitlog_dir(manifest: str) -> Path | None:
    """Resolve a thin manifest's explicit ``garak.report_dir`` as native Agent Hardener does.

    Native Agent Hardener passes this value to :class:`pathlib.Path` and then
    resolves it from the process working directory. It does not resolve this
    setting relative to the manifest, unlike ``storage.root_dir``.
    """
    manifest_path = Path(manifest)
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Agent Hardener config root must be a mapping: {manifest_path}")
    garak = data.get("garak")
    report_dir = garak.get("report_dir") if isinstance(garak, dict) else None
    return Path(report_dir).resolve() if report_dir else None


def _refresh_artifact_capture(capture: ArtifactCapture, ctx: JobContext, manifest: str) -> None:
    """Arm capture after manifest setup and immediately before the native invocation.

    Always replace the snapshot, even if the resolved storage root is
    unchanged. A saved project bundle can unpack historical run logs into the
    job root while it is materialized, and those files must be treated as
    pre-existing rather than output from this invocation.
    """
    capture.active = False
    try:
        run_logs = _run_logs_dir(ctx, manifest)
        explicit_hitlog_dir = _explicit_hitlog_dir(manifest)
        before = _snapshot(run_logs, explicit_hitlog_dir)
    except Exception:
        logger.warning("failed to snapshot Agent Hardener artifact roots", exc_info=True)
        capture.run_logs = None
        capture.explicit_hitlog_dir = None
        capture.before = {}
        return
    capture.run_logs = run_logs
    capture.explicit_hitlog_dir = explicit_hitlog_dir
    capture.before = before
    capture.active = True


def _changed_files(capture: ArtifactCapture, paths: set[Path]) -> list[Path]:
    """Filter candidate paths to files created or changed since *capture*, newest first."""
    if not capture.active or capture.run_logs is None:
        return []
    changed: dict[Path, int] = {}
    for path in paths:
        try:
            state = _file_state(path)
        except OSError:
            continue
        if capture.before.get(path) != state:
            changed[path] = state[3]
    return [path for path, _ in sorted(changed.items(), key=lambda item: item[1], reverse=True)]


def _changed_artifacts(capture: ArtifactCapture, pattern: str) -> list[Path]:
    """Return changed storage-root artifacts matching *pattern*, newest first."""
    if capture.run_logs is None:
        return []
    return _changed_files(capture, set(capture.run_logs.glob(pattern)))


def _changed_hitlogs(capture: ArtifactCapture) -> list[Path]:
    """Return changed hitlogs using native recursive/default and direct/explicit semantics."""
    if capture.run_logs is None:
        return []
    return _changed_files(capture, _hitlog_files(capture.run_logs, capture.explicit_hitlog_dir))


def _run_logs_dir(ctx: JobContext, manifest: str | None = None) -> Path:
    """Return the native run-log directory for *manifest*.

    Saved manifests are materialized in the job's persistent directory, but a
    local ``--config`` keeps its own base directory. Agent Hardener resolves a
    relative storage root against that manifest directory, so artifact
    discovery must use the same base.
    """
    if manifest is None:
        return ctx.storage.persistent / ".agent-hardener" / "run-logs"

    manifest_path = Path(manifest)
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Agent Hardener config root must be a mapping: {manifest_path}")

    if "agent" in data:
        overrides = data.get("overrides") or {}
        storage = overrides.get("storage") if isinstance(overrides, dict) else None
        root = storage.get("root_dir") if isinstance(storage, dict) else storage
        root = root or ".agent-hardener"
    else:
        storage = data.get("storage")
        if isinstance(storage, dict):
            root = storage.get("root_dir")
        elif isinstance(storage, str):
            root = storage
        else:
            root = None
        if root is None and storage is None:
            root = data.get("storage_dir")
        if not root:
            raise ValueError(f"Agent Hardener config has no storage root: {manifest_path}")

    root_path = Path(str(root))
    if not root_path.is_absolute():
        root_path = manifest_path.parent / root_path
    return root_path / "run-logs"


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
    Mitigations view fetches it via the results API. Only files changed since this invocation's capture are eligible.
    Best-effort — never fail the run over it.
    """
    try:
        run_logs = capture.run_logs
        candidates = _changed_artifacts(capture, "*/mitigations.json")
        if candidates:
            ctx.results.save("mitigations", candidates[0])
            logger.info("saved mitigations result from %s", candidates[0])
        else:
            # Not saving is indistinguishable from having nothing to save once the job's temp storage is
            # reclaimed, and the Harden tab simply never appears — so say which directory came up empty.
            logger.warning(
                "no new or changed mitigations.json under %s; the Harden tab will be hidden for this run "
                "(run-logs present: %s)",
                run_logs,
                sorted(p.name for p in run_logs.glob("*"))
                if run_logs is not None and run_logs.is_dir()
                else "<missing>",
            )
    except Exception:  # capturing the artifact is best-effort, not part of the war-game
        logger.warning("failed to save mitigations result", exc_info=True)


def _save_validation(ctx: JobContext, capture: ArtifactCapture) -> None:
    """Save the run's ``validation.json`` (per-item attack/benign results) as a job result for Studio.

    agent-hardener writes it under ``.agent-hardener/run-logs/<run_id>/`` for any run that ran validators — including
    the frozen validate-only sanity check. Only files changed since this invocation's capture are eligible. Drives the
    Studio scorecard. Best-effort — never fail the run.
    """
    try:
        run_logs = capture.run_logs
        candidates = _changed_artifacts(capture, "*/validation.json")
        if candidates:
            ctx.results.save("validation", candidates[0])
            logger.info("saved validation result from %s", candidates[0])
        else:
            logger.warning(
                "no new or changed validation.json under %s; the run's scorecard will be unavailable", run_logs
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


def _save_hitlog_fileset(sdk: Any, ctx: JobContext, workspace: str, capture: ArtifactCapture) -> str:
    """Upload the run's produced garak hitlog to a fileset so a later run can replay it; return its ref.

    agent-hardener's attacker writes ``*.hitlog.jsonl`` run-scoped under ``.agent-hardener/run-logs/<run_id>/…/garak/``.
    Local configs can reuse their project artifact directory across jobs, so only a file changed since this
    invocation's capture is eligible. Persist the newest eligible hitlog as a fileset and record its ref on the run
    entity. Best-effort — returns ``""`` on any failure (a run with no attack has no hitlog to save).
    """
    if sdk is None:
        return ""
    try:
        hitlogs = _changed_hitlogs(capture)
        if not hitlogs:
            return ""
        return upload_file_to_fileset(sdk, hitlogs[0], workspace=workspace)
    except Exception:  # persisting the hitlog is best-effort, not part of the war-game
        logger.warning("failed to save hitlog fileset", exc_info=True)
        return ""
