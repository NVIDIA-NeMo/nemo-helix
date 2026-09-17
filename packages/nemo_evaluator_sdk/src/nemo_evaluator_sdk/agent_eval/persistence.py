# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistence helpers for standalone agent-eval result bundles."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from nemo_evaluator_sdk.agent_eval.dashboard import write_dashboard
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult, BundleLocation
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial
from pydantic import BaseModel

#: Filename of the rendered HTML dashboard inside a bundle.
DASHBOARD_FILENAME = "report.html"


def persist_run(
    result: AgentEvalResult,
    output_dir: str | Path,
    *,
    write_html_dashboard: bool = True,
) -> BundleLocation:
    """Write a completed run to a bundle at ``output_dir`` and report where it landed.

    Explicit rather than a side effect of :meth:`AgentEvaluator.run`: computing an evaluation and
    storing one are different decisions, and folding them together is what forced the result object to
    carry paths it could not know at construction time. (Same reasoning as ``publish_to_intake``.)

    Set ``write_html_dashboard=False`` to skip rendering ``report.html`` — the dashboard is written
    here so the manifest can record it in a single pass.
    """
    validated_trials = [AgentEvalTrial.model_validate(trial) for trial in result.trials]
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    # Render first so the manifest below can name it; the dashboard reads only the run's own contents.
    dashboard_path = write_dashboard(result, path / DASHBOARD_FILENAME) if write_html_dashboard else None

    _write_json(path / "metadata.json", result.metadata)
    _write_jsonl(path / "tasks.jsonl", result.tasks)
    _write_trials(path / "trials.jsonl", validated_trials, base=path)
    _write_jsonl(path / "scores.jsonl", result.scores)
    _write_json(path / "summary.json", result.summary)

    location = BundleLocation(output_dir=path, dashboard_path=dashboard_path)
    _write_json(path / "run.json", _run_manifest(result, location))
    return location


def _run_manifest(result: AgentEvalResult, location: BundleLocation) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "output_dir": str(location.output_dir),
        "dashboard_path": str(location.dashboard_path) if location.dashboard_path is not None else None,
        "artifacts": {
            "metadata": "metadata.json",
            "tasks": "tasks.jsonl",
            "trials": "trials.jsonl",
            "scores": "scores.jsonl",
            "summary": "summary.json",
        },
    }


def _write_json(path: Path, value: BaseModel | dict[str, Any]) -> None:
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json")
    else:
        payload = value
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[BaseModel]) -> None:
    # Stream row-by-row instead of joining the whole payload in memory first.
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.model_dump(mode="json"), sort_keys=True))
            handle.write("\n")


def _write_trials(path: Path, trials: Sequence[AgentEvalTrial], *, base: Path) -> None:
    """Write trials, rewriting evidence refs bundle-relative so the bundle is self-contained.

    A trial's evidence lives under the bundle (``<output_dir>/evidence/...``); storing the ref relative to
    the bundle (rather than the launch CWD) means a moved or copied bundle re-scores without any path fixups.
    Refs that point outside the bundle (rare) are left verbatim.
    """
    resolved_base = base.resolve()
    with path.open("w", encoding="utf-8") as handle:
        row: dict[str, Any]
        for trial in trials:
            row = trial.model_dump(mode="json")
            for descriptor in ((row.get("evidence") or {}).get("descriptors") or {}).values():
                descriptor["ref"] = _relativize_ref(descriptor.get("ref"), resolved_base)
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _relativize_ref(ref: str | None, base: Path) -> str | None:
    """Make an evidence ref relative to the bundle dir when it lives under it; else leave it verbatim."""
    if not ref:
        return ref
    try:
        return Path(ref).resolve().relative_to(base).as_posix()
    except ValueError:
        return ref  # evidence written outside the bundle — cannot relativize


def read_trials(run_dir: str | Path) -> list[AgentEvalTrial]:
    """Read the persisted trials of a run bundle — the inverse of the ``trials.jsonl`` ``persist_run`` writes.

    Each row is loaded back into an ``AgentEvalTrial`` with its evidence pointing at the on-disk
    deliverables, so a stored run can be **re-scored** — ``AgentEvaluator().run(tasks=…, trials=…)`` with
    fresh metrics/judge — without re-running the agent. Evidence refs are resolved relative to ``run_dir``
    when the stored (launch-relative) ref no longer resolves, so a moved or copied bundle still works.
    """
    directory = Path(run_dir)
    trials: list[AgentEvalTrial] = []
    for row in _read_jsonl(directory / "trials.jsonl"):
        for descriptor in ((row.get("evidence") or {}).get("descriptors") or {}).values():
            descriptor["ref"] = _resolve_evidence_ref(directory, descriptor.get("ref"))
        trials.append(AgentEvalTrial.model_validate(row))
    return trials


def _resolve_evidence_ref(run_dir: Path, ref: str | None) -> str | None:
    """Resolve a persisted evidence ref against the bundle dir.

    Self-contained bundles store refs relative to the bundle (see ``_write_trials``), so those resolve
    directly under ``run_dir``. Falls back for still-valid absolute refs, and for moved bundles / legacy
    absolute refs (rebuilt under ``run_dir`` from the ``evidence/`` tail).
    """
    if not ref:
        return ref
    base = run_dir.resolve()
    candidate = Path(ref)
    if not candidate.is_absolute():
        rebuilt = run_dir / candidate
        if _resolves_within(base, rebuilt):
            return str(rebuilt)
    elif candidate.exists():
        return ref
    parts = candidate.parts
    if "evidence" in parts:
        rebuilt = run_dir / Path(*parts[parts.index("evidence") :])
        if _resolves_within(base, rebuilt):
            return str(rebuilt)
    return ref


def _resolves_within(base: Path, path: Path) -> bool:
    """Whether ``path`` exists and stays inside ``base`` after resolving — no ``..``/symlink escape.

    Rebuilt refs are joined onto ``run_dir``; a bundle is designed to be moved/copied, so a ref with ``..``
    (or a symlink) must not be allowed to point loaded evidence outside the bundle it was loaded from.
    """
    resolved = path.resolve()
    if not resolved.exists():
        return False
    return resolved == base or base in resolved.parents


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)
