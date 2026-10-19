# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge a run's task bundles into one bundle with the Docker (single agent-evaluate Job) layout.

M2 story 0's slice of M5 story 7. Files only: the bundle Job never writes records; the run
controller saves the run's result record from what this writes. The whole run is held in memory,
which is fine for hundreds of trials; M5 story 7 streams one task at a time.
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filesets import AsyncFilesetFileSystem, parse_fileset_ref
from nemo_evals.api.task_definitions.harbor import ResolvedHarborTaskDefinition
from nemo_evals.jobs.agent_spec import HarborRunnerTarget
from nemo_evals.jobs.harbor_scoring import harbor_scoring_task
from nemo_evals.taskset_runs.plan_storage import load_plan
from nemo_evals.taskset_runs.specs import AcceptedTaskOutput, TasksetBundleSpec, TasksetRunPlan
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.jobs.file_manager import AsyncFilesetFileManager
from nhx_evals_sdk.agent_eval.results import AgentEvalResult, AgentEvalSummary, RunMetadata
from nhx_evals_sdk.agent_eval.runtimes.harbor.scoring import harbor_scoring_metrics
from nhx_evals_sdk.agent_eval.scores import AgentEvalTaskScore
from nhx_evals_sdk.agent_eval.tasks import AgentEvalTask
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrial

#: Files read from each task bundle. Evidence stays where it is and is linked, not copied.
TASK_BUNDLE_FILES = ("trials.jsonl", "scores.jsonl", "metadata.json")


@dataclass(frozen=True)
class TaskBundle:
    """The rows of one task Job's bundle, with evidence refs pointing back at that bundle."""

    accepted: AcceptedTaskOutput
    trials: list[AgentEvalTrial]
    scores: list[AgentEvalTaskScore]
    metadata: RunMetadata | None


async def read_task_bundle(files: AsyncFilesClient, accepted: AcceptedTaskOutput, scratch: Path) -> TaskBundle:
    """Download one task bundle's rows; evidence refs become fileset refs into that bundle."""
    workspace, fileset, _ = parse_fileset_ref(accepted.bundle_ref, workspace_fallback=None)
    if not workspace or not fileset:
        raise ValueError(
            f"task {accepted.task_index}: bundle_ref {accepted.bundle_ref!r} is not workspace/fileset#path"
        )
    manager = AsyncFilesetFileManager(
        workspace=workspace,
        fileset_name=fileset,
        filesystem=AsyncFilesetFileSystem(client=files),
        ensure_fileset_exists=False,
    )
    local = scratch / f"task-{accepted.task_index}"
    local.mkdir(parents=True)
    for name in TASK_BUNDLE_FILES:
        await manager.download_from_url(f"{accepted.bundle_ref}/{name}", local_dir=local)
    return load_task_bundle(local, accepted)


def load_task_bundle(directory: Path, accepted: AcceptedTaskOutput) -> TaskBundle:
    """Parse a task bundle's rows from ``directory``; evidence refs become fileset refs into that bundle."""
    trials = [
        AgentEvalTrial.model_validate(_link_evidence(row, accepted.bundle_ref))
        for row in _rows(directory / "trials.jsonl")
    ]
    scores = [AgentEvalTaskScore.model_validate(row) for row in _rows(directory / "scores.jsonl")]
    metadata_path = directory / "metadata.json"
    metadata = (
        RunMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
    )
    return TaskBundle(accepted=accepted, trials=trials, scores=scores, metadata=metadata)


def merge_task_bundles(spec: TasksetBundleSpec, plan: TasksetRunPlan, parts: list[TaskBundle]) -> AgentEvalResult:
    """One ``AgentEvalResult`` over every task of the plan, from the accepted tasks' rows.

    The summary is ``AgentEvalSummary.from_scores`` over all rows, the same reducers a single
    Docker run over these tasks uses. Tasks without an accepted bundle are listed and counted
    in ``task_count`` but add no rows.
    """
    target = plan.spec.target
    if not isinstance(target, HarborRunnerTarget):
        raise ValueError("taskset bundles merge Harbor runs only")
    by_index = {part.accepted.task_index: part for part in parts}
    tasks: list[AgentEvalTask] = []
    for index, resolved in enumerate(plan.spec.tasks):
        if not isinstance(resolved.spec, ResolvedHarborTaskDefinition):
            raise ValueError(f"task {index} is not a Harbor task")
        task = harbor_scoring_task(resolved.spec, reward_key=target.reward_key)
        part = by_index.get(index)
        if part is not None:
            # As the Harbor runner does: declare every reward key the trials reported (format_ok, …).
            metrics = harbor_scoring_metrics(task, part.trials, reward_key=target.reward_key)
            task = task.model_copy(update={"metrics": metrics})
        tasks.append(task)
    trials = [trial for part in parts for trial in part.trials]
    scores = [_rename_run(score, spec.run_id) for part in parts for score in part.scores]
    summary = AgentEvalSummary.from_scores(scores, tasks=tasks, trials=trials)
    first = next((part.metadata for part in parts if part.metadata is not None), None)
    metadata = RunMetadata(
        labels=dict(plan.spec.labels),
        target=first.target if first is not None else None,
        started_at=spec.started_at,
        finished_at=spec.finished_at,
        duration_sec=(spec.finished_at - spec.started_at).total_seconds(),
        sdk_version=first.sdk_version if first is not None else None,
    )
    return AgentEvalResult(
        run_id=spec.run_id, tasks=tasks, trials=trials, scores=scores, summary=summary, metadata=metadata
    )


async def write_taskset_bundle(spec: TasksetBundleSpec, out_dir: Path, *, files: AsyncFilesClient) -> AgentEvalResult:
    """Read the plan and every accepted task bundle, then write the merged bundle to ``out_dir``."""
    plan = await load_plan(files, spec.plan_ref, expected_digest=spec.plan_digest)
    with tempfile.TemporaryDirectory(prefix="taskset-bundle-") as scratch:
        parts = [await read_task_bundle(files, accepted, Path(scratch)) for accepted in spec.accepted]
    result = merge_task_bundles(spec, plan, parts)
    result.persist(out_dir, write_dashboard=True)
    return result


def _rename_run(score: AgentEvalTaskScore, run_id: str) -> AgentEvalTaskScore:
    """Point a task Job's score at the taskset run: ``run_id`` and the run part of ``id``."""
    _, sep, rest = score.id.partition(":")
    return score.model_copy(update={"run_id": run_id, "id": f"{run_id}:{rest}" if sep else score.id})


def _link_evidence(row: dict[str, Any], bundle_ref: str) -> dict[str, Any]:
    """Make bundle-relative evidence refs fileset refs into the task bundle, which keeps the evidence."""
    for descriptor in ((row.get("evidence") or {}).get("descriptors") or {}).values():
        ref = descriptor.get("ref")
        if isinstance(ref, str) and ref and not ref.startswith("/") and "#" not in ref:
            descriptor["ref"] = f"{bundle_ref}/{ref}"
    return row


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
