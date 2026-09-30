# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal Evaluator submission and result-validation logic.

``workflow.py`` calls ``submit_and_validate()`` after it has uploaded the custom
environment and configured inference. This module translates the prepared
dataset and FileSet into an Evaluator job payload, waits for that job, downloads
its result archive, and verifies that the persisted trial and reward agree. It
is not a standalone developer entry point; run ``run.py`` instead.
"""

from __future__ import annotations

import math
import shutil
import tarfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from artifacts import read_jsonl, write_json
from nemo_evaluator.filesets import FilesetRef
from nemo_evaluator.jobs.agent_spec import GymRunnerTarget
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.inline import InlineMetricBundlePackager
from nemo_evaluator_sdk.agent_eval.runtimes.gym import GymRewardMetric, discover_gym_tasks
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.jobs.types import PlatformJobResponse

JOB_TIMEOUT_SECONDS = 1200
JOB_POLL_INTERVAL_SECONDS = 2
JOB_PROGRESS_INTERVAL_SECONDS = 30
TERMINAL_JOB_STATUSES = frozenset({"completed", "error", "cancelled"})


@dataclass(frozen=True, slots=True)
class EvaluationSubmission:
    """Inputs needed to submit one custom Gym evaluation."""

    base_url: str
    internal_api: str
    workspace: str
    fileset: str
    model: str
    dataset: Path
    resources_server: str
    evidence_directory: Path


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Validated identifiers and scores produced by the evaluation."""

    job_name: str
    verification: dict[str, Any]


def _post_evaluator_payload(
    sdk: NeMoPlatform,
    workspace: str,
    path: str,
    payload: Mapping[str, object],
) -> object:
    """Post an untyped payload to an Evaluator API route."""
    return sdk.post(
        f"/apis/evaluator/v2/workspaces/{workspace}/{path.lstrip('/')}",
        cast_to=object,
        body=dict(payload),
    )


def _require_job_name(payload: object) -> str:
    """Extract the submitted Platform job name from an API response."""
    job_name = payload.get("name") if isinstance(payload, Mapping) else None
    if not isinstance(job_name, str):
        raise TypeError(f"unexpected job response: {payload!r}")
    return job_name


def _wait_for_job(
    sdk: NeMoPlatform,
    job_name: str,
    workspace: str,
    progress: Callable[[str], None] | None = None,
) -> PlatformJobResponse:
    """Poll until terminal, reporting status changes and occasional heartbeats."""
    started_at = time.monotonic()
    deadline = started_at + JOB_TIMEOUT_SECONDS
    next_progress_at = started_at
    status_history: list[str] = []

    while True:
        job = sdk.jobs.retrieve(job_name, workspace=workspace)
        status = str(getattr(job.status, "value", job.status)).lower()
        status_changed = not status_history or status_history[-1] != status

        if status_changed:
            status_history.append(status)

        now = time.monotonic()
        if progress is not None and (status_changed or now >= next_progress_at):
            progress(f"{status} ({now - started_at:.0f}s elapsed)")
            next_progress_at = now + JOB_PROGRESS_INTERVAL_SECONDS

        if status in TERMINAL_JOB_STATUSES:
            return job

        if now >= deadline:
            history = " -> ".join(status_history)
            raise TimeoutError(
                f"job {job_name!r} did not finish within {JOB_TIMEOUT_SECONDS} seconds; "
                f"last status={status!r}, history={history}"
            )

        time.sleep(JOB_POLL_INTERVAL_SECONDS)


def _task_payload(task: Any, reward_metric: dict[str, Any]) -> dict[str, Any]:
    """Convert one discovered Gym task into the Evaluator API representation."""
    return {
        "id": task.id,
        "intent": task.intent,
        "inputs": task.inputs or {},
        "reference": task.reference or {},
        "metrics": [reward_metric],
        "metadata": [{"key": key, "value": value} for key, value in (task.metadata or {}).items()],
    }


def _output_texts(trial: dict[str, Any]) -> list[str]:
    """Extract assistant output text from a persisted Responses API trial."""
    response = trial.get("output", {}).get("response", {})
    return [
        content["text"]
        for output in response.get("output", [])
        if output.get("type") == "message"
        for content in output.get("content", [])
        if content.get("type") == "output_text"
    ]


def _build_job_payload(submission: EvaluationSubmission) -> dict[str, Any]:
    """Build the Evaluator job payload for one dataset task and one repetition."""
    discovered_tasks = discover_gym_tasks(submission.dataset)[:1]
    if len(discovered_tasks) != 1:
        raise RuntimeError(f"expected exactly one selected task, got {len(discovered_tasks)}")

    # Bundle the standard Gym reward metric inline so the task image can run it
    # without relying on a separately published metric package.
    reward_metric = bundle_metric(
        GymRewardMetric(),
        InlineMetricBundlePackager(),
    ).model_dump(mode="json")

    # The selected resources server is discovered from the package during
    # preparation; no environment-specific name is hard-coded here.
    target = GymRunnerTarget(
        environment=FilesetRef(root=f"{submission.workspace}/{submission.fileset}"),
        agent="simple_agent",
        agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
        resources_server=submission.resources_server,
        num_repeats=1,
        concurrency=1,
        hydra_params={
            "policy_base_url": (
                f"{submission.internal_api}/apis/inference-gateway/v2/workspaces/"
                f"{submission.workspace}/model/{submission.model}/-/v1"
            ),
            "policy_api_key": "not-used",
            "policy_model_name": submission.model,
        },
    )
    return {
        "spec": {
            "tasks": [_task_payload(discovered_tasks[0], reward_metric)],
            "target": target.model_dump(mode="json"),
        }
    }


def _download_results(
    sdk: NeMoPlatform,
    submission: EvaluationSubmission,
    job_name: str,
) -> Path:
    """Download and safely extract the evaluation result archive."""
    archive_path = submission.evidence_directory / "agent-eval-results.tar.gz"
    archive_response = sdk.jobs.results.download(
        "agent-eval-results",
        job=job_name,
        workspace=submission.workspace,
    )
    archive_contents = archive_response.read()
    if not archive_contents:
        raise RuntimeError(f"job {job_name} returned an empty result archive")
    archive_path.write_bytes(archive_contents)

    extraction_directory = submission.evidence_directory / "agent-eval-results"
    if extraction_directory.exists():
        shutil.rmtree(extraction_directory)
    extraction_directory.mkdir(parents=True)
    with tarfile.open(archive_path) as archive:
        # The data filter prevents archive members from escaping the destination.
        archive.extractall(extraction_directory, filter="data")
    return extraction_directory


def _validate_results(
    extraction_directory: Path,
    submission: EvaluationSubmission,
    job_name: str,
    job_status: str,
) -> dict[str, Any]:
    """Validate trial output, custom reward coverage, and score consistency."""
    trials_files = list(extraction_directory.rglob("trials.jsonl"))
    scores_files = list(extraction_directory.rglob("scores.jsonl"))
    if len(trials_files) != 1 or len(scores_files) != 1:
        raise RuntimeError(f"expected one trials.jsonl and one scores.jsonl, got {trials_files=} {scores_files=}")

    # This manual workflow deliberately evaluates one task once so evidence is
    # quick to inspect and inconsistencies have a single source.
    trials = read_jsonl(trials_files[0])
    scores = read_jsonl(scores_files[0])
    if len(trials) != 1 or len(scores) != 1:
        raise RuntimeError(f"expected one trial and one score, got {len(trials)=} {len(scores)=}")

    trial = trials[0]
    score = scores[0]
    if trial["error"] is not None:
        raise RuntimeError(f"trial failed: {trial['error']}")
    output_texts = _output_texts(trial)
    if not output_texts:
        raise RuntimeError(f"trial returned no assistant output text: {trial}")

    # The resources server's custom reward must be finite before comparing it
    # with the metric output persisted by Evaluator.
    trial_reward = trial.get("metadata", {}).get("reward")
    if not isinstance(trial_reward, int | float) or not math.isfinite(trial_reward):
        raise RuntimeError(f"trial reward is not finite: {trial_reward!r}")

    # Prove that the standard reward metric consumed the same custom reward.
    if score.get("status") != "completed":
        raise RuntimeError(f"metric did not complete: {score}")
    metric_outputs = score.get("outputs", [])
    if len(metric_outputs) != 1 or metric_outputs[0].get("name") != "reward":
        raise RuntimeError(f"unexpected metric outputs: {metric_outputs}")
    if metric_outputs[0].get("value") != trial_reward:
        raise RuntimeError(f"metric reward does not match trial reward: {metric_outputs[0]} != {trial_reward}")

    verification = {
        "job_name": job_name,
        "job_status": job_status,
        "trial_id": trial.get("id"),
        "trial_status": trial.get("status"),
        "score_status": score.get("status"),
        "reward": trial_reward,
        "resources_server": submission.resources_server,
        "output_texts": output_texts,
    }
    write_json(submission.evidence_directory / "verification.json", verification)
    return verification


def submit_and_validate(
    submission: EvaluationSubmission,
    *,
    sdk: NeMoPlatform | None = None,
    progress: Callable[[str], None] | None = None,
) -> EvaluationResult:
    """Run the Evaluator lifecycle and return only after evidence is consistent."""
    submission.evidence_directory.mkdir(parents=True, exist_ok=True)
    platform = sdk or NeMoPlatform(base_url=submission.base_url, max_retries=2)

    # Submit through the SDK's generic route until this plugin endpoint has a
    # generated typed method.
    payload = _build_job_payload(submission)
    response = _post_evaluator_payload(
        platform,
        submission.workspace,
        "agent-evaluate/jobs",
        payload,
    )
    job_name = _require_job_name(response)
    (submission.evidence_directory / "job-name.txt").write_text(
        job_name + "\n",
        encoding="utf-8",
    )

    # A failed job's server-side logs are included in the raised exception so a
    # developer does not need a second command to identify the failing stage.
    job = _wait_for_job(platform, job_name, submission.workspace, progress)
    if job.status.lower() != "completed":
        messages = [
            f"[{entry.timestamp}] {entry.message}"
            for entry in platform.jobs.get_logs(
                job_name,
                workspace=submission.workspace,
            )
        ]
        details = "\n".join(messages)
        raise RuntimeError(f"job did not complete: {job.status}\n{details}")

    # Completion alone is insufficient: validate the downloaded trial and score
    # before reporting that the custom environment worked.
    extraction_directory = _download_results(platform, submission, job_name)
    verification = _validate_results(
        extraction_directory,
        submission,
        job_name,
        job.status,
    )
    return EvaluationResult(job_name=job_name, verification=verification)
