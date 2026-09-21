# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manual NMP job matrix for Gym's declared agent/resources-server pairs."""

import json
import os
import tarfile
from collections import Counter
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from nemo_evaluator.jobs.agent_spec import GymRunnerTarget
from nemo_evaluator.shared.metric_bundles.bundles import bundle_metric
from nemo_evaluator.shared.metric_bundles.cloudpickle import CloudpickleMetricBundlePackager
from nemo_evaluator_sdk.agent_eval.runtimes.gym import GymRewardMetric, discover_gym_tasks
from nemo_platform import NeMoPlatform
from nmp.testing import short_unique_name
from nmp.testing.e2e import wait_for_platform_job

from e2e.test_evaluator_plugin import (
    _chat_completion,
    _cleanup_evaluator_job,
    _create_ready_mock_model,
    _internal_model_route,
    _post_evaluator_payload,
    _require_job_name,
)

# Checked-in source of the Gym agent/resource-server pairs exercised by this matrix.
GYM_MATRIX_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "packages/nemo_evaluator_sdk/tests/agent_eval/fixtures/gym_test_matrix_resource_servers.json"
)

# Opt-in switch that prevents normal E2E and CI runs from collecting this manual matrix.
GYM_MATRIX_RUN_ENV = "NEMO_GYM_RUN_PLATFORM_MATRIX"

# Optional comma-separated filter of resource servers, agents, or exact pair IDs.
GYM_MATRIX_CASES_ENV = "NEMO_GYM_MATRIX_CASES"

# Directory containing the example datasets extracted from the Gym task image.
GYM_MATRIX_RESOURCES_ROOT_ENV = "NEMO_GYM_RESOURCES_ROOT"

# Maximum time, in seconds, to wait for each submitted NMP job.
GYM_MATRIX_JOB_TIMEOUT_ENV = "NEMO_GYM_MATRIX_JOB_TIMEOUT"

# This matrix must remain explicitly enabled because a full run takes hours.
if os.environ.get(GYM_MATRIX_RUN_ENV) != "1":
    pytest.skip(
        "manual-only matrix; invoke through plugins/nemo-evaluator/scripts/gym-matrix/run.py "
        f"or set {GYM_MATRIX_RUN_ENV}=1",
        allow_module_level=True,
    )

pytestmark = [
    pytest.mark.container_only,
    pytest.mark.gym_matrix,
]


@dataclass(frozen=True)
class GymPlatformMatrixCase:
    """One agent/resource-server pair stored in the checked-in Gym matrix fixture.

    Attributes:
        resources_server: Resource-server package directory and matrix selection name.
        agent: Agent implementation connected to the resource server.
        agent_server: Composed Gym agent-server component name from the source config.
        resource_server: Gym resource-server component referenced by the agent.
        agent_config: Gym YAML path that declares the composed pair.
        selector: Optional Gym CLI selector for a specific configuration variant.
        model_servers: Additional model-server components required by the configuration.
        needs_gpu: Whether the pair requires a local GPU-backed component.
        rollout_skip_reason: Reason the pair cannot run in this local matrix, if any.
    """

    resources_server: str
    agent: str
    agent_server: str
    resource_server: str
    agent_config: str
    selector: str | None = None
    model_servers: tuple[str, ...] = ()
    needs_gpu: bool = False
    rollout_skip_reason: str | None = None

    @property
    def id(self) -> str:
        """Return the stable pytest ID for this pair."""
        return f"{self.resources_server}+{self.agent}"


@pytest.fixture(scope="module")
def evaluator_workspace(sdk: NeMoPlatform) -> Iterator[str]:
    """Create an isolated workspace shared by one matrix worker."""
    name = short_unique_name("e2e-gym-matrix")
    try:
        sdk.workspaces.create(name=name)
        yield name
    finally:
        # Workspace cleanup should not hide the original test failure.
        with suppress(Exception):
            sdk.workspaces.delete(name)


@pytest.fixture(scope="module")
def evaluator_sdk(sdk: NeMoPlatform, evaluator_workspace: str) -> Iterator[NeMoPlatform]:
    """Provide an SDK client scoped to the matrix workspace."""
    yield sdk.copy(workspace=evaluator_workspace, max_retries=2, timeout=900.0)


@pytest.fixture(scope="module")
def matrix_model_name(sdk: NeMoPlatform, evaluator_workspace: str) -> str:
    """Register the mock policy model reused by all cases in one worker."""
    model_name = short_unique_name("gym-matrix")
    _create_ready_mock_model(
        sdk,
        workspace=evaluator_workspace,
        name=model_name,
        mock_response_body=_chat_completion("NeMo Platform Gym matrix smoke response."),
    )
    return model_name


def _load_gym_matrix_cases() -> tuple[GymPlatformMatrixCase, ...]:
    """Load, validate, and optionally filter the checked-in matrix fixture."""
    raw = json.loads(GYM_MATRIX_MANIFEST.read_text(encoding="utf-8"))
    cases = tuple(
        GymPlatformMatrixCase(
            **{key: value for key, value in item.items() if key != "model_servers"},
            model_servers=tuple(item.get("model_servers", ())),
        )
        for item in raw
    )

    # Pair IDs are the stable identity used by pytest and the command-line filter.
    case_id_counts = Counter(case.id for case in cases)
    duplicate_ids = sorted(case_id for case_id, count in case_id_counts.items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"{GYM_MATRIX_MANIFEST} contains duplicate pair IDs: {duplicate_ids}")

    requested = {value.strip() for value in os.environ.get(GYM_MATRIX_CASES_ENV, "").split(",") if value.strip()}
    if not requested:
        return cases

    # A filter can select one exact pair or every pair for an agent or resource server.
    valid_filters = {value for case in cases for value in (case.id, case.resources_server, case.agent)}
    unknown = requested - valid_filters
    if unknown:
        raise ValueError(
            f"Unknown {GYM_MATRIX_CASES_ENV} values: {sorted(unknown)}. "
            f"Valid resource servers: {sorted({case.resources_server for case in cases})}"
        )

    return tuple(case for case in cases if requested.intersection((case.id, case.resources_server, case.agent)))


def _gym_matrix_resources_root() -> Path:
    """Resolve the directory containing Gym resource-server example datasets."""
    configured = os.environ.get(GYM_MATRIX_RESOURCES_ROOT_ENV)
    if not configured:
        raise RuntimeError(
            f"{GYM_MATRIX_RESOURCES_ROOT_ENV} is not set; invoke the matrix through "
            "plugins/nemo-evaluator/scripts/gym-matrix/run.py"
        )

    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{GYM_MATRIX_RESOURCES_ROOT_ENV}={configured!r} is not a directory")

    return root


def _gym_task_payloads(dataset: Path) -> list[dict[str, object]]:
    """Convert one bundled Gym example into an Evaluator task payload."""
    # One example is sufficient to prove the NMP job can exercise the resource server.
    tasks = discover_gym_tasks(dataset)[:1]
    if not tasks:
        raise ValueError(f"Gym discovered no tasks in {dataset}")

    bundled_reward = bundle_metric(GymRewardMetric(), CloudpickleMetricBundlePackager()).model_dump(mode="json")
    return [
        {
            "id": task.id,
            "intent": task.intent,
            "inputs": task.inputs or {},
            "reference": task.reference or {},
            "metrics": [bundled_reward],
            "metadata": [{"key": key, "value": value} for key, value in (task.metadata or {}).items()],
        }
        for task in tasks
    ]


def _gym_matrix_target(
    case: GymPlatformMatrixCase,
    *,
    workspace: str,
    model_name: str,
) -> GymRunnerTarget:
    """Build a Gym target from the exact pair configuration stored in the fixture."""
    hydra_params: dict[str, Any] = {
        "policy_base_url": _internal_model_route(workspace, model_name),
        "policy_api_key": "not-used-mock-provider",
        "policy_model_name": model_name,
    }

    # Judge and reward models use the same deterministic mock route for this launch smoke test.
    for model_server in case.model_servers:
        if case.resources_server == "genrm_compare" and model_server == "genrm_model":
            hydra_params["genrm_model_name"] = "stub-model"
            continue

        hydra_params[model_server] = {
            "responses_api_models": {
                "inference_provider": {
                    "entrypoint": "app.py",
                    "base_url": "${policy_base_url}",
                    "api_key": "${policy_api_key}",
                    "model": "${policy_model_name}",
                }
            }
        }

    return GymRunnerTarget(
        agent=case.agent_server,
        agent_config=case.agent_config,
        resources_server=case.selector or case.resources_server,
        bind_resources_server=False,
        num_repeats=1,
        concurrency=1,
        startup_timeout_s=300,
        collection_timeout_s=300,
        hydra_params=hydra_params,
    )


def _submit_gym_matrix_job(
    sdk: NeMoPlatform,
    workspace: str,
    case: GymPlatformMatrixCase,
    *,
    model_name: str,
    resources_root: Path,
) -> str:
    """Submit one matrix pair through the NMP Evaluator API."""
    dataset = resources_root / case.resources_server / "data" / "example.jsonl"
    if not dataset.is_file():
        raise FileNotFoundError(f"{case.id} has no bundled example dataset at {dataset}")

    target = _gym_matrix_target(case, workspace=workspace, model_name=model_name)
    payload = _post_evaluator_payload(
        sdk,
        workspace,
        "agent-evaluate/jobs",
        {"spec": {"tasks": _gym_task_payloads(dataset), "target": target.model_dump(mode="json")}},
    )
    return _require_job_name(payload)


def _validate_gym_matrix_result(
    sdk: NeMoPlatform,
    workspace: str,
    job_name: str,
    case: GymPlatformMatrixCase,
    output_dir: Path,
) -> None:
    """Validate that a completed job persisted one successful Gym trial."""
    # Download and unpack the first-party result bundle produced by the Evaluator job.
    case_dir = output_dir / case.id.replace("/", "_")
    case_dir.mkdir(parents=True, exist_ok=True)

    archive_path = case_dir / "agent-eval-results.tar.gz"
    sdk.jobs.results.download("agent-eval-results", job=job_name, workspace=workspace).write_to_file(archive_path)

    extract_dir = case_dir / "results"
    with tarfile.open(archive_path) as archive:
        archive.extractall(extract_dir, filter="data")

    # A single input and one repeat must produce exactly one persisted trial.
    trials_files = list(extract_dir.rglob("trials.jsonl"))
    if not trials_files:
        raise AssertionError(f"{case.id}: trials.jsonl missing from job {job_name}")

    trials = [json.loads(line) for line in trials_files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(trials) != 1:
        raise AssertionError(f"{case.id}: expected one trial from job {job_name}, got {len(trials)}")

    # Completion and a numeric Gym reward prove that the resource server handled the rollout.
    trial = trials[0]
    if trial.get("status") != "completed":
        raise AssertionError(f"{case.id}: trial ended {trial.get('status')!r}: {trial}")

    reward = (trial.get("metadata") or {}).get("reward")
    if reward is None:
        raise AssertionError(f"{case.id}: completed trial has no Gym reward")
    if not isinstance(reward, int | float):
        raise AssertionError(f"{case.id}: Gym reward is not numeric: {reward!r}")


def _gym_matrix_job_diagnostics(sdk: NeMoPlatform, workspace: str, job_name: str) -> str:
    """Collect bounded job status and log diagnostics for an assertion failure."""
    details: list[str] = []

    # Diagnostics are best-effort, but retrieval failures should remain visible.
    try:
        status = sdk.jobs.get_status(workspace=workspace, name=job_name)
        details.append(status.model_dump_json(indent=2))
    except Exception as error:
        details.append(f"Could not retrieve job status: {error}")

    try:
        logs = sdk.jobs.get_logs(workspace=workspace, name=job_name)
        details.extend(f"[{entry.job_step}] {entry.message}" for entry in (logs.data or [])[-20:])
    except Exception as error:
        details.append(f"Could not retrieve job logs: {error}")

    return "\n".join(details)[-8000:]


def _matrix_skip_reason(case: GymPlatformMatrixCase) -> str | None:
    """Return why a case cannot run against the local CPU and mock-policy setup."""
    if case.needs_gpu:
        return "requires a GPU-backed Gym model server"

    if case.rollout_skip_reason:
        return case.rollout_skip_reason

    return None


# Cases loaded at import time become the pytest parameter source.
GYM_MATRIX_CASES = _load_gym_matrix_cases()

# Parameters attach skip marks before fixtures can submit any unsupported job.
GYM_MATRIX_PARAMETERS = [
    pytest.param(case, id=case.id, marks=pytest.mark.skip(reason=reason))
    if (reason := _matrix_skip_reason(case))
    else pytest.param(case, id=case.id)
    for case in GYM_MATRIX_CASES
]


@pytest.mark.parametrize("case", GYM_MATRIX_PARAMETERS)
@pytest.mark.timeout(1800)
def test_gym_resource_server_job_matrix(
    evaluator_sdk: NeMoPlatform,
    evaluator_workspace: str,
    matrix_model_name: str,
    tmp_path: Path,
    case: GymPlatformMatrixCase,
) -> None:
    """Run one declared Gym pair as an NMP job and validate its persisted trial."""
    job_timeout = float(os.environ.get(GYM_MATRIX_JOB_TIMEOUT_ENV, "900"))
    if job_timeout <= 0:
        raise ValueError(f"{GYM_MATRIX_JOB_TIMEOUT_ENV} must be positive")

    # Submission must use the pair-specific config and example dataset from the fixture.
    job_name = _submit_gym_matrix_job(
        evaluator_sdk,
        evaluator_workspace,
        case,
        model_name=matrix_model_name,
        resources_root=_gym_matrix_resources_root(),
    )

    try:
        # Platform completion alone is insufficient; the persisted Gym trial must also be valid.
        job = wait_for_platform_job(evaluator_sdk, job_name, evaluator_workspace, timeout=job_timeout)
        assert job.status.lower() == "completed", (
            f"{case.id}: job {job_name} ended {job.status!r}\n"
            f"{_gym_matrix_job_diagnostics(evaluator_sdk, evaluator_workspace, job_name)}"
        )
        _validate_gym_matrix_result(
            evaluator_sdk,
            evaluator_workspace,
            job_name,
            case,
            tmp_path,
        )
    finally:
        # Always release the job container before the next sequential matrix case starts.
        _cleanup_evaluator_job(evaluator_sdk, job_name)
