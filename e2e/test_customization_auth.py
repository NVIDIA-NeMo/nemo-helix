# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Customization job steps can call the platform from their pods when auth is enabled.

Each test submits a job and waits for its CPU download step to complete, then cancels
it before the GPU training step. Requires an auth-enabled Kubernetes platform.
"""

import json
import time
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.oidc import discover_nhx_config
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetPurpose
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.schemas import HelixJobStatus, HelixJobStatusResponse
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import CreateModelEntityRequest
from nhx.customization_common.sdk.client import CustomizationClient
from nhx.customization_common.sdk.types import CustomizationJobCreateRequest

pytestmark = [
    pytest.mark.customization_auth_e2e,
    pytest.mark.container_only,
    pytest.mark.timeout(900),
    pytest.mark.e2e_config("e2e/configs/local-subprocess.yaml", {"auth": {"enabled": True}}),
]

DOWNLOAD_STEP = "model-and-dataset-download"
DOWNLOAD_TIMEOUT_SECONDS = 600
POLL_INTERVAL_SECONDS = 5
_FAILED_STATUSES = {HelixJobStatus.ERROR, HelixJobStatus.CANCELLED}

# The download step copies files without loading them, so placeholders suffice.
_MODEL_FILES = {"config.json": json.dumps({"model_type": "qwen3"}), "model.safetensors": "placeholder"}
_SFT_ROW = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
_DPO_ROW = {"prompt": "hi", "chosen_response": "hello", "rejected_response": "go away"}

SpecBuilder = Callable[[str, str, str, str], dict[str, Any]]


def _automodel_spec(workspace: str, model: str, dataset: str, output: str) -> dict[str, Any]:
    return {
        "model": f"{workspace}/{model}",
        "dataset": {"training": f"{workspace}/{dataset}"},
        "training": {"training_type": "sft", "finetuning_type": "lora", "max_seq_length": 512},
        "schedule": {"epochs": 1, "max_steps": 1},
        "batch": {"global_batch_size": 1, "micro_batch_size": 1},
        "optimizer": {"learning_rate": 1e-4},
        "parallelism": {"num_nodes": 1, "num_gpus_per_node": 1},
        "output": {"name": output},
    }


def _unsloth_spec(workspace: str, model: str, dataset: str, output: str) -> dict[str, Any]:
    return {
        "model": {"name": f"{workspace}/{model}", "max_seq_length": 512},
        "dataset": {"path": f"{workspace}/{dataset}"},
        "schedule": {"max_steps": 1},
        "output": {"name": output},
    }


def _rl_dpo_spec(workspace: str, model: str, dataset: str, output: str) -> dict[str, Any]:
    return {
        "model": f"{workspace}/{model}",
        "dataset": f"{workspace}/{dataset}",
        "training": {
            "type": "dpo",
            "max_steps": 1,
            "batch_size": 1,
            "micro_batch_size": 1,
            "max_seq_length": 512,
            "parallelism": {"num_nodes": 1, "num_gpus_per_node": 1},
        },
        "output": {"name": output},
    }


BACKENDS: list[tuple[str, SpecBuilder, dict[str, Any]]] = [
    ("automodel", _automodel_spec, _SFT_ROW),
    ("unsloth", _unsloth_spec, _SFT_ROW),
    ("rl", _rl_dpo_spec, _DPO_ROW),
]


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _create_fileset(client: NemoClient, workspace: str, purpose: FilesetPurpose, files: dict[str, str]) -> str:
    name = _unique(f"e2e-{purpose.value}")
    files_client = FilesClient.from_client(client)
    files_client.create_fileset(workspace=workspace, body=CreateFilesetRequest(name=name, purpose=purpose))
    for path, content in files.items():
        files_client.upload_file(workspace=workspace, name=name, path=path, content=content.encode())
    return name


@pytest.fixture(scope="module", autouse=True)
def _require_auth_enabled(client: NemoClient) -> None:
    """Fail rather than pass vacuously when the platform runs without auth."""
    assert discover_nhx_config(str(client.base_url)).auth_enabled, f"platform auth is not enabled at {client.base_url}"


@pytest.fixture
def model_entity(client: NemoClient, workspace: str) -> str:
    fileset = _create_fileset(client, workspace, FilesetPurpose.MODEL, _MODEL_FILES)
    name = _unique("e2e-model")
    ModelsClient.from_client(client).create_model(
        workspace=workspace, body=CreateModelEntityRequest(name=name, fileset=f"{workspace}/{fileset}")
    )
    return name


@pytest.fixture
def submitted_jobs(client: NemoClient, workspace: str) -> Iterator[list[str]]:
    """Job names to cancel on teardown."""
    names: list[str] = []
    yield names
    jobs = JobsClient.from_client(client)
    for name in names:
        try:
            jobs.cancel_job(workspace=workspace, name=name)
        except Exception:
            pass  # Best-effort; the workspace is deleted anyway


def _wait_for_download_step(client: NemoClient, workspace: str, job_name: str) -> HelixJobStatusResponse:
    jobs = JobsClient.from_client(client)
    deadline = time.monotonic() + DOWNLOAD_TIMEOUT_SECONDS
    status = jobs.get_job_status(workspace=workspace, name=job_name).data()
    while time.monotonic() < deadline:
        step = next((s for s in status.steps if s.name == DOWNLOAD_STEP), None)
        if step is not None and step.status == HelixJobStatus.COMPLETED:
            return status
        if status.status in _FAILED_STATUSES or (step is not None and step.status in _FAILED_STATUSES):
            pytest.fail(_failure_details(jobs, workspace, job_name, status))
        time.sleep(POLL_INTERVAL_SECONDS)
        status = jobs.get_job_status(workspace=workspace, name=job_name).data()
    pytest.fail(
        f"{DOWNLOAD_STEP} did not complete within {DOWNLOAD_TIMEOUT_SECONDS}s.\n"
        + _failure_details(jobs, workspace, job_name, status)
    )


def _failure_details(jobs: JobsClient, workspace: str, job_name: str, status: HelixJobStatusResponse) -> str:
    details = [f"Job {workspace}/{job_name} status:", status.model_dump_json(indent=2)]
    try:
        logs = list(jobs.list_job_logs(workspace=workspace, name=job_name).items())
        details.append("Last job logs:")
        details.extend(f"  [{entry.job_step}] {entry.message}" for entry in logs[-40:])
    except Exception as exc:
        details.append(f"Could not fetch job logs: {exc}")
    return "\n".join(details)


@pytest.mark.parametrize(("backend", "spec_builder", "dataset_row"), BACKENDS, ids=[b[0] for b in BACKENDS])
def test_download_step_authenticates_from_job_pod(
    client: NemoClient,
    workspace: str,
    model_entity: str,
    submitted_jobs: list[str],
    backend: str,
    spec_builder: SpecBuilder,
    dataset_row: dict[str, Any],
) -> None:
    rows = "\n".join(json.dumps(dataset_row) for _ in range(4)) + "\n"
    dataset = _create_fileset(
        client, workspace, FilesetPurpose.DATASET, {"training.jsonl": rows, "validation.jsonl": rows}
    )

    job = (
        CustomizationClient.from_client(client)
        .create_customization_job(
            workspace=workspace,
            backend=backend,
            body=CustomizationJobCreateRequest(
                name=_unique(f"e2e-{backend}"),
                spec=spec_builder(workspace, model_entity, dataset, _unique(f"e2e-{backend}-out")),
            ),
        )
        .data()
    )
    submitted_jobs.append(job.name)

    _wait_for_download_step(client, workspace, job.name)
