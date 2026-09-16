# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E coverage for the Safe Synthesizer plugin.

Smoke coverage runs in the local subprocess harness and verifies the API and
Files surfaces. Job execution coverage requires a container/GPU Jobs backend,
such as minikube at ``NHX_BASE_URL=http://localhost:30080``.

Examples:

    uv run --frozen pytest e2e/test_safe_synthesizer.py -v --run-e2e

    NHX_BASE_URL=http://localhost:30080 \
      uv run --frozen pytest e2e/test_safe_synthesizer.py -v \
        --run-e2e --run-slow --feature gpu
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import time
import traceback
import uuid
from collections.abc import Callable, Iterator
from contextlib import suppress
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.schemas import PlatformJobStatus

pytestmark = [
    pytest.mark.timeout(600),
    pytest.mark.e2e_config("e2e/configs/local-subprocess.yaml"),
]

TERMINAL_STATUSES = {"completed", "error", "cancelled"}
STARTED_STATUSES = {"pending", "active", "completed", "error", "cancelled"}

INPUT_REMOTE_PATH = "inputs/safe-synthesizer-e2e.csv"
SMOKE_JOB_TIMEOUT_SECONDS = 180.0
K8S_JOB_TIMEOUT_SECONDS = float(os.environ.get("NSS_E2E_JOB_TIMEOUT_SECONDS", "5400"))
POLL_INTERVAL_SECONDS = float(os.environ.get("NSS_E2E_POLL_INTERVAL_SECONDS", "10"))
RESULT_DOWNLOAD_TIMEOUT_SECONDS = float(os.environ.get("NSS_E2E_RESULT_DOWNLOAD_TIMEOUT_SECONDS", "600"))
DELETE_VERIFY_TIMEOUT_SECONDS = float(os.environ.get("NSS_E2E_DELETE_VERIFY_TIMEOUT_SECONDS", "60"))
DEFAULT_INPUT_ROWS = int(os.environ.get("NSS_E2E_INPUT_ROWS", "250"))
DEFAULT_NUM_RECORDS = int(os.environ.get("NSS_E2E_NUM_RECORDS", "250"))

NssJobFactory = Callable[[str, str, dict[str, Any]], dict[str, Any]]


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _deterministic_pii_replacement_config() -> dict[str, Any]:
    # Select fixture columns explicitly so this e2e does not depend on an
    # external LLM column-classification provider being configured in CI.
    return {
        "enable_synthesis": False,
        "enable_replace_pii": True,
        "replace_pii": {
            "globals": {
                "classify": {"enable_classify": False},
                "seed": 7,
                "locales": ["en_US"],
            },
            "steps": [
                {
                    "vars": {"row_seed": "random.random()"},
                    "rows": {
                        "update": [
                            {"name": "email", "value": "fake.persona(row_index=vars.row_seed + index).email"},
                            {"name": "phone_number", "value": "fake.msisdn()"},
                            {"name": "review", "value": "this | fake_entities"},
                        ]
                    },
                }
            ],
        },
    }


def test_safe_synthesizer_pii_replacement_config_validates() -> None:
    from nemo_safe_synthesizer_plugin.job_config import SafeSynthesizerJobConfig

    spec = SafeSynthesizerJobConfig.model_validate(
        {
            "data_source": "default/data#file.csv",
            "config": _deterministic_pii_replacement_config(),
        }
    )

    assert spec.enable_synthesis is False
    assert spec.config.replace_pii is not None
    assert spec.config.replace_pii.globals.classify.enable_classify is False
    rows = spec.config.replace_pii.steps[0].rows
    assert rows is not None
    updates = rows.update
    assert updates is not None
    assert {rule.name for rule in updates} == {"email", "phone_number", "review"}


def _string_headers(sdk: NeMoHelix) -> dict[str, str]:
    return {key: value for key, value in sdk.default_headers.items() if isinstance(value, str)}


def _nss_url(sdk: NeMoHelix, workspace: str, path: str) -> str:
    return f"{str(sdk.base_url).rstrip('/')}/apis/safe-synthesizer/v2/workspaces/{workspace}/{path.lstrip('/')}"


def _jobs_url(sdk: NeMoHelix, workspace: str, path: str) -> str:
    return f"{str(sdk.base_url).rstrip('/')}/apis/jobs/v2/workspaces/{workspace}/{path.lstrip('/')}"


def _safe_artifact_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _nss_debug_dir(job_name: str) -> Path | None:
    root = os.environ.get("NSS_E2E_DEBUG_DIR") or os.environ.get("JOB_LOGS_DIR")
    if not root:
        return None
    debug_dir = Path(root) / "safe-synthesizer" / _safe_artifact_name(job_name)
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return debug_dir


def _write_nss_debug_text(job_name: str, filename: str, content: str) -> None:
    debug_dir = _nss_debug_dir(job_name)
    if debug_dir is None:
        return
    with suppress(Exception):
        (debug_dir / filename).write_text(content, encoding="utf-8")


def _write_nss_debug_json(job_name: str, filename: str, payload: object) -> None:
    _write_nss_debug_text(job_name, filename, json.dumps(payload, indent=2, default=str))


def _capture_raw_response(sdk: NeMoHelix, job_name: str, filename: str, url: str) -> None:
    with suppress(Exception):
        response = sdk._client.get(
            url,
            headers=_string_headers(sdk),
            timeout=60.0,
        )
        _write_nss_debug_json(
            job_name,
            filename,
            {
                "url": str(response.url),
                "status_code": response.status_code,
                "body": response.text,
            },
        )


def _capture_nss_debug_artifacts(
    sdk: NeMoHelix,
    workspace: str,
    job_name: str,
    reason: str,
    *,
    history: list[str],
    error: BaseException | None = None,
) -> None:
    diagnostic: dict[str, object] = {
        "reason": reason,
        "history": history,
    }
    if error is not None:
        diagnostic["error"] = repr(error)
        diagnostic["traceback"] = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    _write_nss_debug_json(job_name, "diagnostic.json", diagnostic)
    _capture_raw_response(
        sdk, job_name, "jobs-status-response.json", _jobs_url(sdk, workspace, f"jobs/{job_name}/status")
    )
    _capture_raw_response(sdk, job_name, "jobs-logs-response.json", _jobs_url(sdk, workspace, f"jobs/{job_name}/logs"))
    _capture_raw_response(sdk, job_name, "nss-job-response.json", _nss_url(sdk, workspace, f"jobs/{job_name}"))
    _capture_raw_response(
        sdk, job_name, "nss-results-response.json", _nss_url(sdk, workspace, f"jobs/{job_name}/results")
    )


def _files_client(sdk: NeMoHelix) -> FilesClient:
    return client_from_platform(sdk, FilesClient)


def _create_fileset(sdk: NeMoHelix, workspace: str, name: str) -> None:
    _files_client(sdk).create_fileset(
        workspace=workspace,
        body=CreateFilesetRequest(
            name=name,
            description="Safe Synthesizer E2E fileset",
        ),
    )


def _delete_fileset(sdk: NeMoHelix, workspace: str, name: str) -> None:
    with suppress(Exception):
        _files_client(sdk).delete_fileset(name=name, workspace=workspace)


def _dataset_csv(rows: int = DEFAULT_INPUT_ROWS) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "record_id",
            "name",
            "email",
            "phone_number",
            "city",
            "signup_date",
            "favorite_ice_cream_flavor",
            "review",
            "rating",
        ],
    )
    writer.writeheader()
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
    flavors = ["Vanilla", "Chocolate", "Strawberry", "Mint Chip", "Coffee"]
    for index in range(1, rows + 1):
        writer.writerow(
            {
                "record_id": str(index),
                "name": f"Customer {index}",
                "email": f"customer{index}@example.com",
                "phone_number": f"415-555-{index % 10000:04d}",
                "city": cities[index % len(cities)],
                "signup_date": f"2024-{(index % 12) + 1:02d}-{(index % 27) + 1:02d}",
                "favorite_ice_cream_flavor": flavors[index % len(flavors)],
                "review": f"Customer {index} asked support to call 415-555-{index % 10000:04d}.",
                "rating": str((index % 5) + 1),
            }
        )
    return output.getvalue()


def _upload_dataset(sdk: NeMoHelix, workspace: str, *, rows: int = DEFAULT_INPUT_ROWS) -> tuple[str, str]:
    fileset = _unique_name("nss-inputs")
    _create_fileset(sdk, workspace, fileset)
    client_from_platform(sdk, FilesClient).upload_file(
        name=fileset,
        workspace=workspace,
        path=INPUT_REMOTE_PATH,
        content=_dataset_csv(rows).encode(),
    )
    return fileset, f"{workspace}/{fileset}#{INPUT_REMOTE_PATH}"


def _job_payload(
    name: str,
    data_source: str,
    config: dict[str, Any],
    *,
    description: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "spec": {
            "data_source": data_source,
            "config": config,
        },
    }
    if description is not None:
        payload["description"] = description
    return payload


def _create_nss_job(
    sdk: NeMoHelix,
    workspace: str,
    *,
    name: str,
    data_source: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    response = sdk._client.post(
        _nss_url(sdk, workspace, "jobs"),
        json=_job_payload(name, data_source, config),
        headers=_string_headers(sdk),
        timeout=60.0,
    )
    assert response.status_code == 201, f"Failed to create Safe Synthesizer job: {response.text}"
    return response.json()


def _list_nss_jobs(sdk: NeMoHelix, workspace: str) -> dict[str, Any]:
    response = sdk._client.get(
        _nss_url(sdk, workspace, "jobs"),
        headers=_string_headers(sdk),
        timeout=60.0,
    )
    assert response.status_code == 200, f"Failed to list Safe Synthesizer jobs: {response.text}"
    return response.json()


def _job_names(jobs: dict[str, Any]) -> set[str]:
    return {str(entry["name"]) for entry in jobs.get("data", [])}


def _retrieve_nss_job(sdk: NeMoHelix, workspace: str, name: str) -> dict[str, Any]:
    response = sdk._client.get(
        _nss_url(sdk, workspace, f"jobs/{name}"),
        headers=_string_headers(sdk),
        timeout=60.0,
    )
    assert response.status_code == 200, f"Failed to retrieve Safe Synthesizer job {name}: {response.text}"
    return response.json()


def _wait_for_job_absent(
    sdk: NeMoHelix,
    workspace: str,
    name: str,
    *,
    timeout_seconds: float = DELETE_VERIFY_TIMEOUT_SECONDS,
    poll_interval_seconds: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_names: set[str] = set()
    while time.monotonic() < deadline:
        last_names = _job_names(_list_nss_jobs(sdk, workspace))
        if name not in last_names:
            return
        time.sleep(poll_interval_seconds)
    pytest.fail(f"Safe Synthesizer job {name!r} still exists after delete; visible jobs: {sorted(last_names)}")


def _delete_nss_job(sdk: NeMoHelix, workspace: str, name: str, *, verify: bool = True) -> None:
    deadline = time.monotonic() + DELETE_VERIFY_TIMEOUT_SECONDS
    while True:
        response = sdk._client.delete(
            _nss_url(sdk, workspace, f"jobs/{name}"),
            headers=_string_headers(sdk),
            timeout=60.0,
        )
        if response.status_code in {200, 202, 204, 404}:
            break
        if response.status_code != 409:
            response.raise_for_status()
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out deleting Safe Synthesizer job {name!r}; last response={response.text}")
        time.sleep(2.0)
    if verify and response.status_code != 404:
        _wait_for_job_absent(sdk, workspace, name)


def _cancel_nss_job(sdk: NeMoHelix, workspace: str, name: str) -> dict[str, Any] | None:
    response = sdk._client.post(
        _nss_url(sdk, workspace, f"jobs/{name}/cancel"),
        headers=_string_headers(sdk),
        timeout=60.0,
    )
    if response.status_code in {404, 409}:
        return None
    assert response.status_code == 200, f"Failed to cancel Safe Synthesizer job {name}: {response.text}"
    return response.json()


def _list_nss_results(sdk: NeMoHelix, workspace: str, job_name: str) -> dict[str, Any]:
    response = sdk._client.get(
        _nss_url(sdk, workspace, f"jobs/{job_name}/results"),
        headers=_string_headers(sdk),
        timeout=RESULT_DOWNLOAD_TIMEOUT_SECONDS,
    )
    assert response.status_code == 200, f"Failed to list Safe Synthesizer results for {job_name}: {response.text}"
    return response.json()


def _download_nss_result(sdk: NeMoHelix, workspace: str, job_name: str, result_name: str) -> bytes:
    response = sdk._client.get(
        _nss_url(sdk, workspace, f"jobs/{job_name}/results/{result_name}/download"),
        headers=_string_headers(sdk),
        timeout=RESULT_DOWNLOAD_TIMEOUT_SECONDS,
    )
    assert response.status_code == 200, (
        f"Failed to download Safe Synthesizer result {result_name!r} for {job_name}: {response.text}"
    )
    return response.content


def _result_names(results: dict[str, Any]) -> set[str]:
    return {str(result["name"]) for result in results.get("data", [])}


def _status_details(sdk: NeMoHelix, workspace: str, job_name: str) -> str:
    details = [f"Safe Synthesizer job {job_name} did not complete successfully."]
    jobs = client_from_platform(sdk, JobsClient)
    with suppress(Exception):
        job = jobs.get_job(name=job_name, workspace=workspace).data()
        details.append(f"Job: {job.model_dump_json(indent=2)}")
    with suppress(Exception):
        status = jobs.get_job_status(name=job_name, workspace=workspace).data()
        details.append(f"Status: {status.model_dump_json(indent=2)}")
    with suppress(Exception):
        log_entries = list(jobs.list_job_logs(name=job_name, workspace=workspace).items())
        tail = log_entries[-30:] if log_entries else []
        details.append("Recent logs:")
        details.extend(f"[{entry.job_step}] {entry.message}" for entry in tail)
    return "\n".join(details)


def _status_value(status: object) -> str:
    if isinstance(status, Enum):
        return str(status.value)
    return str(status or "")


def _wait_for_status(
    sdk: NeMoHelix,
    workspace: str,
    job_name: str,
    *,
    target_statuses: set[str] | None = None,
    timeout_seconds: float,
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS,
) -> tuple[str, list[str]]:
    target_statuses = target_statuses or TERMINAL_STATUSES
    deadline = time.monotonic() + timeout_seconds
    history: list[str] = []
    last_error: BaseException | None = None

    while time.monotonic() < deadline:
        try:
            status_info = client_from_platform(sdk, JobsClient).get_job_status(name=job_name, workspace=workspace)
            status = _status_value(status_info.status)
            if not history or history[-1] != status:
                history.append(status)
                _write_nss_debug_json(job_name, "status-history.json", {"history": history})
            if status == "error":
                _capture_nss_debug_artifacts(
                    sdk,
                    workspace,
                    job_name,
                    "job reached error status",
                    history=history,
                )
            if status in target_statuses:
                return status, history
        except Exception as exc:
            last_error = exc
            _capture_nss_debug_artifacts(
                sdk,
                workspace,
                job_name,
                "polling raised an exception",
                history=history,
                error=exc,
            )
        time.sleep(poll_interval_seconds)

    _capture_nss_debug_artifacts(
        sdk,
        workspace,
        job_name,
        "timed out waiting for target status",
        history=history,
        error=last_error,
    )
    detail = _status_details(sdk, workspace, job_name)
    if last_error is not None:
        last_traceback = "".join(traceback.format_exception(type(last_error), last_error, last_error.__traceback__))
        detail = f"{detail}\nLast polling error: {last_error!r}\nLast polling traceback:\n{last_traceback}"
    raise TimeoutError(
        f"Timed out waiting for {job_name} to reach {sorted(target_statuses)}; history={history}\n{detail}"
    )


def _assert_job_completed(sdk: NeMoHelix, workspace: str, job_name: str) -> list[str]:
    status, history = _wait_for_status(
        sdk,
        workspace,
        job_name,
        timeout_seconds=K8S_JOB_TIMEOUT_SECONDS,
    )
    if status != "completed":
        _capture_nss_debug_artifacts(
            sdk,
            workspace,
            job_name,
            f"expected completed status, got {status}",
            history=history,
        )
    assert status == "completed", _status_details(sdk, workspace, job_name)
    assert any(seen in STARTED_STATUSES for seen in history), f"Unexpected job status history: {history}"
    return history


def _assert_csv_rows(content: bytes, *, expected_rows: int | None = None, min_rows: int = 1) -> list[dict[str, str]]:
    text = content.decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) >= min_rows, f"Expected at least {min_rows} CSV rows, got {len(rows)}. Content: {text[:500]}"
    if expected_rows is not None:
        assert len(rows) == expected_rows, f"Expected {expected_rows} CSV rows, got {len(rows)}"
    return rows


def _assert_known_pii_replaced(content: bytes) -> None:
    text = content.decode("utf-8")
    for source_value in ("customer1@example.com", "customer250@example.com", "415-555-0001", "415-555-0250"):
        assert source_value not in text


def _platform_root() -> Path:
    candidates: list[Path] = []
    if os.environ.get("NHX_PLATFORM_ROOT"):
        candidates.append(Path(os.environ["NHX_PLATFORM_ROOT"]))
    candidates.extend(
        [
            Path(__file__).resolve().parents[1],
            Path.cwd() / "platform",
            Path.cwd(),
        ]
    )
    for candidate in candidates:
        script = candidate / "plugins/nemo-safe-synthesizer/scripts/setup_model_filesets.py"
        if script.is_file():
            return candidate
    raise FileNotFoundError("Could not locate plugins/nemo-safe-synthesizer/scripts/setup_model_filesets.py")


@pytest.fixture(scope="module")
def nss_model_filesets(sdk: NeMoHelix) -> None:
    if os.environ.get("NSS_E2E_SKIP_MODEL_FILESETS") == "1":
        return

    platform_root = _platform_root()
    setup_module = _load_model_filesets_setup_module(platform_root)
    create_filesets = cast(
        Callable[[NeMoHelix, str, bool], list[str]],
        getattr(setup_module, "create_filesets"),
    )
    model_filesets = cast(list[dict[str, object]], getattr(setup_module, "MODEL_FILESETS"))

    created = create_filesets(sdk, "default", False)
    if len(created) != len(model_filesets):
        pytest.fail(
            "Failed to register all Safe Synthesizer model filesets: "
            f"registered {len(created)} of {len(model_filesets)} ({', '.join(created)})"
        )


def _load_model_filesets_setup_module(platform_root: Path) -> ModuleType:
    setup_script = platform_root / "plugins/nemo-safe-synthesizer/scripts/setup_model_filesets.py"
    spec = importlib.util.spec_from_file_location("nss_setup_model_filesets", setup_script)
    if spec is None or spec.loader is None:
        pytest.fail(f"Failed to load Safe Synthesizer model fileset setup script: {setup_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def nss_dataset(sdk: NeMoHelix, workspace: str) -> Iterator[tuple[str, str]]:
    fileset, data_source = _upload_dataset(sdk, workspace)
    try:
        yield fileset, data_source
    finally:
        _delete_fileset(sdk, workspace, fileset)


@pytest.fixture
def nss_job(sdk: NeMoHelix, workspace: str) -> Iterator[NssJobFactory]:
    job_names: list[str] = []

    def create(prefix: str, data_source: str, config: dict[str, Any]) -> dict[str, Any]:
        job_name = _unique_name(prefix)
        job = _create_nss_job(sdk, workspace, name=job_name, data_source=data_source, config=config)
        job_names.append(job_name)
        return job

    try:
        yield create
    finally:
        cleanup_errors: list[Exception] = []
        for job_name in reversed(job_names):
            try:
                cancel_response = _cancel_nss_job(sdk, workspace, job_name)
                if cancel_response is not None:
                    _wait_for_status(
                        sdk,
                        workspace,
                        job_name,
                        timeout_seconds=SMOKE_JOB_TIMEOUT_SECONDS,
                        poll_interval_seconds=2.0,
                    )
            except Exception as exc:
                cleanup_errors.append(exc)
                _capture_nss_debug_artifacts(
                    sdk,
                    workspace,
                    job_name,
                    "cleanup cancel/wait failed",
                    history=[],
                    error=exc,
                )
            try:
                _delete_nss_job(sdk, workspace, job_name)
            except Exception as exc:
                cleanup_errors.append(exc)
                _capture_nss_debug_artifacts(
                    sdk,
                    workspace,
                    job_name,
                    "cleanup delete failed",
                    history=[],
                    error=exc,
                )
        if cleanup_errors:
            raise ExceptionGroup("Safe Synthesizer job cleanup failed", cleanup_errors)


def test_safe_synthesizer_api_health(sdk: NeMoHelix, workspace: str) -> None:
    response = sdk._client.get(
        f"{str(sdk.base_url).rstrip('/')}/status",
        headers=_string_headers(sdk),
        timeout=60.0,
    )
    response.raise_for_status()
    ready_services = response.json().get("services", {}).get("ready", [])
    assert "safe-synthesizer" in ready_services

    jobs = _list_nss_jobs(sdk, workspace)
    assert isinstance(jobs.get("data"), list)


def test_safe_synthesizer_status_value_accepts_sdk_status_enum() -> None:
    assert _status_value(PlatformJobStatus.CANCELLED) == "cancelled"
    assert _status_value("completed") == "completed"


def test_safe_synthesizer_fileset_upload_download_round_trips(
    sdk: NeMoHelix,
    workspace: str,
    nss_dataset: tuple[str, str],
) -> None:
    fileset, _ = nss_dataset
    downloaded = (
        client_from_platform(sdk, FilesClient)
        .download_file(
            name=fileset,
            workspace=workspace,
            path=INPUT_REMOTE_PATH,
        )
        .read()
    )

    assert downloaded.decode("utf-8") == _dataset_csv()


@pytest.mark.container_only
@pytest.mark.requires_gpu
def test_safe_synthesizer_job_create_list_retrieve_cancel_delete(
    sdk: NeMoHelix,
    workspace: str,
    nss_dataset: tuple[str, str],
    nss_job: NssJobFactory,
) -> None:
    _, data_source = nss_dataset

    job = nss_job(
        "nss-smoke",
        data_source,
        {
            "enable_synthesis": False,
            "enable_replace_pii": False,
        },
    )
    job_name = str(job["name"])
    assert job["name"] == job_name

    jobs = _list_nss_jobs(sdk, workspace)
    assert job_name in _job_names(jobs)

    retrieved = _retrieve_nss_job(sdk, workspace, job_name)
    assert retrieved["name"] == job_name
    assert retrieved["spec"]["data_source"] == data_source

    cancel_response = _cancel_nss_job(sdk, workspace, job_name)
    status, _ = _wait_for_status(
        sdk,
        workspace,
        job_name,
        timeout_seconds=SMOKE_JOB_TIMEOUT_SECONDS,
        poll_interval_seconds=2.0,
    )
    assert status != "error"
    if cancel_response is not None:
        assert status == "cancelled"


@pytest.mark.container_only
@pytest.mark.requires_gpu
@pytest.mark.slow
@pytest.mark.timeout(7200)
def test_safe_synthesizer_k8s_job_cancel_transitions(
    sdk: NeMoHelix,
    workspace: str,
    nss_dataset: tuple[str, str],
    nss_job: NssJobFactory,
    nss_model_filesets: None,
) -> None:
    _, data_source = nss_dataset
    job = nss_job(
        "nss-cancel",
        data_source,
        {
            "enable_synthesis": True,
            "enable_replace_pii": False,
            "generation": {"num_records": DEFAULT_NUM_RECORDS},
            "evaluation": {"enabled": False},
            "privacy": {"dp_enabled": False},
        },
    )
    job_name = str(job["name"])

    _, history = _wait_for_status(
        sdk,
        workspace,
        job_name,
        target_statuses=STARTED_STATUSES,
        timeout_seconds=300,
    )
    assert "error" not in history
    cancel_response = _cancel_nss_job(sdk, workspace, job_name)
    assert cancel_response is not None

    final_status, final_history = _wait_for_status(
        sdk,
        workspace,
        job_name,
        timeout_seconds=600,
    )
    assert final_status == "cancelled"
    assert "cancelled" in final_history


@pytest.mark.container_only
@pytest.mark.requires_gpu
@pytest.mark.slow
@pytest.mark.timeout(7200)
def test_safe_synthesizer_pii_replacement_job_completes(
    sdk: NeMoHelix,
    workspace: str,
    nss_dataset: tuple[str, str],
    nss_job: NssJobFactory,
    nss_model_filesets: None,
) -> None:
    _, data_source = nss_dataset
    job = nss_job(
        "nss-pii",
        data_source,
        _deterministic_pii_replacement_config(),
    )
    job_name = str(job["name"])

    _assert_job_completed(sdk, workspace, job_name)
    results = _list_nss_results(sdk, workspace, job_name)
    assert {"summary", "synthetic-data"}.issubset(_result_names(results))
    synthetic_content = _download_nss_result(sdk, workspace, job_name, "synthetic-data")
    _assert_csv_rows(
        synthetic_content,
        expected_rows=DEFAULT_INPUT_ROWS,
    )
    _assert_known_pii_replaced(synthetic_content)
    summary = json.loads(_download_nss_result(sdk, workspace, job_name, "summary"))
    assert summary.get("timing") is not None


@pytest.mark.container_only
@pytest.mark.requires_gpu
@pytest.mark.slow
@pytest.mark.timeout(7200)
def test_safe_synthesizer_full_workflow_downloads_artifacts(
    sdk: NeMoHelix,
    workspace: str,
    nss_dataset: tuple[str, str],
    nss_job: NssJobFactory,
    nss_model_filesets: None,
) -> None:
    _, data_source = nss_dataset
    job = nss_job(
        "nss-full",
        data_source,
        {
            "enable_synthesis": True,
            "enable_replace_pii": True,
            "generation": {"num_records": DEFAULT_NUM_RECORDS},
            "evaluation": {"enabled": True},
            "privacy": {"dp_enabled": False},
        },
    )
    job_name = str(job["name"])

    _assert_job_completed(sdk, workspace, job_name)
    results = _list_nss_results(sdk, workspace, job_name)
    result_names = _result_names(results)
    assert {"summary", "synthetic-data", "evaluation-report", "adapter"}.issubset(result_names)

    synthetic_rows = _assert_csv_rows(
        _download_nss_result(sdk, workspace, job_name, "synthetic-data"),
        expected_rows=DEFAULT_NUM_RECORDS,
    )
    assert set(synthetic_rows[0]) >= {"name", "email", "favorite_ice_cream_flavor"}

    summary = json.loads(_download_nss_result(sdk, workspace, job_name, "summary"))
    timing = summary["timing"]
    assert timing["training_time_sec"] is not None
    assert timing["generation_time_sec"] is not None

    report = _download_nss_result(sdk, workspace, job_name, "evaluation-report")
    assert b"<html" in report.lower() or b"<!doctype html" in report.lower()
