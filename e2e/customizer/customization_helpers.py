# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for plugin-based customization E2E tests."""

import json
import logging
import os
import re
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from filesets import transfer
from httpx import HTTPStatusError
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.storage_config import HuggingfaceStorageConfig
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import CreateModelEntityRequest
from nemo_helix_plugin.schema import SecretRef
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from nhx.testing.e2e.customizer import wait_for_model_spec
from pydantic import SecretStr

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"completed", "cancelled", "error"}
COMPLETED_STATUS = "completed"
_NAME_SAFE_RE = re.compile(r"[^a-z0-9-]+")


def unique_name(prefix: str) -> str:
    """Return a short, API-safe test resource name."""
    clean_prefix = _NAME_SAFE_RE.sub("-", prefix.lower()).strip("-") or "e2e"
    clean_prefix = clean_prefix[:40].rstrip("-")
    return f"{clean_prefix}-{uuid.uuid4().hex[:8]}"


def create_hf_token_secret(
    client: NemoClient,
    workspace: str,
    *,
    env_var: str = "HF_TOKEN",
    name_prefix: str = "e2e-hf-token",
) -> str | None:
    """Create an HF token secret when the configured environment variable is set."""
    token = os.environ.get(env_var)
    if not token:
        logger.info("%s is not set; creating HuggingFace filesets without a token", env_var)
        return None

    secret_name = unique_name(name_prefix)
    logger.info("Creating HuggingFace credential resource in workspace %s", workspace)
    SecretsClient.from_client(client).create_secret(
        workspace=workspace,
        body=HelixSecretCreateRequest(name=secret_name, value=SecretStr(token)),
    )
    return secret_name


def delete_secret_if_present(client: NemoClient, workspace: str, secret_name: str | None) -> None:
    """Best-effort delete for optional E2E secrets."""
    if secret_name is None:
        return
    try:
        SecretsClient.from_client(client).delete_secret(name=secret_name, workspace=workspace)
        logger.info("Deleted credential resource in workspace %s", workspace)
    except Exception as exc:
        _handle_cleanup_error("credential resource", workspace, secret_name, exc)


def _is_not_found_error(exc: Exception) -> bool:
    return isinstance(exc, NotFoundError) or (isinstance(exc, HTTPStatusError) and exc.response.status_code == 404)


def _debug_not_found(resource: str, workspace: str, name: str) -> None:
    logger.debug("%s in workspace %s already deleted", resource, workspace)


def _warn_cleanup_failed(resource: str, workspace: str, name: str) -> None:
    logger.warning("Failed to delete %s in workspace %s", resource, workspace, exc_info=True)


def _handle_cleanup_error(resource: str, workspace: str, name: str, exc: Exception) -> None:
    if _is_not_found_error(exc):
        _debug_not_found(resource, workspace, name)
        return
    _warn_cleanup_failed(resource, workspace, name)


def create_hf_model_entity(
    client: NemoClient,
    workspace: str,
    *,
    hf_repo: str,
    name_prefix: str,
    token_secret: str | None = None,
    trust_remote_code: bool = False,
    wait_for_spec: bool = True,
    wait_timeout: int = 600,
) -> str:
    """Create an HF-backed fileset and matching model entity."""
    fileset_name = unique_name(name_prefix)
    storage = HuggingfaceStorageConfig(
        repo_id=hf_repo,
        repo_type="model",
        token_secret=SecretRef(token_secret) if token_secret else None,
    )

    logger.info("Creating HuggingFace fileset %s for %s", fileset_name, hf_repo)
    FilesClient.from_client(client).create_fileset(
        workspace=workspace,
        body=CreateFilesetRequest(
            name=fileset_name,
            description=f"E2E model source: {hf_repo}",
            storage=storage,
        ),
    )

    logger.info("Creating model entity %s backed by %s", fileset_name, fileset_name)
    ModelsClient.from_client(client).create_model(
        workspace=workspace,
        body=CreateModelEntityRequest(
            name=fileset_name,
            fileset=f"{workspace}/{fileset_name}",
            description=f"E2E model entity for {hf_repo}",
            trust_remote_code=trust_remote_code,
        ),
    )
    if wait_for_spec:
        wait_for_model_spec(client, workspace, fileset_name, timeout=wait_timeout)
    return fileset_name


def delete_model_entity_and_fileset(client: NemoClient, workspace: str, name: str | None) -> None:
    """Best-effort delete for a model entity and its matching fileset."""
    if name is None:
        return
    for resource_name, delete_fn in (
        ("model entity", ModelsClient.from_client(client).delete_model),
        ("fileset", FilesClient.from_client(client).delete_fileset),
    ):
        try:
            delete_fn(name=name, workspace=workspace)
            logger.info("Deleted %s %s/%s", resource_name, workspace, name)
        except Exception as exc:
            _handle_cleanup_error(resource_name, workspace, name, exc)


def upload_jsonl_fileset(
    client: NemoClient,
    workspace: str,
    *,
    source_dir: Path,
    name_prefix: str,
    file_map: Mapping[str, str] | None = None,
) -> str:
    """Upload JSONL files from a source directory into a new fileset."""
    fileset_name = unique_name(name_prefix)
    file_map = file_map or {
        "training.jsonl": "train.jsonl",
        "validation.jsonl": "validation.jsonl",
    }

    files = FilesClient.from_client(client)
    files.create_fileset(
        workspace=workspace,
        body=CreateFilesetRequest(name=fileset_name, description=f"E2E dataset from {source_dir.name}"),
    )

    for source_name, remote_name in file_map.items():
        source_path = source_dir / source_name
        _assert_materialized_jsonl(source_path)
        logger.info(
            "Uploading %s to %s/%s#%s",
            source_path,
            workspace,
            fileset_name,
            remote_name,
        )
        transfer.upload(
            files,
            local_path=str(source_path),
            remote_path=remote_name,
            fileset=fileset_name,
            workspace=workspace,
        )

    assert_fileset_has_files(client, workspace, fileset_name)
    return fileset_name


def upload_records_fileset(
    client: NemoClient,
    workspace: str,
    *,
    records: Sequence[Mapping[str, Any]],
    name_prefix: str,
    remote_filename: str = "train.jsonl",
) -> str:
    """Create a fileset from in-memory JSONL records."""
    if not records:
        raise ValueError("records must not be empty")

    fileset_name = unique_name(name_prefix)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as temp_file:
        for record in records:
            temp_file.write(json.dumps(record) + "\n")
        temp_path = Path(temp_file.name)

    files = FilesClient.from_client(client)
    try:
        files.create_fileset(
            workspace=workspace,
            body=CreateFilesetRequest(name=fileset_name, description="E2E inline JSONL dataset"),
        )
        transfer.upload(
            files,
            local_path=str(temp_path),
            remote_path=remote_filename,
            fileset=fileset_name,
            workspace=workspace,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    assert_fileset_has_files(client, workspace, fileset_name)
    return fileset_name


def delete_fileset(client: NemoClient, workspace: str, fileset_name: str | None) -> None:
    """Best-effort delete for a fileset."""
    if fileset_name is None:
        return
    try:
        FilesClient.from_client(client).delete_fileset(name=fileset_name, workspace=workspace)
        logger.info("Deleted fileset %s/%s", workspace, fileset_name)
    except Exception as exc:
        _handle_cleanup_error("fileset", workspace, fileset_name, exc)


def assert_fileset_has_files(client: NemoClient, workspace: str, fileset_name: str) -> None:
    """Assert that a fileset exists and has at least one file."""
    files = FilesClient.from_client(client).list_files(name=fileset_name, workspace=workspace).data().data
    logger.info("Fileset %s/%s has %d file(s)", workspace, fileset_name, len(files))
    for file_info in files:
        logger.info("  %s", getattr(file_info, "path", file_info))
    assert files, f"Expected fileset {workspace}/{fileset_name} to contain files"


def wait_for_backend_job(
    client: NemoClient,
    workspace: str,
    label: str,
    *,
    timeout: float,
    poll_interval: float = 15.0,
) -> object:
    """Poll the core platform job until it reaches a terminal state."""
    jobs = JobsClient.from_client(client)
    deadline = time.time() + timeout
    last_status: str | None = None
    status = None
    while time.time() < deadline:
        status = jobs.get_job_status(name=label, workspace=workspace).data()
        current = _status_value(getattr(status, "status", "unknown"))
        if current != last_status:
            logger.info("Job %s status: %s", label, current)
            _log_status_details(status)
            last_status = current
        if current in TERMINAL_STATUSES:
            return status
        time.sleep(poll_interval)

    pytest.fail(f"Timed out waiting for job {label} after {timeout}s. Last status: {last_status}")


def assert_backend_job_completed(
    client: NemoClient,
    workspace: str,
    job_name: str,
    status: object,
) -> None:
    """Assert a backend platform job completed, with diagnostics on failure."""
    current = _status_value(getattr(status, "status", "unknown"))
    if current == COMPLETED_STATUS:
        return

    pytest.fail(_backend_job_failure_details(client, workspace, job_name, status))


def assert_output_registered(
    client: NemoClient,
    workspace: str,
    *,
    base_model_name: str,
    output_name: str,
) -> None:
    """Assert a completed backend output exists as a model entity or adapter."""
    models = ModelsClient.from_client(client)
    try:
        models.get_model(name=output_name, workspace=workspace)
        logger.info("Output registered as model entity %s/%s", workspace, output_name)
        return
    except NotFoundError:
        pass

    base_model = models.get_model(name=base_model_name, workspace=workspace).data()
    adapter_names = [adapter.name for adapter in (base_model.adapters or [])]
    if output_name in adapter_names:
        logger.info(
            "Output registered as adapter %s on base model %s",
            output_name,
            base_model_name,
        )
        return

    try:
        models.get_adapter(name=output_name, workspace=workspace)
        logger.info("Output registered as top-level adapter %s/%s", workspace, output_name)
        return
    except NotFoundError:
        pass

    pytest.fail(
        f"Output {output_name} was not registered as a model entity or adapter. "
        f"Adapters on {base_model_name}: {adapter_names}"
    )


def delete_output_model_or_adapter(
    client: NemoClient,
    workspace: str,
    *,
    output_name: str | None,
    base_model_name: str | None = None,
) -> None:
    """Best-effort delete for backend output model or adapter resources."""
    if output_name is None:
        return

    models = ModelsClient.from_client(client)
    try:
        models.delete_model(name=output_name, workspace=workspace)
        logger.info("Deleted output model entity %s/%s", workspace, output_name)
    except Exception as exc:
        _handle_cleanup_error("output model entity", workspace, output_name, exc)

    if base_model_name is not None:
        try:
            models.delete_model_adapter(workspace=workspace, model_name=base_model_name, adapter=output_name)
            logger.info(
                "Deleted output adapter %s from model %s/%s",
                output_name,
                workspace,
                base_model_name,
            )
        except Exception as exc:
            _handle_cleanup_error("output adapter", workspace, output_name, exc)

    try:
        models.delete_adapter(name=output_name, workspace=workspace)
        logger.info("Deleted top-level adapter %s/%s", workspace, output_name)
    except Exception as exc:
        _handle_cleanup_error("top-level adapter", workspace, output_name, exc)


def _assert_materialized_jsonl(path: Path) -> None:
    if not path.exists():
        pytest.fail(f"Data file not found: {path}. Stage assets from S3 first (see e2e/customizer/stage_assets.py).")

    first_line = path.read_text().splitlines()[0] if path.stat().st_size else ""
    if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
        pytest.fail(f"{path} is an LFS pointer. Stage assets from S3 instead of using git-tracked payloads.")


def _status_value(status: object) -> str:
    value = getattr(status, "value", status)
    return str(value).lower()


def _log_status_details(status: object) -> None:
    details = getattr(status, "status_details", None)
    if details:
        logger.info("Status details: %s", details)
    error_details = getattr(status, "error_details", None)
    if error_details:
        logger.error("Error details: %s", error_details)


def _backend_job_failure_details(
    client: NemoClient,
    workspace: str,
    job_name: str,
    status: object,
) -> str:
    current = _status_value(getattr(status, "status", "unknown"))
    details = [
        (f"Job {job_name} should complete successfully, got: {current}"),
        "",
        "Job Status:",
        _format_status(status),
    ]

    try:
        entries = list(JobsClient.from_client(client).list_job_logs(name=job_name, workspace=workspace).items())
        if entries:
            details.extend(["", f"Job Logs (last {min(len(entries), 40)} entries):"])
            for entry in entries[-40:]:
                job_step = getattr(entry, "job_step", "unknown")
                message = getattr(entry, "message", entry)
                details.append(f"  [{job_step}] {message}")
        else:
            details.extend(["", "No logs available"])
    except Exception as exc:
        details.extend(["", f"Failed to get job logs: {exc}"])

    return "\n".join(details)


def _format_status(status: object) -> str:
    dump_json = getattr(status, "model_dump_json", None)
    if callable(dump_json):
        return str(dump_json(indent=2))

    dump = getattr(status, "model_dump", None)
    if callable(dump):
        return json.dumps(dump(), indent=2, default=str)

    return str(status)
