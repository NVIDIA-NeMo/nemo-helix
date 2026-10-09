# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP and polling helpers for the instance acceptance suite."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import (
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError,
)
from nemo_helix_plugin.client.oidc import NHXOIDCConfig
from nemo_helix_plugin.client.types import PreparedRequest
from nemo_helix_plugin.deployments.client import DeploymentsClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_helix_plugin.inference_gateway.types import JsonBody
from nemo_helix_plugin.jobs.client import JobsClient
from probe import parse_auth_report

T = TypeVar("T")
FILESET_NAME = "check-reports"
# Local platforms publish /health/ready and /status. Hosted ingress often exposes only /cluster-info.
PLATFORM_PROBE_PATHS = ("/health/ready", "/status", "/cluster-info")
JOB_SUCCESS = frozenset({"completed"})
JOB_TERMINAL = frozenset({"completed", "error", "cancelled"})
DEPLOYMENT_SUCCESS = frozenset({"SUCCEEDED"})
DEPLOYMENT_TERMINAL = frozenset({"SUCCEEDED", "FAILED", "LOST"})
MODEL_SUCCESS = frozenset({"READY"})
DEFAULT_MODEL_IMAGE = "docker.io/library/python:3.12-alpine"
MODEL_TERMINAL = frozenset({"READY", "ERROR", "DELETED", "FAILED"})
_CAPACITY_MARKERS = (
    "gpu",
    "insufficient",
    "unschedulable",
    "no nodes",
    "accelerator",
    "capacity",
    "quota",
)


@dataclass(frozen=True)
class CheckSettings:
    """Flags the check script passes to the suite through the environment."""

    timeout: float
    keep: bool
    models: bool
    model: str | None
    model_image: str
    require_token_exchange: bool
    job_profile: str | None
    executor: str | None
    image: str | None
    workspace: str | None


def load_settings() -> CheckSettings:
    """Read the suite flags. A model name turns model checks on."""
    model = os.environ.get("NHX_CHECK_MODEL") or None
    return CheckSettings(
        timeout=float(os.environ.get("NHX_CHECK_TIMEOUT", "600")),
        keep=os.environ.get("NHX_CHECK_KEEP") == "1",
        models=os.environ.get("NHX_CHECK_MODELS") == "1" or model is not None,
        model=model,
        model_image=os.environ.get("NHX_CHECK_MODEL_IMAGE") or DEFAULT_MODEL_IMAGE,
        require_token_exchange=os.environ.get("NHX_CHECK_REQUIRE_TOKEN_EXCHANGE") == "1",
        job_profile=os.environ.get("NHX_CHECK_JOB_PROFILE") or None,
        executor=os.environ.get("NHX_CHECK_EXECUTOR") or None,
        image=os.environ.get("NHX_CHECK_IMAGE") or None,
        workspace=os.environ.get("NHX_CHECK_WORKSPACE") or None,
    )


def request_json(
    client: NemoClient,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send one JSON request through the authenticated platform client."""
    content = None if body is None else json.dumps(body).encode()
    response = client.send(
        PreparedRequest(
            path_template=path,
            path_params={},
            method=method,
            content=content,
            content_type="application/json" if body is not None else None,
            response_type=None,
        )
    )
    raw = response.http_response.content
    if not raw:
        return {}
    parsed = response.http_response.json()
    if isinstance(parsed, dict):
        return parsed
    raise TypeError(f"{method} {path} returned {type(parsed).__name__}")


def read_platform_probe(client: NemoClient) -> tuple[str, dict[str, Any]]:
    """Return the first platform probe that exists.

    ``/health/ready`` is the local readiness endpoint. Hosted deployments often
    publish only ``/cluster-info``, which is enough to show the instance is up.
    """
    last_error: NotFoundError | None = None
    for path in PLATFORM_PROBE_PATHS:
        try:
            return path, request_json(client, "GET", path)
        except NotFoundError as exc:
            last_error = exc
    raise RuntimeError(f"platform did not respond at {', '.join(PLATFORM_PROBE_PATHS)}") from last_error


AuthProfile = Literal["anonymous", "trusted_headers", "token_exchange"]


def auth_flags(discovery: NHXOIDCConfig) -> tuple[bool, bool]:
    """Return ``(auth_enabled, token_exchange_enabled)`` from discovered OIDC config."""
    return discovery.auth_enabled, discovery.workload_token_exchange_enabled


def auth_profile(discovery: NHXOIDCConfig) -> AuthProfile:
    """Classify the server into the auth profile this suite knows how to test.

    ``anonymous`` has auth disabled. ``trusted_headers`` has auth and no workload
    token exchange, so jobs receive ``NHX_PRINCIPAL`` and deployments authenticate
    through the auth-proxy sidecar. ``token_exchange`` issues a workload-identity
    token to the workload.
    """
    if not discovery.auth_enabled:
        return "anonymous"
    if discovery.workload_token_exchange_enabled:
        return "token_exchange"
    return "trusted_headers"


def default_task_image(client: NemoClient, profile: str | None) -> str | None:
    """Return the execution profile's default task image, if one is configured."""
    profiles = JobsClient.from_client(client).list_execution_profiles()
    if profile:
        for item in profiles:
            if getattr(item, "profile", None) == profile:
                image = getattr(getattr(item, "config", None), "default_task_image", None)
                return image or None
    for item in profiles:
        image = getattr(getattr(item, "config", None), "default_task_image", None)
        if image:
            return image
    return None


def task_image_from_launcher(launcher_image: str) -> str | None:
    """Derive the ``nhx-tasks`` image from a profile's ``nhx-api`` launcher image.

    Hosted profiles often leave ``default_task_image`` empty and set
    ``launcher_image`` to ``…/nhx-api:<revision>``. Jobs then run
    ``…/nhx-tasks:<revision>``.
    """
    repository, separator, tag = launcher_image.rpartition(":")
    if not separator or "/" in tag:
        repository, tag = launcher_image, ""
    prefix, slash, name = repository.rpartition("/")
    if name != "nhx-api":
        return None
    rebuilt = f"{prefix}{slash}nhx-tasks"
    if tag:
        return f"{rebuilt}:{tag}"
    return rebuilt


def resolve_check_image(client: NemoClient, explicit: str | None, profile: str | None) -> str | None:
    """Prefer an explicit image, then a profile default, then the launcher's task image."""
    if explicit:
        return explicit
    configured = default_task_image(client, profile)
    if configured:
        return configured
    profiles = JobsClient.from_client(client).list_execution_profiles()
    for item in profiles:
        if profile and getattr(item, "profile", None) != profile:
            continue
        launcher = getattr(getattr(item, "config", None), "launcher_image", None)
        if not launcher:
            continue
        derived = task_image_from_launcher(launcher)
        if derived:
            return derived
    return None


def chat_target(model: str, workspace: str) -> tuple[str, str]:
    """Return the workspace and body model for one chat completion.

    ``owner/name`` is called in ``owner``. The body keeps ``owner/name`` so the
    gateway accepts the workspace prefix. A bare name is called in *workspace*.
    """
    owner, separator, name = model.partition("/")
    if separator and owner and name and "/" not in name:
        return owner, model
    return workspace, model


def retry_until_workspace_granted(operation: Callable[[], T], *, timeout: float = 30) -> T:
    """Retry while a new workspace's grants are still propagating.

    Create calls return 403, and entity reads return 422, for a few seconds
    after the workspace exists. A 422 for any other reason is returned immediately.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            return operation()
        except PermissionDeniedError:
            if time.monotonic() >= deadline:
                raise
        except UnprocessableEntityError as exc:
            if "Not allowed to list entities" not in str(exc) or time.monotonic() >= deadline:
                raise
        time.sleep(2)


def ensure_fileset(client: NemoClient, workspace: str) -> None:
    """Create the report fileset, leaving an existing one in place."""

    def create() -> None:
        try:
            FilesClient.from_client(client).create_fileset(
                workspace=workspace,
                body=CreateFilesetRequest(name=FILESET_NAME),
            )
        except ConflictError:
            return

    retry_until_workspace_granted(create)


def download_report(client: NemoClient, workspace: str, path: str) -> dict[str, Any] | None:
    """Read a probe report uploaded into the check fileset."""
    response = FilesClient.from_client(client).download_file(
        workspace=workspace,
        name=FILESET_NAME,
        path=path,
    )
    parsed = json.loads(response.read().decode())
    if isinstance(parsed, dict):
        return parsed
    return None


def job_log_text(client: NemoClient, workspace: str, name: str) -> str:
    """Join job log messages. Empty when the platform has not stored any yet."""
    logs = JobsClient.from_client(client).get_logs(name=name, workspace=workspace)
    return "\n".join(item.message for item in logs.data if item.message)


def load_job_report(client: NemoClient, workspace: str, name: str, path: str) -> dict[str, Any]:
    """Prefer the fileset report, then the ``INSTANCE_CHECK_AUTH`` log line."""
    try:
        report = download_report(client, workspace, path)
    except Exception:
        report = None
    if report is not None:
        return report
    parsed = parse_auth_report(job_log_text(client, workspace, name))
    if parsed is None:
        raise RuntimeError(f"job {name} produced no auth report")
    return parsed


def load_deployment_report(client: NemoClient, workspace: str, path: str) -> dict[str, Any]:
    """Read the deployment report. Deployments have no log API."""
    deadline = time.monotonic() + 30
    last_error = "report was not uploaded"
    while time.monotonic() < deadline:
        try:
            report = download_report(client, workspace, path)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            report = None
        if report is not None:
            return report
        time.sleep(2)
    raise RuntimeError(f"deployment report {path} was not stored: {last_error}")


def wait_for_job(client: NemoClient, workspace: str, name: str, timeout: float) -> Any:
    """Poll a job until it reaches a terminal status."""
    jobs = JobsClient.from_client(client)
    deadline = time.monotonic() + timeout
    last = ""

    def fetch() -> Any:
        return retry_until_workspace_granted(lambda: jobs.get_job(name=name, workspace=workspace).data())

    job = fetch()
    while time.monotonic() < deadline:
        job = fetch()
        status = str(getattr(job.status, "value", job.status))
        if status != last:
            print(f"job {name}: {status}", flush=True)
            last = status
        if status in JOB_TERMINAL:
            return job
        time.sleep(2)
    raise TimeoutError(f"job {name} still {last or 'unknown'} after {timeout:.0f}s")


def wait_for_deployment(client: NemoClient, workspace: str, name: str, timeout: float) -> Any:
    """Poll a deployment until it reaches a terminal status."""
    deployments = DeploymentsClient.from_client(client)
    deadline = time.monotonic() + timeout
    last = ""
    deployment = deployments.get_deployment(workspace=workspace, name=name).data()
    while time.monotonic() < deadline:
        deployment = deployments.get_deployment(workspace=workspace, name=name).data()
        status = str(deployment.status)
        if status != last:
            print(f"deployment {name}: {status}", flush=True)
            last = status
        if status in DEPLOYMENT_TERMINAL:
            return deployment
        time.sleep(2)
    raise TimeoutError(f"deployment {name} still {last or 'unknown'} after {timeout:.0f}s")


def is_capacity_error(payload: dict[str, Any]) -> bool:
    """True when a status or error message says the cluster has no capacity."""
    parts = (payload.get("status_message"), payload.get("message"), payload.get("error"))
    text = " ".join(str(part) for part in parts if part).lower()
    return any(marker in text for marker in _CAPACITY_MARKERS)


def split_image(image: str) -> tuple[str, str]:
    """Split ``repository:tag`` on the last colon when the tag contains no slash."""
    repository, separator, tag = image.rpartition(":")
    if not separator or not repository or "/" in tag:
        return image, "latest"
    return repository, tag


def chat_completion(client: NemoClient, workspace: str, model: str) -> dict[str, Any]:
    """Call an existing inference-gateway model from the test process."""
    request_workspace, body_model = chat_target(model, workspace)
    body = (
        InferenceGatewayClient.from_client(client)
        .openai_post(
            workspace=request_workspace,
            trailing_uri="v1/chat/completions",
            body=JsonBody(
                {
                    "model": body_model,
                    "messages": [{"role": "user", "content": "Reply with the single word ok."}],
                    "max_tokens": 16,
                }
            ),
        )
        .data()
    )
    if isinstance(body, dict):
        return body
    raise TypeError(f"chat completion returned {type(body).__name__}")


def show_report(kind: str, report: dict[str, Any], notes: list[str]) -> None:
    """Print the auth report and any non-failing notes."""
    print(f"\n=== {kind} auth report ===", flush=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    for note in notes:
        print(f"note: {note}", flush=True)
