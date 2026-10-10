# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance checks for one running NeMo Helix instance.

Run ``tools/instance_check/check.py``. It is not part of ``make test-unit``.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NotFoundError, PermissionDeniedError, UnprocessableEntityError
from nemo_helix_plugin.client.oidc import NHXOIDCConfig
from nemo_helix_plugin.deployments.client import DeploymentsClient
from nemo_helix_plugin.deployments.types import (
    CreateDeploymentConfigRequest,
    CreateDeploymentRequest,
    RequestEnvVar,
)
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.providers import ContainerSpec, CPUExecutionProvider
from nemo_helix_plugin.jobs.spec import (
    HelixJobEnvironmentVariable,
    HelixJobSpec,
    HelixJobStepSpec,
)
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest, HelixJobResponse
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import (
    ContainerExecutorConfig,
    CreateModelDeploymentConfigRequest,
    CreateModelDeploymentRequest,
    Engine,
    ModelDeploymentConfigModelSpec,
)
from nemo_helix_plugin.secrets.client import SecretsClient
from probe import evaluate_auth_report, workload_probe_script
from support import (
    DEPLOYMENT_SUCCESS,
    FILESET_NAME,
    JOB_SUCCESS,
    MODEL_SUCCESS,
    auth_flags,
    auth_profile,
    chat_completion,
    is_capacity_error,
    load_deployment_report,
    load_job_report,
    load_settings,
    read_platform_probe,
    report_path,
    retry_until_workspace_granted,
    show_report,
    split_image,
    wait_for_deployment,
    wait_for_job,
)

_PROBE = workload_probe_script()


def test_platform_ready(platform_client: NemoClient) -> None:
    """The instance answers a health or cluster-info probe."""
    path, payload = read_platform_probe(platform_client)
    print(f"\nplatform probe {path}", flush=True)
    if path == "/health/ready":
        assert payload.get("status") == "ready", payload
    elif path == "/status":
        assert payload.get("status") != "unhealthy", payload
    else:
        assert payload.get("platform_version") or payload.get("revision"), payload


def test_auth_discovery(auth_discovery: NHXOIDCConfig) -> None:
    """Print discovery so a manual run shows whether auth and token exchange are on."""
    print("\n=== auth discovery ===", flush=True)
    print(f"auth profile: {auth_profile(auth_discovery)}", flush=True)
    print(json.dumps(asdict(auth_discovery), indent=2, sort_keys=True), flush=True)


def test_secret_round_trip(
    platform_client: NemoClient,
    workspace: str,
    check_secret: dict[str, str],
) -> None:
    """A created secret can be read back, and the read does not return the value."""
    secrets = SecretsClient.from_client(platform_client)
    fetched = retry_until_workspace_granted(
        lambda: secrets.get_secret(workspace=workspace, name=check_secret["name"]).data()
    )
    assert fetched.name == check_secret["name"]
    assert check_secret["value"] not in fetched.model_dump_json()


def test_secret_value_access(
    platform_client: NemoClient,
    workspace: str,
    check_secret: dict[str, str],
    auth_discovery: NHXOIDCConfig,
) -> None:
    """Secret values follow the auth profile.

    With auth disabled, the caller can read the value. With auth enabled, a
    user token is denied and only service credentials can read it.
    """
    profile = auth_profile(auth_discovery)
    secrets = SecretsClient.from_client(platform_client)
    try:
        accessed = secrets.access_secret(workspace=workspace, name=check_secret["name"]).data()
    except UnprocessableEntityError as exc:
        if "Not allowed to list entities" not in str(exc):
            raise
        accessed = retry_until_workspace_granted(
            lambda: secrets.access_secret(workspace=workspace, name=check_secret["name"]).data()
        )
    except PermissionDeniedError as exc:
        if profile == "anonymous":
            pytest.fail(f"secret value access was denied on a no-auth server: {exc}")
        print(f"secret value access denied for the caller, as required: {exc}", flush=True)
        return
    if profile != "anonymous":
        pytest.fail("a user token read a secret value; auth-enabled servers reserve that for service credentials")
    assert accessed.value == check_secret["value"]
    print("secret value access ok", flush=True)


def test_job_auth_probe(
    platform_client: NemoClient,
    workspace: str,
    auth_discovery: NHXOIDCConfig,
    task_image: str | None,
    selected_model: str | None,
) -> None:
    """A job can call an authenticated API, and its auth report matches discovery."""
    settings = load_settings()
    jobs = JobsClient.from_client(platform_client)
    report_id = uuid.uuid4().hex
    container: dict[str, object] = {"entrypoint": ["python", "-c"], "command": [_PROBE]}
    if task_image:
        container["image"] = task_image
    created = jobs.create_job(
        workspace=workspace,
        body=CreateHelixJobRequest(
            source="instance-check",
            spec={},
            platform_spec=HelixJobSpec(
                steps=[
                    HelixJobStepSpec(
                        name="check",
                        executor=CPUExecutionProvider(
                            provider="cpu",
                            profile=settings.job_profile or "default",
                            container=ContainerSpec.model_validate(container),
                        ),
                        environment=_job_environment(workspace, selected_model, report_id),
                        config={"workspace": workspace},
                    )
                ]
            ),
        ),
    ).data()
    try:
        job = wait_for_job(platform_client, workspace, created.name, settings.timeout)
        report = _job_report_or_fail(platform_client, workspace, job, report_id)
        _assert_report(
            "job",
            report,
            auth_discovery,
            model_required=selected_model is not None,
            status_problem=_job_status_problem(job),
        )
    finally:
        if not settings.keep:
            _ignore_missing(lambda: jobs.delete_job(workspace=workspace, name=created.name))


def test_deployment_auth_probe(
    platform_client: NemoClient,
    workspace: str,
    check_secret: dict[str, str],
    auth_discovery: NHXOIDCConfig,
    task_image: str | None,
    selected_model: str | None,
) -> None:
    """A deployment calls the platform with the identity its auth profile provides.

    Trusted-headers servers authenticate deployments through the auth-proxy
    sidecar. The public deployment-config API cannot set that sidecar, so this
    probe is off for that profile and ``test_deployment_runs`` covers execution.
    """
    profile = auth_profile(auth_discovery)
    if profile == "trusted_headers":
        pytest.skip("trusted-headers deployment identity is not settable through the public deployment-config API")
    if not task_image:
        pytest.skip("no task image; pass --image")
    settings = load_settings()
    report_id = uuid.uuid4().hex
    config_name = f"check-cfg-{uuid.uuid4().hex[:8]}"
    deployment_name = f"check-dep-{uuid.uuid4().hex[:8]}"
    deployments = DeploymentsClient.from_client(platform_client)
    deployments.create_deployment_config(
        workspace=workspace,
        body=CreateDeploymentConfigRequest.model_validate(
            {
                "name": config_name,
                "restart_policy": "Never",
                "backoff_limit": 1,
                "containers": [
                    {
                        "name": "check",
                        "image": task_image,
                        "command": ["python", "-c"],
                        "args": [_PROBE],
                        "env": _deployment_environment(
                            workspace,
                            check_secret,
                            selected_model,
                            profile == "token_exchange",
                            str(platform_client.base_url).rstrip("/"),
                            report_id,
                        ),
                    }
                ],
                **({"workload_identity": {"enabled": True}} if profile == "token_exchange" else {}),
            }
        ),
    )
    try:
        deployments.create_deployment(
            workspace=workspace,
            body=CreateDeploymentRequest(
                name=deployment_name,
                deployment_config=config_name,
                desired_state="READY",
                executor=settings.executor,
            ),
        )
        deployment = wait_for_deployment(platform_client, workspace, deployment_name, settings.timeout)
        try:
            report = load_deployment_report(platform_client, workspace, report_path("deployment", report_id), report_id)
        except RuntimeError as exc:
            payload = deployment.model_dump(mode="json")
            raise RuntimeError(
                f"{exc} status={payload.get('status')!r} message={payload.get('status_message')!r} "
                f"history={payload.get('status_history')!r}"
            ) from exc
        status = str(deployment.status)
        status_problem = (
            None if status in DEPLOYMENT_SUCCESS else f"deployment ended {status}: {deployment.status_message}"
        )
        _assert_report(
            "deployment",
            report,
            auth_discovery,
            model_required=selected_model is not None,
            status_problem=status_problem,
        )
    finally:
        if not settings.keep:
            _delete_deployment(deployments, workspace, deployment_name, config_name)


def test_deployment_runs(
    platform_client: NemoClient,
    workspace: str,
    auth_discovery: NHXOIDCConfig,
    task_image: str | None,
) -> None:
    """A trusted-headers server can start a one-shot deployment and have it finish.

    The job probe covers trusted-header identity. This check only proves the
    deployment itself runs.
    """
    if auth_profile(auth_discovery) != "trusted_headers":
        pytest.skip("this profile's deployment check is the auth probe")
    if not task_image:
        pytest.skip("no task image; pass --image")
    settings = load_settings()
    config_name = f"check-cfg-{uuid.uuid4().hex[:8]}"
    deployment_name = f"check-dep-{uuid.uuid4().hex[:8]}"
    deployments = DeploymentsClient.from_client(platform_client)
    deployments.create_deployment_config(
        workspace=workspace,
        body=CreateDeploymentConfigRequest.model_validate(
            {
                "name": config_name,
                "restart_policy": "Never",
                "backoff_limit": 1,
                "containers": [
                    {
                        "name": "check",
                        "image": task_image,
                        "command": ["python", "-c"],
                        "args": ["raise SystemExit(0)"],
                    }
                ],
            }
        ),
    )
    try:
        deployments.create_deployment(
            workspace=workspace,
            body=CreateDeploymentRequest(
                name=deployment_name,
                deployment_config=config_name,
                desired_state="READY",
                executor=settings.executor,
            ),
        )
        deployment = wait_for_deployment(platform_client, workspace, deployment_name, settings.timeout)
        status = str(deployment.status)
        assert status in DEPLOYMENT_SUCCESS, f"deployment ended {status}: {deployment.status_message}"
    finally:
        if not settings.keep:
            _delete_deployment(deployments, workspace, deployment_name, config_name)


def test_model_chat(
    platform_client: NemoClient,
    workspace: str,
    selected_model: str | None,
) -> None:
    """Call an existing inference-gateway model when ``--models`` is set."""
    settings = load_settings()
    if not settings.models:
        pytest.skip("pass --models to call an existing model")
    if selected_model is None:
        pytest.skip("no models are listed; pass --model NAME to require one")
    body = chat_completion(platform_client, workspace, selected_model)
    choices = body.get("choices")
    assert choices, body


def test_model_deploy(platform_client: NemoClient, workspace: str) -> None:
    """Deploy a CPU container through the models API and wait until it is ready.

    Uses ``--model-image`` when set, otherwise ``docker.io/library/python:3.12-alpine``.
    A missing GPU or capacity error is a skip.
    """
    settings = load_settings()
    print(f"model deploy image: {settings.model_image}", flush=True)
    models = ModelsClient.from_client(platform_client)
    config_name = f"check-mcfg-{uuid.uuid4().hex[:8]}"
    deployment_name = f"check-mdep-{uuid.uuid4().hex[:8]}"
    repository, tag = split_image(settings.model_image)
    created = False
    try:
        models.create_deployment_config(
            workspace=workspace,
            body=CreateModelDeploymentConfigRequest(
                name=config_name,
                engine=Engine.GENERIC,
                model_spec=ModelDeploymentConfigModelSpec(),
                executor_config=ContainerExecutorConfig(
                    gpu=0,
                    image_name=repository,
                    image_tag=tag,
                    additional_args=["python3", "-m", "http.server", "8000"],
                    health_check_path="/",
                ),
            ),
        )
        try:
            models.create_deployment(
                workspace=workspace,
                body=CreateModelDeploymentRequest(name=deployment_name, config=config_name),
            )
            created = True
            deployment = _wait_for_model(models, workspace, deployment_name, settings.timeout)
        except Exception as exc:
            if is_capacity_error({"error": str(exc)}):
                pytest.skip(f"model deployment has no capacity: {exc}")
            raise
        status = str(getattr(deployment.status, "value", deployment.status))
        payload = deployment.model_dump(mode="json")
        message = str(payload.get("status_message") or "")
        if status not in MODEL_SUCCESS and is_capacity_error({"status_message": message}):
            pytest.skip(f"model deployment has no capacity: {message or status}")
        assert status in MODEL_SUCCESS, payload
    finally:
        if created:
            try:
                models.delete_deployment(workspace=workspace, name=deployment_name)
            except NotFoundError:
                pass
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    current = models.get_deployment(workspace=workspace, name=deployment_name).data()
                except NotFoundError:
                    break
                status = str(getattr(current.status, "value", current.status))
                if status == "DELETED":
                    break
                time.sleep(1)
        _ignore_missing(lambda: models.delete_deployment_config(workspace=workspace, name=config_name))


def _job_environment(workspace: str, model: str | None, report_id: str) -> list[HelixJobEnvironmentVariable]:
    environment = [
        HelixJobEnvironmentVariable(name="INSTANCE_CHECK_KIND", value="job"),
        HelixJobEnvironmentVariable(name="INSTANCE_CHECK_SERVICE", value="jobs"),
        HelixJobEnvironmentVariable(name="INSTANCE_CHECK_WORKSPACE", value=workspace),
        HelixJobEnvironmentVariable(name="INSTANCE_CHECK_FILESET", value=FILESET_NAME),
        HelixJobEnvironmentVariable(name="INSTANCE_CHECK_REPORT_ID", value=report_id),
        HelixJobEnvironmentVariable(name="INSTANCE_CHECK_REPORT_PATH", value=report_path("job", report_id)),
    ]
    if model:
        environment.append(HelixJobEnvironmentVariable(name="INSTANCE_CHECK_MODEL", value=model))
    return environment


def _deployment_environment(
    workspace: str,
    secret: dict[str, str],
    model: str | None,
    workload_identity: bool,
    platform_url: str | None,
    report_id: str,
) -> list[RequestEnvVar]:
    environment = [
        RequestEnvVar(name="INSTANCE_CHECK_KIND", value="deployment"),
        RequestEnvVar(name="INSTANCE_CHECK_SERVICE", value="deployments"),
        RequestEnvVar(name="INSTANCE_CHECK_WORKSPACE", value=workspace),
        RequestEnvVar(name="INSTANCE_CHECK_FILESET", value=FILESET_NAME),
        RequestEnvVar(name="INSTANCE_CHECK_REPORT_ID", value=report_id),
        RequestEnvVar(name="INSTANCE_CHECK_REPORT_PATH", value=report_path("deployment", report_id)),
        RequestEnvVar(name="INSTANCE_CHECK_SECRET_SHA256", value=secret["sha256"]),
        RequestEnvVar.model_validate(
            {
                "name": "INSTANCE_CHECK_SECRET",
                "secret_ref": {"workspace": workspace, "name": secret["name"]},
            }
        ),
    ]
    if platform_url:
        environment.append(RequestEnvVar(name="NHX_BASE_URL", value=platform_url))
    if model:
        environment.append(RequestEnvVar(name="INSTANCE_CHECK_MODEL", value=model))
    if workload_identity:
        environment.append(RequestEnvVar(name="INSTANCE_CHECK_WORKLOAD_IDENTITY", value="1"))
    return environment


def _job_report_or_fail(client: NemoClient, workspace: str, job: HelixJobResponse, report_id: str) -> dict[str, Any]:
    try:
        return load_job_report(client, workspace, job.name, report_path("job", report_id), report_id)
    except RuntimeError as exc:
        pytest.fail(f"{exc}\n{_job_status_problem(job) or ''}")


def _job_status_problem(job: HelixJobResponse) -> str | None:
    status = str(getattr(job.status, "value", job.status))
    if status in JOB_SUCCESS:
        return None
    return f"job {job.name} ended {status}: {job.status_details} {job.error_details}"


def _assert_report(
    kind: str,
    report: dict[str, Any],
    discovery: NHXOIDCConfig,
    *,
    model_required: bool,
    status_problem: str | None,
) -> None:
    settings = load_settings()
    enabled, exchange = auth_flags(discovery)
    evaluation = evaluate_auth_report(
        report,
        auth_enabled=enabled,
        token_exchange_enabled=exchange,
        require_token_exchange=settings.require_token_exchange,
        model_required=model_required,
    )
    show_report(kind, report, evaluation.notes)
    problems = list(evaluation.problems)
    if status_problem:
        problems.insert(0, status_problem)
    if problems:
        pytest.fail("\n".join(problems))


def _wait_for_model(models: ModelsClient, workspace: str, name: str, timeout: float) -> Any:
    deadline = time.monotonic() + timeout
    last = ""
    deployment = models.get_deployment(workspace=workspace, name=name).data()
    while time.monotonic() < deadline:
        deployment = models.get_deployment(workspace=workspace, name=name).data()
        status = str(getattr(deployment.status, "value", deployment.status))
        if status != last:
            print(f"model {name}: {status}", flush=True)
            last = status
        if status in {"READY", "ERROR", "DELETED", "FAILED", "LOST", "UNKNOWN"}:
            return deployment
        time.sleep(2)
    raise TimeoutError(f"model {name} still {last or 'unknown'} after {timeout:.0f}s")


def _delete_deployment(
    deployments: DeploymentsClient,
    workspace: str,
    deployment_name: str,
    config_name: str,
) -> None:
    """Delete the deployment, then its config once nothing still references it."""
    try:
        deployments.delete_deployment(workspace=workspace, name=deployment_name)
    except NotFoundError:
        pass
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            deployments.get_deployment(workspace=workspace, name=deployment_name)
        except NotFoundError:
            break
        time.sleep(1)
    _ignore_missing(lambda: deployments.delete_deployment_config(workspace=workspace, name=config_name))


def _ignore_missing(action: Callable[[], object]) -> None:
    try:
        action()
    except NotFoundError:
        return
    except Exception as exc:
        print(f"cleanup failed: {type(exc).__name__}: {exc}", flush=True)
