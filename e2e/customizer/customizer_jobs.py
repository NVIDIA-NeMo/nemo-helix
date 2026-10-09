# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job submission, vLLM deploy, and teardown helpers for customizer GPU e2e tests."""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import httpx
import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoTransportError, NotFoundError
from nemo_helix_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_helix_plugin.inference_gateway.types import JsonBody
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.types import HelixJobResponse
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import (
    ContainerExecutorConfig,
    CreateModelDeploymentConfigRequest,
    CreateModelDeploymentRequest,
    Engine,
    ListDeploymentConfigsQueryParams,
    ListDeploymentsQueryParams,
    ModelDeploymentConfigModelSpec,
)
from nhx.customization_common.sdk.client import CustomizationClient
from nhx.customization_common.sdk.types import CustomizationJobCreateRequest
from nhx.testing.e2e import wait_for_platform_job
from pydantic import BaseModel

from e2e.customizer.customization_helpers import _format_status, unique_name

logger = logging.getLogger(__name__)

_DOCKER_LOG_TAIL_LINES = 300
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)\b("
    r"HF_TOKEN|NGC_API_KEY|NEMO_JOBS_IMAGE_REGISTRY_PASSWORD|AWS_SECRET_ACCESS_KEY|"
    r"TOKEN|PASSWORD|SECRET|API_KEY"
    r")=([^\s,;]+)"
)
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def get_job_failure_details(client: NemoClient, job_name: str, workspace: str) -> str:
    """Fetch job status and recent logs for a failed customization job."""
    details = [f"Job {job_name} failed. Details:"]
    jobs = JobsClient.from_client(client)

    try:
        job_status = jobs.get_job_status(name=job_name, workspace=workspace).data()
        details.append(f"\nJob Status: {job_status.model_dump_json(indent=2)}")
    except Exception as exc:
        details.append(f"\nFailed to get job status: {exc}")

    try:
        entries = list(jobs.list_job_logs(name=job_name, workspace=workspace).items())
        if entries:
            details.append("\nJob Logs (last 20 entries):")
            for entry in entries[-20:]:
                details.append(f"  [{entry.job_step}] {entry.message}")
        else:
            details.append("\nNo logs available")
    except Exception as exc:
        details.append(f"\nFailed to get job logs: {exc}")

    return "\n".join(details)


def get_deployment_failure_details(
    client: NemoClient,
    workspace: str,
    deployment_name: str,
    deployment_config_name: str | None = None,
) -> str:
    """Fetch deployment/config details for a failed inference deployment."""
    details = [f"Deployment {deployment_name} failed. Details:"]
    models = ModelsClient.from_client(client)

    try:
        deployment_status = models.get_deployment(name=deployment_name, workspace=workspace).data()
        details.extend(["", "Deployment Status:", _format_status(deployment_status)])
    except Exception as exc:
        details.extend(["", f"Failed to get deployment status: {exc}"])

    _append_substrate_deployment_details(client, workspace, deployment_name, details)

    if deployment_config_name:
        try:
            deployment_config = models.get_deployment_config(name=deployment_config_name, workspace=workspace).data()
            details.extend(["", "Deployment Config:", _format_status(deployment_config)])
        except Exception as exc:
            details.extend(["", f"Failed to get deployment config: {exc}"])

    return "\n".join(details)


def _append_substrate_deployment_details(
    client: NemoClient,
    workspace: str,
    deployment_name: str,
    details: list[str],
) -> None:
    """Append raw deployments-plugin status for model deployment substrate."""
    base_url = str(client.base_url).rstrip("/")
    for substrate_name in _substrate_deployment_names(deployment_name):
        path = (
            f"{base_url}/apis/deployments/v2/workspaces/{quote(workspace, safe='')}"
            f"/deployments/{quote(substrate_name, safe='')}"
        )
        try:
            response = httpx.get(path, timeout=5.0)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            details.extend(
                [
                    "",
                    f"Substrate Deployment {substrate_name}:",
                    json.dumps(response.json(), indent=2, sort_keys=True),
                ]
            )
        except Exception as exc:
            details.extend(["", f"Failed to get substrate deployment {substrate_name}: {exc}"])


def _substrate_deployment_names(deployment_name: str) -> tuple[str, str]:
    return (f"{deployment_name}-puller", f"{deployment_name}-server")


def _collect_substrate_docker_logs(workspace: str, deployment_name: str) -> None:
    """Collect deployments-plugin Docker container logs before model cleanup removes them."""
    try:
        import docker
    except Exception as exc:
        logger.warning("Docker SDK unavailable; cannot collect substrate logs: %s", exc)
        return

    log_dir = Path(os.environ.get("JOB_LOGS_DIR", "docker/customizer-logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        client = docker.from_env()
    except Exception as exc:
        logger.warning("Docker client unavailable; cannot collect substrate logs: %s", exc)
        return

    try:
        for substrate_name in _substrate_deployment_names(deployment_name):
            filters = {
                "label": [
                    "managed-by=nemo-deployments",
                    f"nemo.nvidia.com/deployment-workspace={workspace}",
                    f"nemo.nvidia.com/deployment-name={substrate_name}",
                ]
            }
            containers = client.containers.list(all=True, filters=filters)
            if not containers:
                (log_dir / f"containers-{_safe_filename(substrate_name)}.txt").write_text(
                    f"No Docker containers found for {workspace}/{substrate_name}\n"
                )
                continue
            for container in containers:
                _write_container_diagnostics(log_dir, container, substrate_name)
    except Exception as exc:
        logger.warning("Failed to collect substrate Docker logs: %s", exc, exc_info=True)
    finally:
        try:
            client.close()
        except Exception:
            pass


def _write_container_diagnostics(log_dir: Path, container: Any, substrate_name: str) -> None:
    name = _safe_filename(getattr(container, "name", "") or substrate_name)
    try:
        container.reload()
    except Exception:
        logger.debug("Could not reload container %s before diagnostics", name, exc_info=True)

    attrs = getattr(container, "attrs", {}) or {}
    config = attrs.get("Config") if isinstance(attrs, dict) else {}
    state = attrs.get("State") if isinstance(attrs, dict) else {}
    image = config.get("Image") if isinstance(config, dict) else None
    labels = config.get("Labels") if isinstance(config, dict) else None
    summary = {
        "id": getattr(container, "id", None),
        "name": getattr(container, "name", None),
        "status": getattr(container, "status", None),
        "image": image,
        "state": state,
        "labels": labels,
    }
    (log_dir / f"inspect-{name}.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))

    try:
        raw_logs = container.logs(stdout=True, stderr=True, timestamps=True, tail=_DOCKER_LOG_TAIL_LINES)
        text = raw_logs.decode("utf-8", errors="replace") if isinstance(raw_logs, bytes) else str(raw_logs)
    except Exception as exc:
        text = f"Failed to get logs for {substrate_name}: {exc}\n"
    (log_dir / f"logs-{name}.txt").write_text(_redact_sensitive_values(text))


def _safe_filename(value: str) -> str:
    return _SAFE_FILENAME_RE.sub("-", value).strip("-") or "container"


def _redact_sensitive_values(text: str) -> str:
    return _SENSITIVE_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=<redacted>", text)


def require_kubernetes_backend(client: NemoClient) -> None:
    """Skip unless the platform has a Kubernetes job backend, which rl jobs require."""
    try:
        profiles = JobsClient.from_client(client).list_execution_profiles()
    except Exception as exc:
        pytest.skip(f"could not list execution profiles to verify rl backend: {exc}")
    backends = {profile.backend for profile in profiles}
    if not backends & {"kubernetes_job", "volcano_job"}:
        pytest.skip("rl requires a kubernetes_job or volcano_job execution backend")


def submit_and_wait_customization_job(
    client: NemoClient,
    backend: str,
    spec: BaseModel,
    workspace: str,
    *,
    name: str | None = None,
    timeout: float = 3600,
    image_pull_timeout: float = 1800,
    poll_interval: float = 30,
) -> tuple[str, HelixJobResponse]:
    """Submit a customization job to ``backend`` and wait for a terminal state."""
    job = (
        CustomizationClient.from_client(client)
        .create_customization_job(
            workspace=workspace,
            backend=backend,
            body=CustomizationJobCreateRequest(spec=spec.model_dump(mode="json"), name=name),
        )
        .data()
    )
    logger.info("Submitted %s customization job: %s", backend, job.name)
    final_job = wait_for_platform_job(
        client,
        job.name,
        workspace,
        timeout=timeout,
        image_pull_timeout=image_pull_timeout,
        poll_interval=poll_interval,
    )
    logger.info("Customization job %s finished with status: %s", job.name, final_job.status)
    return job.name, final_job


def deploy_vllm_model(
    client: NemoClient,
    workspace: str,
    model_entity: str,
    *,
    lora_enabled: bool = False,
    max_lora_rank: int = 16,
    gpu: int = 1,
    disk_size: str | None = None,
    additional_args: list[str] | None = None,
    ready_timeout: int = 3600,
    gateway_timeout: int = 120,
) -> tuple[str, str]:
    """Deploy ``model_entity`` with vLLM and wait until inference is routable.

    The models service picks the vLLM image, injects ``--enable-lora`` when
    ``lora_enabled`` is set, and sizes ``--tensor-parallel-size`` from ``gpu`` and the
    model spec unless ``additional_args`` sets it.
    """
    args = list(additional_args or [])
    if lora_enabled:
        args.extend(["--max-lora-rank", str(max_lora_rank)])
    executor_config: dict[str, object] = {"gpu": gpu, "additional_args": args}
    if disk_size is not None:
        executor_config["disk_size"] = disk_size

    deployment_config_name = unique_name("e2e-vllm-config")
    deployment_name = unique_name("e2e-vllm")
    models = ModelsClient.from_client(client)

    logger.info("Deploying %s/%s on vLLM (lora_enabled=%s, gpu=%s)", workspace, model_entity, lora_enabled, gpu)
    models.create_deployment_config(
        workspace=workspace,
        body=CreateModelDeploymentConfigRequest(
            name=deployment_config_name,
            description=f"E2E vLLM deployment for {model_entity}",
            model_entity_id=model_entity,
            engine=Engine.VLLM,
            model_spec=ModelDeploymentConfigModelSpec(
                model_name=model_entity,
                model_namespace=workspace,
                lora_enabled=lora_enabled,
            ),
            executor_config=ContainerExecutorConfig.model_validate(executor_config),
        ),
    )
    models.create_deployment(
        workspace=workspace,
        body=CreateModelDeploymentRequest(name=deployment_name, config=deployment_config_name),
    )
    try:
        _wait_for_deployment_ready(
            client,
            workspace,
            deployment_name,
            deployment_config_name=deployment_config_name,
            timeout=ready_timeout,
        )
        _wait_for_gateway_ready(client, workspace, deployment_name, timeout=gateway_timeout)
    except BaseException:
        # Readiness helpers raise (pytest.fail) before we can return the names to the
        # caller, so the caller's try/finally never runs — clean up here to avoid
        # leaking a GPU deployment on every readiness failure.
        logger.warning("Deployment %s failed to become ready; tearing it down", deployment_name)
        _collect_substrate_docker_logs(workspace, deployment_name)
        delete_deployment(client, workspace, deployment_name, deployment_config_name, wait_seconds=0)
        raise
    logger.info("Deployment %s ready and routable", deployment_name)
    return deployment_name, deployment_config_name


def delete_deployment(
    client: NemoClient,
    workspace: str,
    deployment_name: str,
    deployment_config_name: str | None = None,
    *,
    wait_seconds: int = 30,
) -> None:
    """Best-effort teardown of a deployment and its config."""
    models = ModelsClient.from_client(client)
    try:
        models.delete_deployment(name=deployment_name, workspace=workspace)
        logger.info("Deleted deployment %s", deployment_name)
        if wait_seconds:
            time.sleep(wait_seconds)
    except Exception as exc:
        logger.warning("Failed to delete deployment %s: %s", deployment_name, exc)
    if deployment_config_name:
        try:
            models.delete_deployment_config(name=deployment_config_name, workspace=workspace)
            logger.info("Deleted deployment config %s", deployment_config_name)
        except Exception as exc:
            logger.warning("Failed to delete deployment config %s: %s", deployment_config_name, exc)


def create_vllm_deployment_template(
    client: NemoClient,
    workspace: str,
    *,
    lora_enabled: bool,
    max_lora_rank: int = 16,
    gpu: int = 1,
) -> str:
    """Create an unbound vLLM deployment config to pass as a job's ``deployment_config``.

    It names no model, so the job's model_entity step binds it to the trained output: a
    full-weight output gets its own deployment, and a LoRA adapter is served from a
    deployment of its base model.
    """
    name = unique_name("e2e-vllm-template")
    args = ["--max-lora-rank", str(max_lora_rank)] if lora_enabled else []
    ModelsClient.from_client(client).create_deployment_config(
        workspace=workspace,
        body=CreateModelDeploymentConfigRequest(
            name=name,
            description="E2E vLLM template for customization auto-deploy",
            engine=Engine.VLLM,
            model_spec=ModelDeploymentConfigModelSpec(lora_enabled=lora_enabled),
            executor_config=ContainerExecutorConfig.model_validate({"gpu": gpu, "additional_args": args}),
        ),
    )
    logger.info("Created vLLM deployment template %s/%s (lora_enabled=%s)", workspace, name, lora_enabled)
    return name


def wait_for_auto_deployment(
    client: NemoClient,
    workspace: str,
    model_entity: str,
    *,
    find_timeout: int = 300,
    ready_timeout: int = 3600,
    gateway_timeout: int = 120,
) -> tuple[str, str]:
    """Wait for the deployment a customization job created for ``model_entity`` to serve.

    The job's model_entity step only creates the deployment. Find it the way that step
    does (configs by ``model_entity_id``, then deployments by config), then wait until it
    is ready and routable through the inference gateway.
    """
    deployment_name, config_name = _find_model_deployment(client, workspace, model_entity, timeout=find_timeout)
    try:
        _wait_for_deployment_ready(
            client, workspace, deployment_name, deployment_config_name=config_name, timeout=ready_timeout
        )
        _wait_for_gateway_ready(client, workspace, deployment_name, timeout=gateway_timeout)
    except BaseException:
        logger.warning("Deployment %s failed to become ready; tearing it down", deployment_name)
        _collect_substrate_docker_logs(workspace, deployment_name)
        delete_deployment(client, workspace, deployment_name, config_name, wait_seconds=0)
        raise
    logger.info("Deployment %s ready and routable", deployment_name)
    return deployment_name, config_name


def _find_model_deployment(
    client: NemoClient, workspace: str, model_entity: str, *, timeout: int, poll_interval: int = 10
) -> tuple[str, str]:
    models = ModelsClient.from_client(client)
    config_query = ListDeploymentConfigsQueryParams(
        filter=json.dumps({"model_entity_id": f"{workspace}/{model_entity}"})
    )
    deadline = time.time() + timeout
    while True:
        for config in models.list_deployment_configs(workspace=workspace, query_params=config_query).items():
            deployment_query = ListDeploymentsQueryParams(
                filter=json.dumps({"config": config.name, "workspace": workspace})
            )
            for deployment in models.list_deployments(workspace=workspace, query_params=deployment_query).items():
                logger.info(
                    "Found deployment %s (config %s) for %s/%s", deployment.name, config.name, workspace, model_entity
                )
                return deployment.name, config.name
        if time.time() > deadline:
            pytest.fail(
                f"No deployment for {workspace}/{model_entity} appeared within {timeout}s of the job finishing."
            )
        time.sleep(poll_interval)


def assert_chat_completion(client: NemoClient, workspace: str, deployment_name: str, model_field: str) -> None:
    """Send one chat completion through the inference gateway and assert it returns text."""
    response = cast(
        dict[str, Any],
        InferenceGatewayClient.from_client(client)
        .provider_post(
            trailing_uri="v1/chat/completions",
            name=deployment_name,
            workspace=workspace,
            body=JsonBody(
                {
                    "model": model_field,
                    "messages": [{"role": "user", "content": "What is 2 + 2?"}],
                    "max_tokens": 32,
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            ),
        )
        .data(),
    )
    content = response["choices"][0]["message"]["content"]
    assert isinstance(content, str) and content.strip(), f"{model_field} returned no text: {response}"
    logger.info("%s answered: %r", model_field, content)


def assert_embeddings(client: NemoClient, workspace: str, deployment_name: str, model_field: str) -> None:
    """Embed one query through the inference gateway and assert it returns a vector."""
    response = cast(
        dict[str, Any],
        InferenceGatewayClient.from_client(client)
        .provider_post(
            trailing_uri="v1/embeddings",
            name=deployment_name,
            workspace=workspace,
            body=JsonBody({"model": model_field, "input": ["query: What does NeMo Helix customize?"]}),
        )
        .data(),
    )
    embedding = response["data"][0]["embedding"]
    assert embedding, f"{model_field} returned an empty embedding: {response}"
    logger.info("%s returned a %d-dimensional embedding", model_field, len(embedding))


def training_metric_values(client: NemoClient, workspace: str, job_name: str, metric: str) -> list[float]:
    """Return a recorded training metric's values in step order.

    Training tasks store each metric's history under ``status_details.metrics`` as
    ``{"step", "epoch", "value"}`` points.
    """
    status = JobsClient.from_client(client).get_job_status(name=job_name, workspace=workspace).data()
    recorded: set[str] = set()
    for step in status.steps:
        for task in step.tasks:
            metrics = task.status_details.get("metrics") or {}
            recorded.update(metrics)
            if points := metrics.get(metric):
                return [float(point["value"]) for point in sorted(points, key=lambda point: point["step"])]
    pytest.fail(f"Job {job_name} recorded no {metric!r} series; recorded metrics: {sorted(recorded)}")


def _wait_for_gateway_ready(
    client: NemoClient,
    workspace: str,
    deployment_name: str,
    *,
    timeout: int = 120,
    poll_interval: float = 2.0,
) -> None:
    logger.info("Waiting for inference gateway to sync...")
    gateway = InferenceGatewayClient.from_client(client)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            gateway.provider_ready(name=deployment_name, workspace=workspace)
            logger.info("Inference gateway is ready")
            return
        except (NotFoundError, NemoTransportError):
            time.sleep(poll_interval)
    pytest.fail(f"Inference gateway did not become ready for deployment '{deployment_name}' within {timeout}s.")


def _wait_for_deployment_ready(
    client: NemoClient,
    workspace: str,
    deployment_name: str,
    *,
    deployment_config_name: str | None = None,
    timeout: int = 3600,
    poll_interval: int = 30,
    max_consecutive_errors: int = 5,
    ready_statuses: tuple[str, ...] = ("READY",),
) -> None:
    logger.info("Waiting for deployment to be ready...")
    models = ModelsClient.from_client(client)
    start_time = time.time()
    consecutive_errors = 0

    while True:
        try:
            deployment_status = models.get_deployment(name=deployment_name, workspace=workspace).data()
            consecutive_errors = 0
        except (NemoTransportError, httpx.TimeoutException, httpx.ConnectError, ConnectionError, OSError) as exc:
            consecutive_errors += 1
            elapsed = time.time() - start_time
            logger.warning(
                "Transient error polling deployment (attempt %d/%d, elapsed %.0fs): %s",
                consecutive_errors,
                max_consecutive_errors,
                elapsed,
                exc,
            )
            if consecutive_errors >= max_consecutive_errors:
                pytest.fail(
                    f"Deployment API unreachable after {consecutive_errors} consecutive errors "
                    f"(elapsed: {elapsed:.0f}s). Last error: {exc}"
                )
            time.sleep(poll_interval)
            continue

        if deployment_status.status in ready_statuses:
            break
        if deployment_status.status in ("ERROR", "LOST"):
            logger.error("Deployment entered terminal failure state: %s", deployment_status.status)
            pytest.fail(get_deployment_failure_details(client, workspace, deployment_name, deployment_config_name))
        if time.time() - start_time > timeout:
            pytest.fail(
                f"Deployment did not become ready within timeout.\n\n"
                f"{get_deployment_failure_details(client, workspace, deployment_name, deployment_config_name)}"
            )

        logger.info("Deployment status: %s, waiting...", deployment_status.status)
        time.sleep(poll_interval)

    logger.info("Deployment is ready")
