# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Customizer E2E tests."""

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from nemo_helix import NeMoHelix
from nemo_helix.types.inference import ContainerExecutorConfigParam, ModelDeploymentConfigModelSpecParam
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import NemoTransportError
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.models.client import ModelsClient

logger = logging.getLogger(__name__)

LOGS_DIR = Path("e2e-logs")

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "error"})


@dataclass(frozen=True)
class TrainingTypeConfig:
    """Per-training-type configuration parsed from the TRAINING_TYPES_OVERRIDES JSON env var."""

    data_format: str = "prompt_completion"
    skip_deployment: bool = False
    nim_image_name: str = "nvcr.io/nim/nvidia/llm-nim"
    nim_image_tag: str = "1.15.5"
    num_gpus_per_node: int = 1
    num_nodes: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    teacher_model_hf_repo: str = ""


def parse_training_types(raw: str) -> dict[str, TrainingTypeConfig]:
    """Parse the TRAINING_TYPES_OVERRIDES JSON env var into a typed config dict.

    Expects a JSON object mapping training type names to override dicts.
    Use an empty dict for a type that needs all defaults::

        {
            "lora":        {"nim_image_tag": "1.13.1"},
            "all_weights": {},
            "dpo":         {"data_format": "dpo"}
        }

    Fields not specified use the dataclass defaults.

    Returns:
        Ordered dict of ``{training_type: TrainingTypeConfig}``.
    """
    data = json.loads(raw)

    configs: dict[str, TrainingTypeConfig] = {}
    for training_type, overrides in data.items():
        configs[training_type] = TrainingTypeConfig(**overrides)
    return configs


def get_unique_name(prefix: str) -> str:
    """Generate a unique name with the given prefix.

    Args:
        prefix: The prefix for the name (e.g., "e2e-customizer")

    Returns:
        A unique name like "e2e-customizer-a1b2c3d4"
    """
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def sanitize_name(prefix: str, name: str) -> str:
    """Build a platform-compatible resource name from a prefix and an arbitrary string.

    Mirrors the naming convention the platform uses for auto-deployed models
    (see LoRA customization tutorial).
    """
    name = name.split("/")[-1]
    sanitized = re.sub(r"[^a-z0-9@.+_-]", "-", name.lower())
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    return f"{prefix}-{sanitized}"[:59].rstrip("-")


def build_training_config(training_type: str, config: TrainingTypeConfig) -> dict:
    """Build training configuration for a customization job.

    Args:
        training_type: "lora", "all_weights", "dpo", or "distillation".
        config: Per-training-type settings (parallelism, etc.).

    Returns:
        Dictionary for the ``spec.training`` field.
        For distillation, the caller must set ``teacher_model`` on the
        returned dict (it requires workspace context not available here).
    """
    is_dpo = training_type == "dpo"
    is_distillation = training_type == "distillation"

    if is_distillation:
        api_type = "distillation"
    elif is_dpo:
        api_type = "dpo"
    else:
        api_type = "sft"

    training: dict = {
        "type": api_type,
        "epochs": int(os.environ.get("EPOCHS", "1")),
        "batch_size": int(os.environ.get("BATCH_SIZE", "16" if is_dpo else "64")),
        "learning_rate": float(os.environ.get("LEARNING_RATE", "5e-5" if is_dpo else "1e-5")),
        "weight_decay": float(os.environ.get("WEIGHT_DECAY", "0.01")),
        "max_seq_length": int(os.environ.get("MAX_SEQ_LENGTH", "4096")),
        "micro_batch_size": int(os.environ.get("MICRO_BATCH_SIZE", "1")),
        "val_check_interval": float(os.environ.get("VAL_CHECK_INTERVAL", "0.5")),
        "parallelism": {
            "num_gpus_per_node": config.num_gpus_per_node,
            "num_nodes": config.num_nodes,
            "tensor_parallel_size": config.tensor_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
        },
    }

    if training_type == "lora":
        training["peft"] = {
            "type": "lora",
            "rank": int(os.environ.get("LORA_RANK", "8")),
            "alpha": int(os.environ.get("LORA_ALPHA", "32")),
        }
    elif is_dpo:
        training["ref_policy_kl_penalty"] = float(os.environ.get("REF_POLICY_KL_PENALTY", "0.1"))
    elif is_distillation:
        training["distillation_ratio"] = float(os.environ.get("DISTILLATION_RATIO", "0.5"))
        training["distillation_temperature"] = float(os.environ.get("DISTILLATION_TEMPERATURE", "2.0"))
        training["teacher_precision"] = os.environ.get("TEACHER_PRECISION", "bf16")

    return training


def log_status_details(status_details: dict | None, prefix: str = "status_details") -> None:
    """Log status_details dict in JSON format."""
    if not status_details:
        logger.debug(f"No {prefix} available (empty or None)")
        return

    logger.info(f"{prefix}:")
    logger.info(json.dumps(status_details, indent=2))


def save_job_logs_to_file(jobs: JobsClient, job_name: str, workspace: str) -> tuple[Path | None, list | None]:
    """Save complete job logs to a file for CI artifact collection.

    Returns:
        Tuple of (log_file_path, log_entries). log_entries is the raw list
        from the SDK so callers can inspect without re-fetching.
    """
    LOGS_DIR.mkdir(exist_ok=True)
    log_file = LOGS_DIR / f"{job_name}.log"
    try:
        logs = list(jobs.list_job_logs(name=job_name, workspace=workspace).items())
        if logs:
            with log_file.open("w") as f:
                for log_entry in logs:
                    f.write(f"[{log_entry.job_step}] {log_entry.message}\n")
            logger.info(f"Saved {len(logs)} log entries to {log_file}")
            return log_file, logs
        else:
            logger.warning(f"No log data returned by SDK for job {job_name} — logs may not be available yet")
    except Exception as e:
        logger.warning(f"Failed to save job logs via SDK for {job_name}: {e}")

    status_file = LOGS_DIR / f"{job_name}-status.json"
    try:
        job_status = jobs.get_job_status(name=job_name, workspace=workspace).data()
        status_file.write_text(job_status.model_dump_json(indent=2))
        logger.info(f"Saved job status to {status_file}")
    except Exception as e:
        logger.warning(f"Failed to save job status for {job_name}: {e}")

    return None, None


def get_job_failure_details(sdk: NeMoHelix, job_name: str, workspace: str) -> str:
    """Get detailed failure information for a failed job.

    Fetches job logs and status to help diagnose failures.
    Also saves complete logs to a file for CI artifact collection.
    """
    details = [f"Job {job_name} failed. Details:"]
    jobs = client_from_platform(sdk, JobsClient)

    try:
        job_status = jobs.get_job_status(name=job_name, workspace=workspace).data()
        details.append(f"\nJob Status: {job_status.model_dump_json(indent=2)}")
    except Exception as e:
        details.append(f"\nFailed to get job status: {e}")

    log_file, log_entries = save_job_logs_to_file(jobs, job_name, workspace)
    if log_file:
        details.append(f"\nFull logs saved to: {log_file}")

    if log_entries:
        details.append("\nJob Logs (last 20 entries):")
        for log_entry in log_entries[-20:]:
            details.append(f"  [{log_entry.job_step}] {log_entry.message}")
    else:
        details.append("\nNo logs available")

    return "\n".join(details)


def _log_training_progress(status) -> None:
    """Extract and log training progress from the job status steps structure."""
    for job_step in status.steps or []:
        if job_step.name == "training":
            for task in job_step.tasks or []:
                task_details = task.status_details or {}
                step = task_details.get("step")
                max_steps = task_details.get("max_steps")
                training_phase = task_details.get("phase")
                if step is not None and max_steps is not None and int(str(max_steps)) > 0:
                    progress_pct = (int(str(step)) / int(str(max_steps))) * 100
                    phase_info = f" [{training_phase}]" if training_phase else ""
                    logger.info(f"Training progress: step {step}/{max_steps} ({progress_pct:.1f}%){phase_info}")
                elif training_phase:
                    logger.info(f"Training phase: {training_phase}")
            break


def wait_for_customization_job(
    sdk: NeMoHelix,
    job_name: str,
    workspace: str,
    timeout: float = 2700,
    poll_interval: float = 30,
    max_consecutive_errors: int = 5,
):
    """Wait for a customization job to reach a terminal state.

    Uses ``client_from_platform(sdk, JobsClient).get_job_status()`` to poll, logs training
    progress from the steps structure, and returns the full job object
    via ``sdk.customization.jobs.retrieve()`` once terminal.

    Args:
        sdk: NeMo Helix SDK client.
        job_name: Customization job name.
        workspace: Workspace name.
        timeout: Maximum seconds to wait before raising ``TimeoutError``.
        poll_interval: Seconds between status checks.
        max_consecutive_errors: Number of transient API failures allowed
            before giving up.

    Returns:
        The final ``CustomizationJob`` object from ``sdk.customization.jobs.retrieve()``.

    Raises:
        TimeoutError: If the job doesn't reach a terminal state within *timeout*.
    """
    start_time = time.time()
    last_status = None
    consecutive_errors = 0
    jobs = client_from_platform(sdk, JobsClient)

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(
                f"Customization job {job_name} timed out after {elapsed:.0f}s "
                f"(limit: {timeout}s), last status: {last_status}"
            )

        try:
            status = jobs.get_job_status(name=job_name, workspace=workspace).data()
            consecutive_errors = 0
        except NemoTransportError as exc:
            consecutive_errors += 1
            logger.warning(
                f"Transient error polling customization job (attempt {consecutive_errors}/{max_consecutive_errors}, "
                f"elapsed {elapsed:.0f}s): {exc}"
            )
            if consecutive_errors >= max_consecutive_errors:
                raise TimeoutError(
                    f"API unreachable after {consecutive_errors} consecutive errors "
                    f"(elapsed: {elapsed:.0f}s). Last error: {exc}"
                ) from exc
            time.sleep(poll_interval)
            continue

        if status.status != last_status:
            logger.info(f"Job status: {status.status} (elapsed: {elapsed:.0f}s)")
            log_status_details(status.status_details)
            last_status = status.status

        _log_training_progress(status)

        if status.status in TERMINAL_STATUSES:
            break

        time.sleep(poll_interval)

    return jobs.get_job(name=job_name, workspace=workspace)


def wait_for_model_spec(
    sdk: NeMoHelix,
    workspace: str,
    model_entity_name: str,
    timeout: int = 600,
    poll_interval: int = 10,
) -> None:
    """Wait for the background model-spec-analysis job to populate me.spec.

    The platform launches a background job (using the customizer-tasks image)
    when a model entity is created with a fileset.  Until that job completes,
    me.spec remains None, which causes the customizer training compiler to
    skip critical flags like force_hf and v4_compatible.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        me = client_from_platform(sdk, ModelsClient).get_model(name=model_entity_name, workspace=workspace).data()
        if me.spec is not None:
            logger.info(
                f"✓ Model spec populated for {model_entity_name}: checkpoint_model_name={me.spec.checkpoint_model_name}"
            )
            return
        logger.info(f"Waiting for model spec on {model_entity_name} ({int(deadline - time.time())}s remaining)...")
        time.sleep(poll_interval)

    pytest.fail(
        f"Model spec not populated for {model_entity_name} after {timeout}s. "
        "The model-spec-analysis background job may have failed."
    )


_SFT_TEST_PROMPT = (
    "Based on the following context, answer the question.\n\n"
    "Context: The Apollo 11 mission was the first manned mission to land on the Moon. "
    "It was launched on July 16, 1969, and Neil Armstrong became the first person to "
    "walk on the lunar surface on July 20, 1969.\n\n"
    "Question: Who was the first person to walk on the Moon?"
)

_DPO_TEST_PROMPT = (
    "Imagine you were developing software for a grocery store. "
    "Explain how you would build a database to store that information."
)

_CHAT_TEST_MESSAGES: list[dict[str, str]] = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant that answers questions based on the provided context. "
            "Give concise, accurate answers extracted from the context."
        ),
    },
    {
        "role": "user",
        "content": (
            "Context: The Apollo 11 mission was the first manned mission to land on the Moon. "
            "It was launched on July 16, 1969, and Neil Armstrong became the first person to "
            "walk on the lunar surface on July 20, 1969.\n\n"
            "Question: Who was the first person to walk on the Moon?"
        ),
    },
]


def _build_inference_messages(training_type: str, data_format: str) -> list[dict[str, str]]:
    """Pick the right messages payload for a post-training inference check."""
    if data_format == "chat_format":
        return list(_CHAT_TEST_MESSAGES)
    if training_type == "dpo":
        return [{"role": "user", "content": _DPO_TEST_PROMPT}]
    return [{"role": "user", "content": _SFT_TEST_PROMPT}]


def run_inference_test(
    sdk: NeMoHelix,
    workspace: str,
    deployment_name: str,
    model_identifier: str,
    training_type: str = "sft",
    data_format: str = "prompt_completion",
) -> None:
    """Send a test prompt via the provider gateway and validate the response.

    Args:
        sdk: NeMo Helix SDK client.
        workspace: Workspace name.
        deployment_name: Provider deployment name to route the request.
        model_identifier: Model string passed in the ``model`` field of the
            chat-completions request (e.g. ``"ws/model"`` or a LoRA adapter name).
        training_type: Training type used (e.g. "sft", "dpo", "distillation").
        data_format: Dataset format used for training (e.g. "prompt_completion", "chat_format").
    """
    inference_response = sdk.inference.gateway.provider.post(
        "v1/chat/completions",
        name=deployment_name,
        workspace=workspace,
        body={
            "model": model_identifier,
            "messages": _build_inference_messages(training_type, data_format),
            "max_tokens": 250,
        },
    )

    logger.info("✓ Inference successful")

    assert isinstance(inference_response, dict), f"Expected dict response, got {type(inference_response)}"
    response_dict = cast(dict[str, Any], inference_response)
    content = response_dict["choices"][0]["message"]["content"]
    assert isinstance(content, str) and len(content) > 0, "Message content should be a non-empty string"

    logger.info(f"Generated response ({len(content)} chars):")
    logger.info("-" * 80)
    logger.info(content)
    logger.info("-" * 80)


def deploy_and_test_model(
    sdk: NeMoHelix,
    workspace: str,
    model_name: str,
    nim_image_name: str | None = None,
    nim_image_tag: str | None = None,
    training_type: str = "sft",
    data_format: str = "prompt_completion",
) -> None:
    """Deploy a customized all-weights model and test inference.

    LoRA models are enabled by default, and available to NIM deployments with lora_enabled config.

    Note:
        NGC_API_KEY is passed to the models service via environment (see docker.py),
        which then passes it to NIM containers for model downloads.
    """
    deployment_config_name = get_unique_name("e2e-deploy-config")
    deployment_name = get_unique_name("e2e-deploy")

    logger.info("=" * 80)
    logger.info(f"Deploying customized model: {model_name}")
    logger.info("=" * 80)

    try:
        logger.info(f"Creating deployment config: {deployment_config_name}")
        model_spec_params: ModelDeploymentConfigModelSpecParam = {
            "model_name": model_name,
            "model_namespace": workspace,
        }
        executor_config_params: ContainerExecutorConfigParam = {
            "gpu": 1,
            "additional_envs": {
                "NIM_MODEL_PROFILE": "vllm",
            },
        }
        if nim_image_name:
            executor_config_params["image_name"] = nim_image_name
        if nim_image_tag:
            executor_config_params["image_tag"] = nim_image_tag

        deployment_config = sdk.inference.deployment_configs.create(
            workspace=workspace,
            name=deployment_config_name,
            description=f"E2E test deployment config for {model_name}",
            model_entity_id=model_name,
            engine="nim",
            model_spec=model_spec_params,
            executor_config=executor_config_params,
        )
        logger.info(f"✓ Deployment config created: {deployment_config.name}")

        logger.info(f"Creating deployment: {deployment_name}")
        deployment = sdk.inference.deployments.create(
            workspace=workspace,
            name=deployment_name,
            config=deployment_config_name,
        )
        logger.info(f"✓ Deployment created: {deployment.name}")
        logger.info(f"Initial deployment status: {deployment.status}")

        _wait_for_deployment_ready(sdk, workspace, deployment_name)
        _wait_for_gateway_ready(sdk, workspace, deployment_name)

        logger.info("Testing inference on deployed model...")
        run_inference_test(
            sdk,
            workspace,
            deployment_name,
            f"{workspace}/{model_name}",
            training_type=training_type,
            data_format=data_format,
        )
        logger.info("✓ Model deployment and inference test passed")

    finally:
        _cleanup_deployment(sdk, workspace, deployment_name)

        logger.info(f"Cleaning up deployment config: {deployment_config_name}")
        try:
            sdk.inference.deployment_configs.delete(deployment_config_name, workspace=workspace)
            logger.info(f"✓ Deployment config deleted: {deployment_config_name}")
        except Exception as e:
            logger.warning(f"Failed to delete deployment config {deployment_config_name}: {e}")


def wait_and_test_auto_deployment(
    sdk: NeMoHelix,
    workspace: str,
    deployment_name: str,
    output_model_name: str,
) -> None:
    """Wait for an auto-deployed LoRA model to become ready and test inference.

    When ``deployment_config`` is set in the customization job spec, the platform
    creates a deployment automatically. This function waits for that deployment
    to reach a ready state and runs a basic inference check.
    """
    logger.info("=" * 80)
    logger.info(f"Testing auto-deployed LoRA model: {output_model_name}")
    logger.info(f"Deployment name: {deployment_name}")
    logger.info("=" * 80)

    _wait_for_deployment_ready(sdk, workspace, deployment_name, ready_statuses=("READY", "RUNNING"))
    _wait_for_gateway_ready(sdk, workspace, deployment_name)

    logger.info("Testing inference on auto-deployed LoRA model...")
    run_inference_test(sdk, workspace, deployment_name, output_model_name)
    logger.info("✓ Auto-deployed LoRA model inference test passed")


def _wait_for_gateway_ready(
    sdk: NeMoHelix,
    workspace: str,
    deployment_name: str,
    timeout: int = 60,
) -> None:
    """Wait for the inference gateway to be able to route to the deployment's provider.

    The gateway refreshes its provider cache periodically. After a deployment
    reaches READY, there is a short window where the gateway may not yet know
    about the provider. This function polls the gateway's ``/ready`` endpoint
    until it confirms the provider is routable.
    """
    logger.info("Waiting for inference gateway to sync...")
    if not sdk.models.wait_for_gateway(deployment_name, workspace=workspace, timeout=timeout):
        pytest.fail(
            f"Inference gateway did not become ready for deployment '{deployment_name}' "
            f"within {timeout}s. The deployment's model provider may not have been created. "
            "Check deployment status and controller logs."
        )
    logger.info("✓ Inference gateway is ready")


def _wait_for_deployment_ready(
    sdk: NeMoHelix,
    workspace: str,
    deployment_name: str,
    timeout: int = 3600,
    poll_interval: int = 30,
    max_consecutive_errors: int = 5,
    ready_statuses: tuple[str, ...] = ("READY",),
) -> None:
    """Poll a deployment until it reaches a ready state or fails."""
    logger.info("Waiting for deployment to be ready...")
    start_time = time.time()
    consecutive_errors = 0

    while True:
        try:
            deployment_status = sdk.inference.deployments.retrieve(deployment_name, workspace=workspace)
            consecutive_errors = 0
        except (httpx.TimeoutException, httpx.ConnectError, ConnectionError, OSError) as exc:
            consecutive_errors += 1
            elapsed = time.time() - start_time
            logger.warning(
                f"Transient error polling deployment (attempt {consecutive_errors}/{max_consecutive_errors}, "
                f"elapsed {elapsed:.0f}s): {exc}"
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
            logger.error(f"Deployment entered terminal failure state: {deployment_status.status}")
            logger.error(f"Full deployment details: {deployment_status.model_dump_json(indent=2)}")
            pytest.fail(f"Deployment failed with status: {deployment_status.status}")
        if time.time() - start_time > timeout:
            logger.error(f"Deployment timed out with status: {deployment_status.status}")
            logger.error(f"Full deployment details: {deployment_status.model_dump_json(indent=2)}")
            pytest.fail("Deployment did not become ready within timeout")

        logger.info(f"Deployment status: {deployment_status.status}, waiting...")
        time.sleep(poll_interval)

    logger.info("✓ Deployment is ready")


def _cleanup_deployment(sdk: NeMoHelix, workspace: str, deployment_name: str) -> None:
    """Delete a deployment unless it is in an error state (left for log collection)."""
    try:
        dep_status = sdk.inference.deployments.retrieve(deployment_name, workspace=workspace)
        if dep_status.status in ("ERROR", "LOST"):
            logger.warning(
                f"Skipping cleanup of deployment {deployment_name} "
                f"(status={dep_status.status}) — leaving pod for log collection"
            )
        else:
            logger.info(f"Cleaning up deployment: {deployment_name}")
            sdk.inference.deployments.delete(deployment_name, workspace=workspace)
            logger.info(f"✓ Deployment deleted: {deployment_name}")
            logger.info("Waiting 60s for deployment deletion")
            time.sleep(60)
    except Exception as e:
        logger.warning(f"Failed to clean up deployment {deployment_name}: {e}")
