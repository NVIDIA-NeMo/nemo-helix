# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for model deployment backends (Docker and K8s NIM Operator)."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol

from nemo_platform_plugin.models.types import K8sNIMOperatorConfig, ModelDeployment, ToolCallConfig

LOG_TAIL_LINES = 80
LOG_MAX_CHARS = 2048


class _DeploymentConfigLike(Protocol):
    """Structural type for the objects ``deployment_config_view`` accepts.

    Both the SDK ``ModelDeploymentConfig`` and the create/update request models
    expose ``model_spec`` and ``executor_config`` groups; we read them
    defensively via ``getattr`` so partial/None groups don't raise.
    """

    model_spec: Any
    executor_config: Any


@dataclass
class DeploymentConfigView:
    """Flattened view of the fields drawn from ``model_spec`` and ``executor_config``.

    The ``ModelDeploymentConfig`` schema splits model facts (``model_spec``) from
    compute/container facts (``executor_config``). Several backend code paths need
    a single object exposing the union of both groups. This view re-presents that
    union so engine-agnostic plumbing (weight resolution, GPU/image selection) and
    the existing NIM compilers can read one flat object.

    The k8s-specific NIM fields (``k8s_nim_operator_config`` / ``override_config``)
    live on ``executor_config`` and are surfaced here for the NIM-on-operator path.
    """

    # model_spec fields
    model_type: Optional[str] = None
    model_namespace: Optional[str] = None
    model_name: Optional[str] = None
    model_revision: Optional[str] = None
    chat_template: Optional[str] = None
    tool_call_config: Optional[ToolCallConfig] = None
    lora_enabled: bool = False
    # executor_config fields
    gpu: int = 0
    disk_size: Optional[str] = None
    image_name: Optional[str] = None
    image_tag: Optional[str] = None
    health_check_path: Optional[str] = None
    run_as_user: Optional[int] = None
    run_as_group: Optional[int] = None
    additional_envs: Optional[Dict[str, str]] = None
    additional_args: Optional[List[str]] = None
    k8s_nim_operator_config: Optional[K8sNIMOperatorConfig] = None
    override_config: Optional[Dict[str, Any]] = None


def deployment_config_view(config: Optional[_DeploymentConfigLike]) -> DeploymentConfigView:
    """Build a flattened config view from an engine-split deployment config."""
    model_spec = getattr(config, "model_spec", None)
    executor = getattr(config, "executor_config", None)
    return DeploymentConfigView(
        model_type=getattr(model_spec, "model_type", None),
        model_namespace=getattr(model_spec, "model_namespace", None),
        model_name=getattr(model_spec, "model_name", None),
        model_revision=getattr(model_spec, "model_revision", None),
        chat_template=getattr(model_spec, "chat_template", None),
        tool_call_config=getattr(model_spec, "tool_call_config", None),
        lora_enabled=bool(getattr(model_spec, "lora_enabled", False)),
        gpu=getattr(executor, "gpu", 0) or 0,
        disk_size=getattr(executor, "disk_size", None),
        image_name=getattr(executor, "image_name", None),
        image_tag=getattr(executor, "image_tag", None),
        health_check_path=getattr(executor, "health_check_path", None),
        run_as_user=getattr(executor, "run_as_user", None),
        run_as_group=getattr(executor, "run_as_group", None),
        additional_envs=getattr(executor, "additional_envs", None),
        additional_args=getattr(executor, "additional_args", None),
        k8s_nim_operator_config=getattr(executor, "k8s_nim_operator_config", None),
        override_config=getattr(executor, "override_config", None),
    )


def _elapsed_seconds_since(timestamp: datetime | None) -> float:
    if timestamp is None:
        return 0.0
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - timestamp).total_seconds()


def deployment_elapsed_seconds(deployment: ModelDeployment) -> float:
    """Seconds since the deployment entity was created.

    Uses the entity-store ``created_at`` timestamp so the value survives
    controller restarts.
    """
    return _elapsed_seconds_since(deployment.created_at)


def deleting_elapsed_seconds(deployment: ModelDeployment) -> float:
    """Seconds since the deployment last changed state (for DELETING timeout)."""
    return _elapsed_seconds_since(deployment.updated_at or deployment.created_at)


def format_duration(seconds: float) -> str:
    """Human-readable duration string (e.g. '2h 5m 30s')."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)
