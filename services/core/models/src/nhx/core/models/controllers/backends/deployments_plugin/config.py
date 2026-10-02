# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the deployments-plugin models backend."""

from typing import Any

from pydantic import BaseModel, Field


class DeploymentsPluginConfig(BaseModel):
    default_executor: str | None = None
    docker_executor: str | None = None
    k8s_executor: str | None = None
    pending_timeout_seconds: int = Field(
        default=7200,
        ge=60,
        description="Maximum seconds a deployment may stay PENDING before ERROR.",
    )
    max_restart_count: int = 5
    default_storage_class: str | None = None
    default_pvc_size: str = "200Gi"
    default_pod_annotations: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Platform-default pod annotations applied to every k8s model deployment/job "
            "(all engines). Merged key-wise into the compiled K8sDeploymentConfig; a "
            "per-entity annotation for the same key wins over the platform default. "
            "Used to ship the Istio native-sidecar annotation so a mesh-injected proxy "
            "terminates when a puller Job's main container exits."
        ),
    )
    default_node_selector: dict[str, str] = Field(
        default_factory=dict,
        description="Platform-default nodeSelector applied to every k8s model deployment/job (all engines).",
    )
    default_tolerations: list[dict[str, str | int]] = Field(
        default_factory=list,
        description="Platform-default pod tolerations applied to every k8s model deployment/job (all engines).",
    )
    default_affinity: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Platform-default pod affinity applied to every k8s model deployment/job (all engines) "
            "when the deployment does not already set an affinity. Raw Kubernetes affinity object "
            "(nodeAffinity / podAffinity / podAntiAffinity)."
        ),
    )
    default_topology_spread_constraints: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Platform-default pod topology spread constraints applied to every k8s model "
            "deployment/job (all engines) when the deployment does not already set them. Each entry "
            "is a raw Kubernetes topologySpreadConstraint object."
        ),
    )
    default_nimservice_image: str = "nvcr.io/nim/meta/llama-3.1-8b-instruct"
    default_nimservice_image_tag: str = "1.8.5"
    default_vllm_image: str = Field(
        default="docker.io/vllm/vllm-openai",
        description="Default vLLM image repository. Fully qualified so it resolves on runtimes that block docker.io short names.",
    )
    default_vllm_image_tag: str = "v0.22.1"
    default_user_id: int | None = 1000
    default_group_id: int | None = 2000
    default_vllm_user_id: int | None = 2000
    default_vllm_group_id: int | None = 0
    peft_refresh_interval: int = 30
    lora_sidecar_image_name: str = "nhx-api"
    lora_sidecar_command: list[str] = Field(
        default_factory=lambda: ["python", "-m", "nhx.core.models.sidecars.adapters.main"],
        description=(
            "Container command for the LoRA adapters sidecar. Replaces the image ENTRYPOINT "
            "(e.g. `nemo`), so the adapters module is invoked directly and does not write "
            "platform instance state under the image $HOME (which is unwritable under the "
            "pod's runtime uid)."
        ),
    )
    lora_sidecar_args: list[str] = Field(default_factory=list)
    busybox_image: str = Field(
        default="docker.io/library/busybox",
        description="BusyBox image repository for LoRA cache init containers. "
        "Fully qualified so it resolves on runtimes that block docker.io short names.",
    )
    busybox_image_tag: str = Field(
        default="latest",
        description="BusyBox image tag for LoRA cache init containers.",
    )
    deleting_timeout_seconds: int = Field(
        default=60,
        ge=0,
        description=("Maximum seconds a deployment may stay DELETING before ERROR. 0 disables timeout escalation."),
    )


class DeploymentsPluginBackendConfigModel(DeploymentsPluginConfig):
    """Flat registry configuration, including the enablement switch."""

    enabled: bool = False
