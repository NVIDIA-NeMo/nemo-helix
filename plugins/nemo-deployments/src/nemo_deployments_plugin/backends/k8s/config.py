# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executor-level Kubernetes backend configuration."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from typing import Any

from nemo_deployments_plugin.backends.k8s.compiler import ExecutorK8sDefaults
from nemo_deployments_plugin.constants import (
    DEFAULT_JOB_TTL_SECONDS_AFTER_FINISHED,
    MIN_JOB_TTL_SECONDS_AFTER_FINISHED,
)
from nemo_helix_plugin.config import ImagePullSecret
from pydantic import BaseModel, Field, field_validator

_DNS_LABEL_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")

# Downward-API env var carrying the controller pod's own namespace. The core
# controller deployment (where the deployments backend runs) injects this via
# ``fieldRef: metadata.namespace`` (see k8s/helm/templates/core/controller-deployment.yaml).
_POD_NAMESPACE_ENV = "POD_NAMESPACE"

# Last-resort namespace when neither an explicit config value nor the downward
# API is available (e.g. local/out-of-cluster runs).
_FALLBACK_NAMESPACE = "default"


class K8sExecutorConfig(BaseModel):
    """Knobs for a named k8s executor instance (not entity backend_config)."""

    kubeconfig_path: str | None = Field(
        default=None,
        description="Path to kubeconfig file. When unset, uses in-cluster config or default kubeconfig.",
    )
    default_namespace: str | None = Field(
        default=None,
        min_length=1,
        max_length=63,
        description=(
            "Namespace for resources when entity backend_config.k8s.namespace is unset. When this is "
            "itself unset, the executor defaults to its own pod namespace (POD_NAMESPACE, injected via "
            "the downward API), so a deployed platform places agents beside itself; if that is also "
            "unavailable it falls back to 'default'. See effective_namespace."
        ),
    )
    request_timeout: int = Field(
        default=60,
        ge=1,
        description="Kubernetes API client timeout in seconds.",
    )
    image_pull_secrets: list[ImagePullSecret] = Field(
        default_factory=list,
        description="Image pull secrets merged with platform image_pull_secrets on every pod.",
    )
    default_pod_annotations: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Executor-level default pod annotations stamped onto every Job/Deployment pod this "
            "executor renders (all consumers: models, agents, ...). Merged key-wise with per-entity "
            "backend_config.k8s.podAnnotations, where the per-entity value wins for a shared key. "
            "Ships the Istio native-sidecar annotation so a mesh-injected proxy terminates when a "
            "Job's main container exits."
        ),
    )
    default_node_selector: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Executor-level default nodeSelector applied to every Job/Deployment pod this executor "
            "renders when the entity does not set backend_config.k8s.nodeSelector."
        ),
    )
    default_tolerations: list[dict[str, str | int]] = Field(
        default_factory=list,
        description=(
            "Executor-level default pod tolerations applied to every Job/Deployment pod this executor "
            "renders when the entity does not set backend_config.k8s.tolerations. Each entry is a raw "
            "Kubernetes toleration object."
        ),
    )
    default_affinity: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Executor-level default pod affinity applied to every Job/Deployment pod this executor "
            "renders when the entity does not set backend_config.k8s.affinity. Raw Kubernetes affinity "
            "object (nodeAffinity / podAffinity / podAntiAffinity)."
        ),
    )
    default_topology_spread_constraints: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Executor-level default pod topology spread constraints applied to every Job/Deployment pod "
            "this executor renders when the entity does not set backend_config.k8s.topologySpreadConstraints. "
            "Each entry is a raw Kubernetes topologySpreadConstraint object."
        ),
    )
    default_job_ttl_seconds_after_finished: int | None = Field(
        default=DEFAULT_JOB_TTL_SECONDS_AFTER_FINISHED,
        description=(
            "Executor-level ttlSecondsAfterFinished applied to every finite (Never/OnFailure) Job this "
            "executor renders, such as the weight-puller. A completed puller pod holds its ReadWriteOnce "
            "weights volume attachment until reaped; a short TTL releases it promptly so the serving "
            "Deployment can mount the volume instead of failing with MultiAttachError. Set to null to omit "
            "the field and defer to the cluster default (effectively never reaped). A set value must be at "
            f"least {MIN_JOB_TTL_SECONDS_AFTER_FINISHED}s so the reconciler can observe Job completion before "
            "the pod is reaped. Overridden per-entity by backend_config.k8s.jobTtlSecondsAfterFinished."
        ),
    )

    @field_validator("default_job_ttl_seconds_after_finished")
    @classmethod
    def _validate_job_ttl(cls, value: int | None) -> int | None:
        if value is not None and value < MIN_JOB_TTL_SECONDS_AFTER_FINISHED:
            raise ValueError(
                f"default_job_ttl_seconds_after_finished must be >= {MIN_JOB_TTL_SECONDS_AFTER_FINISHED} "
                "(or null to defer to the cluster default); a smaller value races the reconciler's "
                "completion read and can mis-record the Job as FAILED"
            )
        return value

    @field_validator("default_namespace")
    @classmethod
    def _validate_default_namespace(cls, value: str | None) -> str | None:
        if value is not None and not _DNS_LABEL_PATTERN.fullmatch(value):
            raise ValueError("default_namespace must be a lowercase DNS-1123 label (alphanumeric, interior hyphens)")
        return value

    @property
    def effective_namespace(self) -> str:
        """Resolve the namespace to deploy into.

        Precedence: an explicit ``default_namespace`` config value wins; else the
        controller's own pod namespace (``POD_NAMESPACE`` downward API); else
        ``"default"``. This lets a deployed platform omit the setting and still
        place agents in its own namespace, while an operator can always pin one.
        """
        if self.default_namespace is not None:
            return self.default_namespace
        pod_namespace = os.environ.get(_POD_NAMESPACE_ENV, "").strip()
        if pod_namespace:
            return pod_namespace
        return _FALLBACK_NAMESPACE

    def to_k8s_defaults(self) -> ExecutorK8sDefaults:
        """Bundle the executor-level pod defaults for the workload compiler.

        Shared by every deployments-plugin consumer (models, agents, ...): the
        compiler applies these as the base layer, with per-entity
        ``backend_config.k8s`` values overriding them.
        """
        return ExecutorK8sDefaults(
            pod_annotations=dict(self.default_pod_annotations),
            node_selector=dict(self.default_node_selector),
            tolerations=deepcopy(self.default_tolerations),
            affinity=deepcopy(self.default_affinity),
            topology_spread_constraints=deepcopy(self.default_topology_spread_constraints),
            job_ttl_seconds_after_finished=self.default_job_ttl_seconds_after_finished,
        )
