# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Dict
from typing_extensions import Required, TypedDict

from ..._types import SequenceNotStr
from .k8s_nim_operator_config_param import K8sNIMOperatorConfigParam

__all__ = ["ContainerExecutorConfigParam"]


class ContainerExecutorConfigParam(TypedDict, total=False):
    """Compute + container settings shared by the docker and k8s executors.

    Both the docker and k8s executors run containers and share this shape.
    A future non-container executor (e.g. subprocess) would warrant turning
    ``executor_config`` into a discriminated union.
    """

    gpu: Required[int]
    """Number of GPUs required for the deployment. 0 = CPU-only."""

    additional_args: SequenceNotStr[str]
    """Raw container/`serve` args appended verbatim to the container's arg vector."""

    additional_envs: Dict[str, str]
    """Additional environment variables for the deployment"""

    disk_size: str
    """Disk size for the deployment"""

    health_check_path: str
    """HTTP path used for the container readiness probe.

    If not specified, defaults to the engine's standard health endpoint (e.g.
    '/v1/health/ready' for NIM, '/health' for vLLM). Set this for engine='generic'
    containers that expose a non-standard health endpoint.
    """

    image_name: str
    """Container image name.

    If not specified, defaults to the engine's configured image (e.g.
    default_vllm_image / default_nimservice_image). Required for engine='generic'.
    """

    image_tag: str
    """Container image tag.

    If not specified, defaults to the engine's configured image tag.
    """

    k8s_nim_operator_config: K8sNIMOperatorConfigParam
    """Kubernetes configuration for NIM deployment via k8s-nim-operator.

    These fields provide typed access to commonly-used NIMService Spec fields and
    are applied before override_config in the compilation precedence.
    """

    override_config: Dict[str, object]
    """
    Partial NIMService Spec fragments deep-merged after generated defaults and
    k8s_nim_operator_config (NIM engine on k8s only). Supported keys: image,
    command, args, resources, env, readinessProbe, livenessProbe, startupProbe,
    nodeSelector, tolerations, userID, groupID, labels, initContainers,
    sidecarContainers. Unsupported keys are rejected at compile time. Ignored by
    non-NIM engines and docker runtime.
    """

    run_as_group: int
    """
    Pod securityContext runAsGroup (gid) for the serving container (k8s backend
    only). If unset, the engine default applies. Ignored by the docker backend.
    """

    run_as_user: int
    """Pod securityContext runAsUser (uid) for the serving container (k8s backend
    only).

    If unset, the engine default applies (vLLM pins its image's user; generic uses
    the image's own user). Ignored by the docker backend.
    """
