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

from typing import Dict, Iterable
from typing_extensions import TypedDict

__all__ = ["K8sNIMOperatorConfigParam"]


class K8sNIMOperatorConfigParam(TypedDict, total=False):
    """Kubernetes configuration for NIM deployment via k8s-nim-operator.

    These fields provide typed access to commonly-used NIMService Spec fields
    and are applied before override_config in the compilation precedence.
    """

    node_selector: Dict[str, str]
    """Kubernetes node selector for pod placement.

    Example: {'node-type': 'gpu-node', 'zone': 'us-west1-a'}
    """

    resources: Dict[str, object]
    """Kubernetes resource requirements including requests and limits.

    Example: {'requests': {'cpu': '2', 'memory': '8Gi'}, 'limits': {'memory':
    '16Gi'}}
    """

    startup_probe_grace_seconds: int
    """Grace period in seconds for NIM startup.

    Determines how long Kubernetes will wait for the NIM to become ready before
    restarting it. Example: 600 (10 minutes). Must be a positive integer.
    """

    tolerations: Iterable[Dict[str, object]]
    """Kubernetes tolerations for pod scheduling.

    Example: [{'key': 'nvidia.com/gpu', 'operator': 'Exists', 'effect':
    'NoSchedule'}]
    """
