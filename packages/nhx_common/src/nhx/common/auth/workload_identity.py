# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility re-exports for workload identity helpers.

The canonical implementation lives in ``nemo_helix_plugin.auth`` so jobs,
deployments, and plugins share the same workload OBO token exchange helpers.
"""

from nemo_helix_plugin.auth.workload_identity import DEFAULT_WORKLOAD_AUDIENCE as DEFAULT_WORKLOAD_AUDIENCE
from nemo_helix_plugin.auth.workload_identity import (
    WORKLOAD_DELEGATION_TTL_BUFFER_SECONDS as WORKLOAD_DELEGATION_TTL_BUFFER_SECONDS,
)
from nemo_helix_plugin.auth.workload_identity import (
    WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR as WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR,
)
from nemo_helix_plugin.auth.workload_identity import (
    WORKLOAD_IDENTITY_TOKEN_FILE_PATH as WORKLOAD_IDENTITY_TOKEN_FILE_PATH,
)
from nemo_helix_plugin.auth.workload_identity import (
    WORKLOAD_IDENTITY_VOLUME_NAME as WORKLOAD_IDENTITY_VOLUME_NAME,
)
from nemo_helix_plugin.auth.workload_identity import (
    WORKLOAD_IDENTITY_VOLUME_PATH as WORKLOAD_IDENTITY_VOLUME_PATH,
)
from nemo_helix_plugin.auth.workload_identity import WorkloadIdentityConfigError as WorkloadIdentityConfigError
from nemo_helix_plugin.auth.workload_identity import (
    build_docker_opaque_workload_delegation as build_docker_opaque_workload_delegation,
)
from nemo_helix_plugin.auth.workload_identity import (
    build_kubernetes_pod_uid_workload_delegation as build_kubernetes_pod_uid_workload_delegation,
)
from nemo_helix_plugin.auth.workload_identity import build_token_archive as build_token_archive
from nemo_helix_plugin.auth.workload_identity import (
    get_workload_delegation_audience as get_workload_delegation_audience,
)
from nemo_helix_plugin.auth.workload_identity import (
    get_workload_identity_token_audience as get_workload_identity_token_audience,
)
from nemo_helix_plugin.auth.workload_identity import (
    is_workload_identity_token_exchange_enabled as is_workload_identity_token_exchange_enabled,
)
from nemo_helix_plugin.auth.workload_identity import (
    kubernetes_service_account_subject as kubernetes_service_account_subject,
)
from nemo_helix_plugin.auth.workload_identity import workload_delegation_expires_at as workload_delegation_expires_at
from nemo_helix_plugin.auth.workload_identity import workload_identity_env as workload_identity_env
