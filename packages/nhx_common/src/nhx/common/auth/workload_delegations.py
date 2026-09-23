# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility re-exports for workload delegation helpers.

The canonical implementation lives in ``nemo_helix_plugin.auth`` so jobs,
deployments, and plugins share the same workload OBO token exchange primitives.
"""

from nemo_helix_plugin.auth.workload_delegations import (
    DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE as DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE,
)
from nemo_helix_plugin.auth.workload_delegations import (
    JWT_WORKLOAD_SUBJECT_TOKEN_TYPE as JWT_WORKLOAD_SUBJECT_TOKEN_TYPE,
)
from nemo_helix_plugin.auth.workload_delegations import (
    KUBERNETES_POD_UID_REFERENCE_NAME as KUBERNETES_POD_UID_REFERENCE_NAME,
)
from nemo_helix_plugin.auth.workload_delegations import (
    OPAQUE_DOCKER_PROOF_PREFIX as OPAQUE_DOCKER_PROOF_PREFIX,
)
from nemo_helix_plugin.auth.workload_delegations import (
    SYSTEM_WORKSPACE as SYSTEM_WORKSPACE,
)
from nemo_helix_plugin.auth.workload_delegations import (
    WORKLOAD_DELEGATION_ENTITY_TYPE as WORKLOAD_DELEGATION_ENTITY_TYPE,
)
from nemo_helix_plugin.auth.workload_delegations import (
    InvalidWorkloadProofTokenError as InvalidWorkloadProofTokenError,
)
from nemo_helix_plugin.auth.workload_delegations import (
    ParsedOpaqueDockerProofToken as ParsedOpaqueDockerProofToken,
)
from nemo_helix_plugin.auth.workload_delegations import (
    SyncWorkloadDelegationStore as SyncWorkloadDelegationStore,
)
from nemo_helix_plugin.auth.workload_delegations import (
    WorkloadDelegationConflictError as WorkloadDelegationConflictError,
)
from nemo_helix_plugin.auth.workload_delegations import (
    WorkloadDelegationEntity as WorkloadDelegationEntity,
)
from nemo_helix_plugin.auth.workload_delegations import (
    WorkloadDelegationError as WorkloadDelegationError,
)
from nemo_helix_plugin.auth.workload_delegations import (
    WorkloadDelegationLookupScope as WorkloadDelegationLookupScope,
)
from nemo_helix_plugin.auth.workload_delegations import (
    WorkloadDelegationScope as WorkloadDelegationScope,
)
from nemo_helix_plugin.auth.workload_delegations import (
    WorkloadDelegationStore as WorkloadDelegationStore,
)
from nemo_helix_plugin.auth.workload_delegations import (
    WorkloadDelegationValidationError as WorkloadDelegationValidationError,
)
from nemo_helix_plugin.auth.workload_delegations import (
    as_aware_utc as as_aware_utc,
)
from nemo_helix_plugin.auth.workload_delegations import (
    create_opaque_docker_proof_token as create_opaque_docker_proof_token,
)
from nemo_helix_plugin.auth.workload_delegations import (
    docker_delegation_name as docker_delegation_name,
)
from nemo_helix_plugin.auth.workload_delegations import (
    docker_deployment_delegation_name as docker_deployment_delegation_name,
)
from nemo_helix_plugin.auth.workload_delegations import (
    docker_workload_delegation_name as docker_workload_delegation_name,
)
from nemo_helix_plugin.auth.workload_delegations import (
    kubernetes_pod_uid_delegation_name as kubernetes_pod_uid_delegation_name,
)
from nemo_helix_plugin.auth.workload_delegations import (
    parse_opaque_docker_proof_token as parse_opaque_docker_proof_token,
)
from nemo_helix_plugin.auth.workload_delegations import (
    reference_delegation_name as reference_delegation_name,
)
from nemo_helix_plugin.auth.workload_delegations import (
    subject_token_type_for_exchange as subject_token_type_for_exchange,
)
from nemo_helix_plugin.auth.workload_delegations import (
    verify_opaque_docker_proof_token_hash as verify_opaque_docker_proof_token_hash,
)
