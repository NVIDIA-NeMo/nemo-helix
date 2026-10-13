# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from nemo_deployments_plugin.entities import (
    Container,
    Deployment,
    DeploymentConfig,
    DriftRecoveryPolicy,
    EnvVar,
    Volume,
    WorkloadIdentitySpec,
)
from nemo_deployments_plugin.validation import PrerequisiteCycleError, detect_prerequisite_cycle
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from pydantic import ValidationError


def test_deployment_defaults_to_pending() -> None:
    dep = Deployment(name="d1", workspace="default", deployment_config="cfg")
    assert dep.status == "PENDING"
    assert dep.desired_state == "READY"


def test_deployment_config_requires_containers_shape() -> None:
    cfg = DeploymentConfig(
        name="cfg",
        workspace="default",
        containers=[Container(name="main", image="nginx:latest")],
    )
    assert cfg.containers[0].image == "nginx:latest"


def test_deployment_config_rejects_zero_backoff_limit() -> None:
    with pytest.raises(ValidationError):
        DeploymentConfig.model_validate(
            {
                "name": "cfg",
                "workspace": "default",
                "containers": [Container(name="main", image="nginx:latest")],
                "backoff_limit": 0,
            }
        )


def test_volume_default_status_pending() -> None:
    vol = Volume(name="v1", workspace="default")
    assert vol.status == "PENDING"
    assert vol.size == "1Gi"


def test_prerequisite_cycle_detected() -> None:
    with pytest.raises(PrerequisiteCycleError):
        detect_prerequisite_cycle(
            deployment_name="c",
            prerequisites=["a"],
            existing={"a": ["b"], "b": ["c"]},
        )


def test_invalid_deployment_status_rejected() -> None:
    with pytest.raises(ValidationError):
        Deployment.model_validate(
            {
                "name": "d1",
                "workspace": "default",
                "deployment_config": "cfg",
                "status": "not-a-status",
            }
        )


def test_drift_recovery_policy_rejects_negative_overrides() -> None:
    with pytest.raises(ValidationError):
        DriftRecoveryPolicy(max_attempts=-1)
    with pytest.raises(ValidationError):
        DriftRecoveryPolicy(initial_delay_seconds=-1)
    with pytest.raises(ValidationError):
        DriftRecoveryPolicy(max_delay_seconds=-1)


def test_drift_recovery_policy_rejects_inverted_delays() -> None:
    with pytest.raises(ValidationError, match="initial_delay_seconds"):
        DriftRecoveryPolicy(initial_delay_seconds=60, max_delay_seconds=5)


def test_workload_identity_spec_defaults_disabled() -> None:
    spec = WorkloadIdentitySpec()
    assert spec.enabled is False


def test_deployment_config_accepts_workload_identity_aliases() -> None:
    cfg = DeploymentConfig.model_validate(
        {
            "name": "cfg",
            "workspace": "default",
            "containers": [{"name": "main", "image": "nginx:latest"}],
            "workloadIdentity": {
                "enabled": True,
                "workloadKind": "agent_deployment",
                "workloadId": "dep1",
                "tokenAudience": "nemo-helix",
                "serviceAccountName": "dep-sa",
                "tokenExpirationSeconds": 900,
            },
        }
    )

    assert cfg.workload_identity is not None
    assert cfg.workload_identity.enabled is True
    assert cfg.workload_identity.workload_kind == "agent_deployment"
    assert cfg.workload_identity.workload_id == "dep1"
    assert cfg.workload_identity.token_audience == "nemo-helix"
    assert cfg.workload_identity.service_account_name == "dep-sa"
    assert cfg.workload_identity.token_expiration_seconds == 900


def test_workload_identity_spec_caps_token_expiration_seconds() -> None:
    with pytest.raises(ValidationError, match="tokenExpirationSeconds"):
        WorkloadIdentitySpec(enabled=True, tokenExpirationSeconds=86401)


def test_deployment_config_rejects_user_supplied_workload_token_env_when_enabled() -> None:
    with pytest.raises(ValidationError, match=WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR):
        DeploymentConfig(
            name="cfg",
            workspace="default",
            containers=[
                Container(
                    name="main",
                    image="nginx:latest",
                    env=[EnvVar(name=WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, value="/tmp/token")],
                )
            ],
            workloadIdentity=WorkloadIdentitySpec(enabled=True),
        )


def test_deployment_persists_auth_context_private_attr() -> None:
    auth_context = AuthContext(
        principal_id="user:alice",
        principal_email="alice@example.com",
        principal_groups=["research"],
    )
    dep = Deployment(name="d1", workspace="default", deployment_config="cfg").with_auth_context(auth_context)

    assert dep.auth_context == auth_context
    assert dep._get_data_fields()["_auth_context"]["principal_id"] == "user:alice"


def test_deployment_auth_context_schema_is_nullable() -> None:
    schema = Deployment.model_json_schema(mode="serialization")

    assert schema["properties"]["auth_context"]["nullable"] is True
