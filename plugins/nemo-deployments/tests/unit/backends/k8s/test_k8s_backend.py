# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_deployments_plugin.backends.base import BackendStatusUpdate
from nemo_deployments_plugin.backends.k8s.backend import K8sDeploymentBackend
from nemo_deployments_plugin.backends.k8s.config import K8sExecutorConfig
from nemo_deployments_plugin.entities import Container, DeploymentConfig
from nemo_platform_plugin.entity_client import NemoEntityNotFoundError


def test_executor_config_parsed_from_dict(k8s_backend: K8sDeploymentBackend) -> None:
    assert k8s_backend.executor_config.default_namespace == "nemo-deployments"
    assert k8s_backend.executor_config.request_timeout == 30


def test_shutdown_closes_kubernetes_clients(k8s_backend: K8sDeploymentBackend) -> None:
    mock_clients = MagicMock()
    k8s_backend._clients = mock_clients
    k8s_backend.shutdown()
    mock_clients.close.assert_called_once()


def test_default_namespace_rejects_invalid_dns_label() -> None:
    with pytest.raises(ValueError, match="default_namespace must be a lowercase DNS-1123 label"):
        K8sExecutorConfig(default_namespace="X")


def test_to_k8s_defaults_bundles_executor_pod_defaults() -> None:
    config = K8sExecutorConfig(
        default_pod_annotations={"sidecar.istio.io/nativeSidecar": "true"},
        default_node_selector={"gpu": "a100"},
        default_tolerations=[{"key": "gpu", "operator": "Equal", "value": "true", "effect": "NoSchedule"}],
        default_affinity={"nodeAffinity": {}},
        default_topology_spread_constraints=[{"maxSkew": 1, "topologyKey": "kubernetes.io/hostname"}],
    )
    defaults = config.to_k8s_defaults()
    assert defaults.pod_annotations == {"sidecar.istio.io/nativeSidecar": "true"}
    assert defaults.node_selector == {"gpu": "a100"}
    assert defaults.tolerations[0]["key"] == "gpu"
    assert defaults.affinity == {"nodeAffinity": {}}
    assert defaults.topology_spread_constraints[0]["topologyKey"] == "kubernetes.io/hostname"


def test_to_k8s_defaults_empty_by_default() -> None:
    defaults = K8sExecutorConfig().to_k8s_defaults()
    assert defaults.pod_annotations == {}
    assert defaults.node_selector == {}
    assert defaults.tolerations == []
    assert defaults.affinity == {}
    assert defaults.topology_spread_constraints == []


def test_to_k8s_defaults_deep_copies_nested_objects() -> None:
    # Mutating the returned defaults must not corrupt the shared executor config
    # for later workloads (nested nodeAffinity / labelSelector are deep-copied).
    config = K8sExecutorConfig(
        default_affinity={"nodeAffinity": {"key": "orig"}},
        default_topology_spread_constraints=[{"labelSelector": {"matchLabels": {"app": "orig"}}}],
    )
    defaults = config.to_k8s_defaults()
    defaults.affinity["nodeAffinity"]["key"] = "mutated"
    defaults.topology_spread_constraints[0]["labelSelector"]["matchLabels"]["app"] = "mutated"
    assert config.default_affinity["nodeAffinity"]["key"] == "orig"
    assert config.default_topology_spread_constraints[0]["labelSelector"]["matchLabels"]["app"] == "orig"


def test_effective_namespace_prefers_explicit_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # An explicit config value wins even when POD_NAMESPACE is set.
    monkeypatch.setenv("POD_NAMESPACE", "pod-ns")
    config = K8sExecutorConfig(default_namespace="explicit-ns")
    assert config.effective_namespace == "explicit-ns"


def test_effective_namespace_falls_back_to_pod_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    # No explicit config -> the controller's own pod namespace (downward API).
    monkeypatch.setenv("POD_NAMESPACE", "pod-ns")
    config = K8sExecutorConfig()
    assert config.default_namespace is None
    assert config.effective_namespace == "pod-ns"


def test_effective_namespace_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Neither an explicit config value nor POD_NAMESPACE -> "default".
    monkeypatch.delenv("POD_NAMESPACE", raising=False)
    config = K8sExecutorConfig()
    assert config.effective_namespace == "default"


def test_effective_namespace_ignores_blank_pod_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    # A blank/whitespace POD_NAMESPACE is treated as unset.
    monkeypatch.setenv("POD_NAMESPACE", "  ")
    config = K8sExecutorConfig()
    assert config.effective_namespace == "default"


@pytest.mark.asyncio
async def test_create_passes_secret_env_and_unmodified_config(
    k8s_backend: K8sDeploymentBackend,
    mock_entities: AsyncMock,
    mock_sdk: MagicMock,
) -> None:
    # k8s keeps the stored config's secret_ref env vars intact and passes the
    # resolved secret values separately as secret_env (mounted via a managed
    # Secret + envFrom), so plaintext never lands in the pod manifest.
    stored = DeploymentConfig(
        name="cfg",
        workspace="default",
        containers=[Container(name="server", image="example:latest")],
    )
    secret_env = {"APP_TOKEN": "app-token-value"}
    mock_entities.get.return_value = stored

    with (
        patch(
            "nemo_deployments_plugin.backends.k8s.backend.resolve_deployment_secret_env",
            AsyncMock(return_value=secret_env),
        ) as resolve_mock,
        patch.object(
            k8s_backend._deployment_ops,
            "create_deployment",
            AsyncMock(return_value=BackendStatusUpdate(status="STARTING")),
        ) as create_mock,
    ):
        await k8s_backend.create_deployment(
            workspace="default",
            name="server",
            config_name="cfg",
            labels={},
            backend_config={},
        )

    resolve_mock.assert_awaited_once_with(mock_sdk, stored)
    assert create_mock.await_args is not None
    assert create_mock.await_args.kwargs["config"] is stored
    assert create_mock.await_args.kwargs["secret_env"] == secret_env


@pytest.mark.asyncio
async def test_delete_deployment_revokes_delegations_when_context_missing(
    k8s_backend: K8sDeploymentBackend,
    mock_entities: AsyncMock,
) -> None:
    mock_entities.get.side_effect = NemoEntityNotFoundError("missing")

    with (
        patch.object(
            k8s_backend._deployment_ops,
            "delete_deployment",
            AsyncMock(return_value=BackendStatusUpdate(status="SUCCEEDED")),
        ) as delete_deployment_mock,
        patch.object(
            k8s_backend._job_ops,
            "delete_job",
            AsyncMock(return_value=BackendStatusUpdate(status="SUCCEEDED")),
        ) as delete_job_mock,
        patch.object(
            k8s_backend._workload_identity_ops,
            "revoke_workload_delegations",
            AsyncMock(),
        ) as revoke_mock,
    ):
        update = await k8s_backend.delete_deployment("default", "server")

    assert update.status == "SUCCEEDED"
    delete_deployment_mock.assert_awaited_once()
    delete_job_mock.assert_awaited_once()
    revoke_mock.assert_awaited_once_with(
        k8s_backend._workload_delegations,
        config=None,
        workspace="default",
        deployment_name="server",
    )
