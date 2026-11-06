# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType
from unittest.mock import MagicMock, patch

import nemo_deployments_plugin.backends.k8s.backend as k8s_backend
import nemo_deployments_plugin.backends.k8s.client as k8s_client
import pytest
from kubernetes import config
from nemo_deployments_plugin.backends.k8s.backend import K8sDeploymentBackend
from nemo_deployments_plugin.backends.k8s.client import KubernetesClients, build_api_client
from nemo_deployments_plugin.backends.registry import BACKEND_CLASSES, ExecutorRegistry, ExecutorSpec
from nemo_helix_plugin.client.client import AsyncNemoClient


@pytest.fixture(autouse=True)
def _reset_kubernetes_modules_cache() -> Iterator[None]:
    k8s_client._kubernetes_modules_cache = None
    k8s_client._k8s_client_module = None
    yield
    k8s_client._kubernetes_modules_cache = None
    k8s_client._k8s_client_module = None


def test_k8s_backend_registered() -> None:
    assert "k8s" in BACKEND_CLASSES
    assert BACKEND_CLASSES["k8s"] is K8sDeploymentBackend


def test_executor_registry_accepts_k8s_backend() -> None:
    client = AsyncNemoClient(base_url="http://localhost:8080")
    with patch("nemo_deployments_plugin.backends.k8s.backend.KubernetesClients"):
        registry = ExecutorRegistry.from_config(
            client,
            [ExecutorSpec(name="cluster", backend="k8s", config={"default_namespace": "default"})],
        )
    backend = registry.resolve("cluster")
    assert isinstance(backend, K8sDeploymentBackend)
    assert backend.executor_config.default_namespace == "default"


def test_k8s_backend_missing_kubernetes_package(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_kubernetes() -> ModuleType:
        raise ImportError("no kubernetes")

    monkeypatch.setattr(k8s_backend, "_import_kubernetes", missing_kubernetes)
    with pytest.raises(RuntimeError, match="kubernetes package is required"):
        K8sDeploymentBackend(MagicMock(), {})


def _fake_kubernetes_modules(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Install fake ``kubernetes.client``/``kubernetes.config`` where ``build_api_client`` looks them up."""
    fake_client = MagicMock()
    fake_config = MagicMock()
    fake_config.ConfigException = config.ConfigException
    monkeypatch.setattr(k8s_client, "_kubernetes_modules_cache", (fake_client, fake_config))
    return fake_client, fake_config


def test_build_api_client_prefers_in_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client, fake_config = _fake_kubernetes_modules(monkeypatch)

    build_api_client()
    fake_config.load_incluster_config.assert_called_once()
    fake_config.load_kube_config.assert_not_called()
    fake_client.ApiClient.assert_called_once()


def test_build_api_client_falls_back_to_kubeconfig(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client, fake_config = _fake_kubernetes_modules(monkeypatch)
    fake_config.load_incluster_config.side_effect = config.ConfigException("not in cluster")

    build_api_client()
    fake_config.load_kube_config.assert_called_once_with(client_configuration=fake_client.Configuration.return_value)


def test_build_api_client_honors_explicit_kubeconfig_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _, fake_config = _fake_kubernetes_modules(monkeypatch)

    build_api_client(kubeconfig_path="/tmp/kubeconfig")
    fake_config.load_kube_config.assert_called_once()
    assert fake_config.load_kube_config.call_args.kwargs["config_file"] == "/tmp/kubeconfig"
    fake_config.load_incluster_config.assert_not_called()


def test_kubernetes_clients_close_releases_api_client() -> None:
    mock_api_client = MagicMock()
    with patch("nemo_deployments_plugin.backends.k8s.client.build_api_client", return_value=mock_api_client):
        clients = KubernetesClients()
        _ = clients.core_v1
        clients.close()
        mock_api_client.close.assert_called_once()
        assert clients._api_client is None
        assert clients._core_v1 is None
