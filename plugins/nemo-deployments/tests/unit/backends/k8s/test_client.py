# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from nemo_deployments_plugin.backends.k8s.backend import K8sDeploymentBackend
from nemo_deployments_plugin.backends.k8s.client import KubernetesClients, build_api_client
from nemo_deployments_plugin.backends.registry import BACKEND_CLASSES, ExecutorRegistry, ExecutorSpec
from nemo_helix import AsyncNeMoHelix


@pytest.fixture(autouse=True)
def _reset_kubernetes_modules_cache() -> Iterator[None]:
    import nemo_deployments_plugin.backends.k8s.client as k8s_client

    k8s_client._kubernetes_modules_cache = None
    k8s_client._k8s_client_module = None
    yield
    k8s_client._kubernetes_modules_cache = None
    k8s_client._k8s_client_module = None


def test_k8s_backend_registered() -> None:
    assert "k8s" in BACKEND_CLASSES
    assert BACKEND_CLASSES["k8s"] is K8sDeploymentBackend


def test_executor_registry_accepts_k8s_backend() -> None:
    sdk = AsyncNeMoHelix(base_url="http://localhost:8080")
    with patch("nemo_deployments_plugin.backends.k8s.backend.KubernetesClients"):
        registry = ExecutorRegistry.from_config(
            sdk,
            [ExecutorSpec(name="cluster", backend="k8s", config={"default_namespace": "default"})],
        )
    backend = registry.resolve("cluster")
    assert isinstance(backend, K8sDeploymentBackend)
    assert backend.executor_config.default_namespace == "default"


def test_k8s_backend_missing_kubernetes_package(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "kubernetes":
            raise ImportError("no kubernetes")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(RuntimeError, match="kubernetes package is required"):
        K8sDeploymentBackend(MagicMock(), {})


def test_build_api_client_prefers_in_cluster() -> None:
    with (
        patch("kubernetes.config.load_incluster_config") as mock_incluster,
        patch("kubernetes.config.load_kube_config") as mock_kube,
        patch("kubernetes.client.ApiClient") as mock_api_client,
        patch("kubernetes.client.Configuration"),
    ):
        build_api_client()
        mock_incluster.assert_called_once()
        mock_kube.assert_not_called()
        mock_api_client.assert_called_once()


def test_build_api_client_falls_back_to_kubeconfig() -> None:
    from kubernetes import config

    with (
        patch("kubernetes.config.load_incluster_config", side_effect=config.ConfigException("not in cluster")),
        patch("kubernetes.config.load_kube_config") as mock_kube,
        patch("kubernetes.client.ApiClient"),
        patch("kubernetes.client.Configuration") as mock_configuration,
    ):
        build_api_client()
        mock_kube.assert_called_once_with(client_configuration=mock_configuration.return_value)


def test_build_api_client_honors_explicit_kubeconfig_path() -> None:
    with (
        patch("kubernetes.config.load_kube_config") as mock_kube,
        patch("kubernetes.config.load_incluster_config") as mock_incluster,
        patch("kubernetes.client.ApiClient"),
        patch("kubernetes.client.Configuration"),
    ):
        build_api_client(kubeconfig_path="/tmp/kubeconfig")
        mock_kube.assert_called_once()
        assert mock_kube.call_args.kwargs["config_file"] == "/tmp/kubeconfig"
        mock_incluster.assert_not_called()


def test_kubernetes_clients_close_releases_api_client() -> None:
    mock_api_client = MagicMock()
    with patch("nemo_deployments_plugin.backends.k8s.client.build_api_client", return_value=mock_api_client):
        clients = KubernetesClients()
        _ = clients.core_v1
        clients.close()
        mock_api_client.close.assert_called_once()
        assert clients._api_client is None
        assert clients._core_v1 is None
