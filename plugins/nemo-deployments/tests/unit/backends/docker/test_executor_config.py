# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_deployments_plugin.backends.docker.config import (
    DEFAULT_WORKLOAD_TOKEN_WRITER_IMAGE,
    DockerAdditionalVolumeMount,
    DockerExecutorConfig,
)
from nemo_deployments_plugin.backends.labels import DEFAULT_RESOURCE_SCOPE
from pydantic import ValidationError


def test_docker_executor_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEMO_DEPLOYMENTS_DOCKER_ENDPOINT_MODE", raising=False)
    monkeypatch.delenv("NEMO_DEPLOYMENTS_DOCKER_NETWORK", raising=False)
    monkeypatch.delenv("MODELS_DOCKER_NETWORKING_MODE", raising=False)
    monkeypatch.delenv("MODELS_DOCKER_NETWORK", raising=False)

    cfg = DockerExecutorConfig()
    assert cfg.port_range_start == 49152
    assert cfg.port_range_end == 49251
    assert cfg.resource_scope == DEFAULT_RESOURCE_SCOPE
    assert cfg.oneshot_observe_timeout_seconds == 5
    assert cfg.network is None
    assert cfg.endpoint_mode == "host"
    assert cfg.workload_token_writer_image == DEFAULT_WORKLOAD_TOKEN_WRITER_IMAGE


def test_docker_executor_config_reads_network_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_DEPLOYMENTS_DOCKER_NETWORK", "nhx-e2e-test-network")
    monkeypatch.setenv("MODELS_DOCKER_NETWORK", "legacy-network")

    cfg = DockerExecutorConfig()

    assert cfg.network == "nhx-e2e-test-network"


def test_docker_executor_config_reads_legacy_models_network_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEMO_DEPLOYMENTS_DOCKER_NETWORK", raising=False)
    monkeypatch.setenv("MODELS_DOCKER_NETWORK", "legacy-network")

    cfg = DockerExecutorConfig()

    assert cfg.network == "legacy-network"


def test_docker_executor_config_reads_endpoint_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_DEPLOYMENTS_DOCKER_ENDPOINT_MODE", "NETWORK")
    monkeypatch.setenv("NEMO_DEPLOYMENTS_DOCKER_NETWORK", "nhx-e2e-test-network")

    cfg = DockerExecutorConfig()

    assert cfg.endpoint_mode == "network"


def test_docker_executor_config_reads_legacy_dond_endpoint_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEMO_DEPLOYMENTS_DOCKER_ENDPOINT_MODE", raising=False)
    monkeypatch.setenv("MODELS_DOCKER_NETWORKING_MODE", "dond")
    monkeypatch.setenv("MODELS_DOCKER_NETWORK", "legacy-network")

    cfg = DockerExecutorConfig()

    assert cfg.endpoint_mode == "network"


def test_docker_executor_config_rejects_network_endpoint_mode_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NEMO_DEPLOYMENTS_DOCKER_NETWORK", raising=False)
    monkeypatch.delenv("MODELS_DOCKER_NETWORK", raising=False)

    with pytest.raises(ValidationError, match="endpoint_mode='network' requires network"):
        DockerExecutorConfig(endpoint_mode="network")


def test_docker_executor_config_rejects_inverted_port_range() -> None:
    with pytest.raises(ValidationError, match="port_range_start must not exceed port_range_end"):
        DockerExecutorConfig(port_range_start=9200, port_range_end=9100)


def test_docker_executor_config_parses_additional_volume_mounts() -> None:
    cfg = DockerExecutorConfig(
        additional_volume_mounts=[
            DockerAdditionalVolumeMount(
                volume_name="gateway-tls",
                mount_path="/etc/nhx/gateway-tls",
                read_only=True,
            )
        ]
    )

    assert len(cfg.additional_volume_mounts) == 1
    mount = cfg.additional_volume_mounts[0]
    assert mount.volume_name == "gateway-tls"
    assert mount.mount_path == "/etc/nhx/gateway-tls"
    assert mount.read_only is True


def test_docker_executor_config_accepts_workload_token_writer_image() -> None:
    cfg = DockerExecutorConfig(workload_token_writer_image="registry.example.com/platform/busybox:stable")

    assert cfg.workload_token_writer_image == "registry.example.com/platform/busybox:stable"
