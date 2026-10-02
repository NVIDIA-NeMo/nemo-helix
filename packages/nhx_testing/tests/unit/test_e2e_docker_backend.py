# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Docker E2E backend."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nhx.testing.e2e import docker as docker_backend


class FakeDockerContainer:
    instances: list["FakeDockerContainer"] = []
    start_order: list[str] = []

    def __init__(self, image: str, **kwargs: Any):
        self.image = image
        self.kwargs = kwargs
        self.name: str | None = None
        self.network: object | None = None
        self.aliases: list[str] = []
        self.exposed_ports: list[int] = []
        self.env: dict[str, str] = {}
        self.volume_mappings: list[tuple[str, str, str]] = []
        self.container_kwargs: dict[str, Any] = {}
        self.stopped = False
        FakeDockerContainer.instances.append(self)

    def with_kwargs(self, **kwargs: Any) -> "FakeDockerContainer":
        self.container_kwargs.update(kwargs)
        return self

    def with_name(self, name: str) -> "FakeDockerContainer":
        self.name = name
        return self

    def with_network(self, network: object) -> "FakeDockerContainer":
        self.network = network
        return self

    def with_network_aliases(self, *aliases: str) -> "FakeDockerContainer":
        self.aliases.extend(aliases)
        return self

    def with_exposed_ports(self, *ports: int) -> "FakeDockerContainer":
        self.exposed_ports.extend(ports)
        return self

    def with_volume_mapping(self, source: str, target: str, mode: str) -> "FakeDockerContainer":
        self.volume_mappings.append((source, target, mode))
        return self

    def with_env(self, key: str, value: str) -> "FakeDockerContainer":
        self.env[key] = value
        return self

    def start(self) -> None:
        assert self.name is not None
        FakeDockerContainer.start_order.append(self.name)

    def stop(self) -> None:
        self.stopped = True

    def get_wrapped_container(self) -> object:
        return SimpleNamespace(
            status="running",
            short_id=self.name,
            reload=lambda: None,
            remove=lambda **_kwargs: None,
        )

    def get_exposed_port(self, _port: int) -> str:
        return "32768"

    def get_container_host_ip(self) -> str:
        return "localhost"


class FakeNetwork:
    def __init__(self):
        self.name = "nhx-e2e-test-network"
        self.created = False
        self.removed = False

    def create(self) -> None:
        self.created = True

    def remove(self) -> None:
        self.removed = True


def test_docker_backend_starts_clickhouse_sidecar_on_api_network(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("platform: {}\n")
    FakeDockerContainer.instances = []
    FakeDockerContainer.start_order = []

    monkeypatch.setattr(docker_backend, "DockerContainer", FakeDockerContainer)
    monkeypatch.setattr(docker_backend, "Network", FakeNetwork)
    monkeypatch.setattr(docker_backend.Docker, "_wait_for_clickhouse_healthy", lambda _self: None)
    monkeypatch.setattr(docker_backend.Docker, "_wait_for_healthy", lambda _self: None)
    monkeypatch.setattr(docker_backend.Docker, "_collect_logs", lambda _self: None)

    backend = docker_backend.Docker(config_path, registry="registry.example/nhx", tag="test-tag")

    try:
        backend.start()

        clickhouse_container, api_container = FakeDockerContainer.instances
        assert clickhouse_container.image == docker_backend.DEFAULT_E2E_CLICKHOUSE_IMAGE
        assert clickhouse_container.network is backend.network
        assert clickhouse_container.aliases == [docker_backend.NHX_CLICKHOUSE_NETWORK_ALIAS]
        assert clickhouse_container.env["CLICKHOUSE_SKIP_USER_SETUP"] == "1"

        assert api_container.network is backend.network
        assert api_container.env["NHX_INTAKE_CLICKHOUSE_URL"] == docker_backend._clickhouse_api_url()
        assert api_container.env["NEMO_JOBS_DEFAULT_DOCKER_NETWORK"] == "nhx-e2e-test-network"
        assert api_container.env["NEMO_DEPLOYMENTS_DOCKER_NETWORK"] == "nhx-e2e-test-network"
        assert api_container.env["NEMO_DEPLOYMENTS_DOCKER_ENDPOINT_MODE"] == "network"
        assert api_container.env["NHX_IMAGE_REGISTRY"] == "registry.example/nhx"
        assert api_container.env["NHX_IMAGE_TAG"] == "test-tag"

        assert FakeDockerContainer.start_order == [
            clickhouse_container.name,
            api_container.name,
        ]
    finally:
        backend.stop()


def test_clickhouse_image_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("NHX_E2E_CLICKHOUSE_IMAGE", "example/clickhouse:test")

    assert docker_backend._clickhouse_image() == "example/clickhouse:test"
