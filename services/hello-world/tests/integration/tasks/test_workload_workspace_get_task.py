# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
import respx
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nhx.common.config import Configuration, HelixConfig
from nhx.common.jobs.constants import TASK_CONFIG_ENVVAR
from nhx.hello_world.tasks.workload_workspace_get.run import run as task_run


@pytest.fixture(autouse=True)
def clear_config_cache():
    Configuration.clear_cache()
    try:
        yield
    finally:
        Configuration.clear_cache()


@pytest.fixture
def platform_base_url():
    base_url = "http://nhx.example.test"
    Configuration.set_override(HelixConfig(base_url=base_url))
    try:
        yield base_url
    finally:
        Configuration.clear_override(HelixConfig)


class _StubWorkspaces:
    """Recording fake matching the typed WorkspacesClient shape (get_workspace + .data())."""

    def __init__(self) -> None:
        self.requested: list[str] = []

    def get_workspace(self, *, name: str, **kwargs) -> SimpleNamespace:
        self.requested.append(name)
        return SimpleNamespace(data=lambda: SimpleNamespace(name=name))


@pytest.fixture
def stub_workspaces_client(monkeypatch) -> _StubWorkspaces:
    """Replace WorkspacesClient.from_client so the task talks to an in-memory stub."""
    stub = _StubWorkspaces()
    monkeypatch.setattr(WorkspacesClient, "from_client", classmethod(lambda cls, client: stub))
    return stub


def test_workload_workspace_get_uses_task_client_factory(stub_workspaces_client, monkeypatch):
    client_factory_calls: list[str] = []

    def get_task_nemo_client(service_name: str) -> None:
        client_factory_calls.append(service_name)
        return None

    monkeypatch.setenv(TASK_CONFIG_ENVVAR, '{"workspace":"workload-read-target"}')
    monkeypatch.setattr("nhx.hello_world.tasks.workload_workspace_get.run.get_task_nemo_client", get_task_nemo_client)

    exit_code = task_run()

    assert exit_code == 0
    assert client_factory_calls == ["jobs"]
    assert stub_workspaces_client.requested == ["workload-read-target"]


def test_workload_workspace_get_uses_injected_client_without_workload_token(stub_workspaces_client, monkeypatch):
    monkeypatch.setenv(TASK_CONFIG_ENVVAR, '{"workspace":"workload-read-target"}')
    monkeypatch.delenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, raising=False)
    monkeypatch.delenv("NEMO_WORKLOAD_TOKEN", raising=False)
    monkeypatch.delenv("NEMO_WORKLOAD_TOKEN_FILE", raising=False)

    exit_code = task_run(client=cast(NemoClient, object()))

    assert exit_code == 0
    assert stub_workspaces_client.requested == ["workload-read-target"]


@respx.mock
def test_workload_workspace_get_uses_task_client_without_workload_token(monkeypatch, tmp_path, platform_base_url):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv(TASK_CONFIG_ENVVAR, '{"workspace":"workload-read-target"}')
    monkeypatch.setenv("NHX_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("NHX_BASE_URL", platform_base_url)
    monkeypatch.setenv(
        "NHX_PRINCIPAL",
        json.dumps(
            {
                "id": "creator@example.com",
                "email": "creator@example.com",
                "groups": ["engineering"],
            }
        ),
    )
    monkeypatch.delenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, raising=False)
    monkeypatch.delenv("NHX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("NEMO_WORKLOAD_TOKEN", raising=False)
    monkeypatch.delenv("NEMO_WORKLOAD_TOKEN_FILE", raising=False)

    def workspace_response(request: httpx.Request) -> httpx.Response:
        if request.headers.get("X-NHX-Principal-Id") != "service:jobs":
            return httpx.Response(401, json={"detail": "Unauthorized"})
        if request.headers.get("X-NHX-Principal-On-Behalf-Of") != "creator@example.com":
            return httpx.Response(403, json={"detail": "Forbidden"})
        if request.headers.get("Authorization") == "Bearer service:jobs":
            return httpx.Response(401, json={"detail": "Unauthorized"})
        return httpx.Response(
            200,
            json={
                "id": "workspace-id",
                "name": "workload-read-target",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
        )

    workspace_route = respx.get(f"{platform_base_url}/apis/entities/v2/workspaces/workload-read-target").mock(
        side_effect=workspace_response
    )

    exit_code = task_run()

    assert exit_code == 0
    assert workspace_route.called
