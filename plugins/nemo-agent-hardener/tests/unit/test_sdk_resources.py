# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `client.agent_hardener.runs` / `.manifests` — the auto-pagination bound on `limit`."""

from __future__ import annotations

import types
from typing import Any

import pytest
from _doubles import make_sdk
from nemo_agent_hardener_plugin.sdk import AgentHardenerPluginResource
from nemo_platform_plugin.agent_hardener.client import AgentHardenerClient
from nemo_platform_plugin.agent_hardener.types import (
    AgentHardenerManifest,
    InspectProjectRequest,
    InspectProjectResponse,
    ManifestInit,
    ManifestUpdate,
    ValidateModelRequest,
    ValidateModelResponse,
)


class _AutoPaginatingPage:
    """Stand-in for `SyncDefaultPagination`: iterating it walks every page, like the real client.

    `_base_client.BaseSyncPage.__iter__` loops `while True: ... page.get_next_page()`, so `page_size`
    bounds a page, never the total. That is what made `status --limit` return the whole history.
    """

    def __init__(self, records: list[Any], page_size: int, requests: list[int]) -> None:
        self._records = records
        self._page_size = page_size
        self._requests = requests

    def __iter__(self) -> Any:
        for start in range(0, len(self._records), self._page_size):
            self._requests.append(start)  # one HTTP round-trip per page
            yield from self._records[start : start + self._page_size]


class _FakeEntities:
    def __init__(self, total: int) -> None:
        self.records = [
            types.SimpleNamespace(data={"agent": f"a{i}", "status": "completed"}, name=f"run-{i}", created_at=i)
            for i in range(total)
        ]
        self.requests: list[int] = []
        self.calls: list[dict[str, Any]] = []

    def list(self, entity_type: str, **kwargs: Any) -> _AutoPaginatingPage:
        self.calls.append({"entity_type": entity_type, **kwargs})
        return _AutoPaginatingPage(self.records, kwargs["page_size"], self.requests)


def _resource(total: int) -> tuple[AgentHardenerPluginResource, _FakeEntities]:
    entities = _FakeEntities(total)
    platform: Any = types.SimpleNamespace(entities=entities)
    return AgentHardenerPluginResource(platform), entities


def test_runs_list_returns_at_most_limit() -> None:
    resource, entities = _resource(total=200)

    runs = resource.runs.list(workspace="default", limit=5)

    assert len(runs) == 5
    assert [r["name"] for r in runs] == [f"run-{i}" for i in range(5)]
    assert len(entities.requests) == 1  # one page fetched, not 40


def test_runs_list_sorts_newest_first_in_the_query() -> None:
    resource, entities = _resource(total=3)
    resource.runs.list(workspace="ws", limit=2)
    assert entities.calls[0]["sort"] == "-created_at"
    assert entities.calls[0]["workspace"] == "ws"


def test_runs_list_handles_fewer_records_than_limit() -> None:
    resource, _ = _resource(total=2)
    assert len(resource.runs.list(limit=20)) == 2


def test_latest_fetches_one_record_not_the_whole_history() -> None:
    resource, entities = _resource(total=200)

    latest = resource.runs.latest(workspace="default")

    assert latest is not None and latest["name"] == "run-0"
    assert len(entities.requests) == 1


def test_latest_is_none_when_no_runs_exist() -> None:
    resource, _ = _resource(total=0)
    assert resource.runs.latest() is None


def test_manifests_list_is_bounded_too() -> None:
    resource, entities = _resource(total=50)
    assert len(resource.manifests.list(limit=3)) == 3
    assert len(entities.requests) == 1


def test_manifest_create_uses_typed_agent_hardener_client(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _AgentHardener:
        def create_manifest(self, *, workspace: str, body: ManifestInit) -> Any:
            captured["workspace"] = workspace
            captured["body"] = body
            return types.SimpleNamespace(
                data=lambda: AgentHardenerManifest(name=body.name, workspace=workspace, agent=body.agent or "")
            )

    def _client_from_platform(_platform: Any, client_cls: Any) -> Any:
        assert client_cls is AgentHardenerClient
        return _AgentHardener()

    monkeypatch.setattr("nemo_agent_hardener_plugin.sdk.client_from_platform", _client_from_platform)

    result = AgentHardenerPluginResource(make_sdk()).manifests.create(
        workspace="ws", name="finance", agent="default/finance"
    )

    assert isinstance(captured["body"], ManifestInit)
    assert captured["body"].name == "finance"
    assert result["name"] == "finance"
    assert result["workspace"] == "ws"


def test_manifest_update_validate_and_inspect_use_typed_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _AgentHardener:
        def update_manifest(self, *, workspace: str, name: str, body: ManifestUpdate) -> Any:
            captured["update"] = (workspace, name, body)
            return types.SimpleNamespace(
                data=lambda: AgentHardenerManifest(name=name, workspace=workspace, rounds=body.rounds or 1)
            )

        def validate_model(self, *, workspace: str, body: ValidateModelRequest) -> Any:
            captured["validate"] = (workspace, body)
            return types.SimpleNamespace(data=lambda: ValidateModelResponse(ok=True, available=[body.model or ""]))

        def inspect_project(self, *, workspace: str, body: InspectProjectRequest) -> Any:
            captured["inspect"] = (workspace, body)
            return types.SimpleNamespace(
                data=lambda: InspectProjectResponse(dockerfile=body.dockerfile or "Dockerfile")
            )

    def _client_from_platform(_platform: Any, client_cls: Any) -> Any:
        assert client_cls is AgentHardenerClient
        return _AgentHardener()

    monkeypatch.setattr("nemo_agent_hardener_plugin.sdk.client_from_platform", _client_from_platform)
    manifests = AgentHardenerPluginResource(make_sdk()).manifests

    updated = manifests.update("finance", workspace="ws", rounds=3)
    verdict = manifests.validate_model(workspace="ws", model="model-a", base_url="https://models.example/v1")
    detected = manifests.inspect_project("ws/project", dockerfile="Dockerfile.dev", workspace="ws")

    assert isinstance(captured["update"][2], ManifestUpdate)
    assert updated["rounds"] == 3
    assert isinstance(captured["validate"][1], ValidateModelRequest)
    assert verdict["ok"] is True
    assert isinstance(captured["inspect"][1], InspectProjectRequest)
    assert captured["inspect"][1].project_fileset == "ws/project"
    assert detected["dockerfile"] == "Dockerfile.dev"
