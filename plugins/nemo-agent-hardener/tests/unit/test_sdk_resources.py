# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `client.agent_hardener.runs` / `.manifests` — the auto-pagination bound on `limit`."""

from __future__ import annotations

import types
from typing import Any

from nemo_agent_hardener_plugin.sdk import AgentHardenerPluginResource


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
