# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``with_options`` must not carry a clone's cached plugin-resource namespaces across."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
from nemo_platform_plugin.client.client import NemoClient

BASE = "http://test:8000"


def _make_client(factory: MagicMock, monkeypatch) -> NemoClient:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    client = NemoClient(base_url=BASE, workspace="default", http_client=http_client)
    resources = MagicMock(sync_resource=factory, async_resource=None)
    discovered = MagicMock(get=MagicMock(return_value=resources))
    monkeypatch.setattr("nemo_platform_plugin.discovery.discover_sdk", lambda: discovered)
    return client


def test_with_options_rebuilds_cached_plugin_resources(monkeypatch) -> None:
    factory = MagicMock(side_effect=lambda _owner: object())
    client = _make_client(factory, monkeypatch)

    first = client.example
    assert factory.call_count == 1

    clone = client.with_options(headers={"X-Extra": "1"})

    # The clone must not reuse the resource built for the original client.
    second = clone.example
    assert factory.call_count == 2
    assert second is not first
    assert factory.call_args.args[0] is clone

    # The original keeps its own cached resource.
    assert client.example is first
    assert factory.call_count == 2


def test_with_options_isolates_the_cached_resource_set(monkeypatch) -> None:
    factory = MagicMock(side_effect=lambda _owner: object())
    client = _make_client(factory, monkeypatch)

    client.example
    clone_a = client.with_options(timeout=5.0)
    clone_b = client.with_options(headers={"X-Other": "1"})

    # The original's cache set is not shared with its clones.
    clone_a.other
    assert "other" in clone_a.__dict__["_cached_resources"]
    assert "other" not in client.__dict__.get("_cached_resources", set())
    assert "other" not in clone_b.__dict__.get("_cached_resources", set())
