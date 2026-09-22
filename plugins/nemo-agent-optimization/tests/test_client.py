# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The typed client for this plugin's own API: URL building and response parsing."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from nemo_agent_optimization_plugin.client import AgentOptimizationClient
from nemo_agent_optimization_plugin.schemas.strategies import STRATEGIES_PATH
from nemo_helix_plugin.client.errors import NotFoundError

BASE = "http://platform"

_LISTING = {"data": [{"name": "nat", "description": "Numeric HPO."}]}


def _client(handler: Callable[[httpx.Request], httpx.Response], *, base_url: str = BASE) -> AgentOptimizationClient:
    return AgentOptimizationClient(
        base_url=base_url,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_list_strategies_asks_the_route_the_service_mounts() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_LISTING)

    listing = _client(handler).list_strategies().data()

    assert [strategy.name for strategy in listing.data] == ["nat"]
    assert listing.data[0].description == "Numeric HPO."
    assert seen[0].method == "GET"
    assert seen[0].url == f"{BASE}{STRATEGIES_PATH}"


def test_a_trailing_slash_on_the_base_url_does_not_double_up() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_LISTING)

    _client(handler, base_url=f"{BASE}/").list_strategies()

    assert seen[0].url == f"{BASE}{STRATEGIES_PATH}"


def test_an_error_status_raises_a_typed_error() -> None:
    """The CLI reports every failure the same way, so it needs them all as one exception family."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "no such route"})

    with pytest.raises(NotFoundError):
        _client(handler).list_strategies()
