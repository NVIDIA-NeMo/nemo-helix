# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Insight routes act only on Insights in the workspace named in the path.

Authorization checks the caller's permissions against the path's workspace, so
a route that loads an Insight by id alone would let a caller with access to one
workspace read, change or delete an Insight in another. These run the real
Compass and Entities services so the lookup the route actually makes is the one
under test.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from nemo_compass_plugin.service import CompassService
from nhx.core.entities.service import EntitiesService
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
from nhx.testing import ClientContext, create_test_client

pytestmark = pytest.mark.integration

OWNER = "team-owner"
OTHER = "team-other"


class _TestCompassService(NemoServiceAdapter):
    def __init__(self) -> None:
        super().__init__(CompassService())


def _insights_url(workspace: str, insight_id: str = "") -> str:
    url = f"/apis/compass/v2/workspaces/{workspace}/insights"
    return f"{url}/{insight_id}" if insight_id else url


@pytest.fixture
def http() -> Iterator[TestClient]:
    with create_test_client(
        _TestCompassService,
        EntitiesService,
        client_type=ClientContext,
        workspaces=[OWNER, OTHER],
        workspace=OWNER,
    ) as ctx:
        yield ctx.test_client


@pytest.fixture
def insight(http: TestClient) -> dict[str, Any]:
    response = http.post(
        _insights_url(OWNER),
        json={"title": "Retrieval truncates long documents", "agent": "research-agent", "description": "d"},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_an_insight_is_readable_in_its_own_workspace(http: TestClient, insight: dict[str, Any]) -> None:
    """The control: the lookup the other tests rely on finds the Insight at all."""
    response = http.get(_insights_url(OWNER, insight["id"]))

    assert response.status_code == 200, response.text
    assert response.json()["workspace"] == OWNER


def test_another_workspace_cannot_read_an_insight(http: TestClient, insight: dict[str, Any]) -> None:
    response = http.get(_insights_url(OTHER, insight["id"]))

    assert response.status_code == 404, response.text


def test_another_workspace_cannot_update_an_insight(http: TestClient, insight: dict[str, Any]) -> None:
    response = http.patch(_insights_url(OTHER, insight["id"]), json={"description": "overwritten"})

    assert response.status_code == 404, response.text
    assert http.get(_insights_url(OWNER, insight["id"])).json()["description"] == "d"


def test_another_workspace_cannot_delete_an_insight(http: TestClient, insight: dict[str, Any]) -> None:
    response = http.delete(_insights_url(OTHER, insight["id"]))

    assert response.status_code == 404, response.text
    assert http.get(_insights_url(OWNER, insight["id"])).status_code == 200
