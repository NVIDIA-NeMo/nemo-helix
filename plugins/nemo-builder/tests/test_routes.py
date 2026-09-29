# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The builder's routes, over HTTP, with the entity store faked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_builder_plugin.service import BuilderService
from nemo_helix_plugin.entity_client import get_entity_client


def _app(entities: AsyncMock) -> FastAPI:
    app = FastAPI()
    for spec in BuilderService().get_routers():
        app.include_router(spec.router, prefix=spec.prefix)
    app.dependency_overrides[get_entity_client] = lambda: entities
    return app


def _empty_page() -> MagicMock:
    page = MagicMock()
    page.data = []
    return page


class TestListingOneJobsImages:
    """How the credential broker finds what a job may publish, without reading the workspace."""

    def test_a_job_and_a_status_filter_on_the_stored_fields(self) -> None:
        entities = AsyncMock()
        entities.list.return_value = _empty_page()
        response = TestClient(_app(entities)).get(
            "/v2/workspaces/team-a/container-images", params={"job": "demo-1", "status": "pending"}
        )
        assert response.status_code == 200
        assert entities.list.await_args.kwargs["filter_obj"] == {
            "provenance.job": "team-a/demo-1",
            "status": "pending",
        }

    def test_the_job_is_always_in_the_path_workspace(self) -> None:
        """A job name is only unique within a workspace, so another workspace's cannot be named."""
        entities = AsyncMock()
        response = TestClient(_app(entities)).get(
            "/v2/workspaces/team-a/container-images", params={"job": "team-b/demo-1"}
        )
        assert response.status_code == 422
        entities.list.assert_not_awaited()

    def test_no_filter_lists_the_workspace(self) -> None:
        entities = AsyncMock()
        entities.list.return_value = _empty_page()
        TestClient(_app(entities)).get("/v2/workspaces/team-a/container-images")
        assert entities.list.await_args.kwargs["filter_obj"] is None

    def test_an_unknown_status_is_refused(self) -> None:
        entities = AsyncMock()
        response = TestClient(_app(entities)).get("/v2/workspaces/team-a/container-images", params={"status": "done"})
        assert response.status_code == 422
        entities.list.assert_not_awaited()
