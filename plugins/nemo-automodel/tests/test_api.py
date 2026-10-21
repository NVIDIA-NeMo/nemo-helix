# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_automodel_plugin.contributor import AutomodelContributor
from nemo_customizer.router import CustomizationRouterService


def _route_paths(app: FastAPI) -> set[str]:
    """Collect all route paths, compatible with FastAPI 0.138+ _IncludedRouter."""
    paths: set[str] = set()
    queue = list(app.routes)
    while queue:
        route = queue.pop()
        # Route types differ (APIRoute, Mount, included routers); any with a string path counts.
        path = getattr(route, "path", None)
        if isinstance(path, str):
            paths.add(path)
        fn = getattr(route, "effective_candidates", None)
        if callable(fn):
            queue.extend(fn())  # type: ignore[arg-type]
    return paths


def _make_automodel_app() -> FastAPI:
    app = FastAPI()
    for spec in AutomodelContributor().get_routers():
        app.include_router(spec.router, prefix=spec.prefix, tags=[spec.tag] if spec.tag else None)
    return app


def test_automodel_jobs_collection_path() -> None:
    paths = _route_paths(_make_automodel_app())
    assert "/v2/workspaces/{workspace}/automodel/jobs" in paths


def test_contributor_exposes_no_per_backend_healthz() -> None:
    """Backend health is not exposed per contributor; the customization router
    reports a single ``/v2/healthz`` that enumerates contributors instead."""
    paths = _route_paths(_make_automodel_app())
    assert not any(p.endswith("/healthz") for p in paths)


def test_customization_router_healthz_lists_contributors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nemo_customizer.router.discover_customization_contributors",
        lambda: {"automodel": AutomodelContributor()},
    )
    service = CustomizationRouterService()
    app = FastAPI()
    for spec in service.get_routers():
        prefix = spec.prefix or ""
        app.include_router(spec.router, prefix=prefix)

    client = TestClient(app)
    assert client.get("/v2/healthz").json()["contributors"] == ["automodel"]


def test_workspace_isolation_list_uses_path_segment() -> None:
    """Job routes are under ``/v2/workspaces/{workspace}/automodel/jobs`` — distinct per workspace."""
    app = _make_automodel_app()
    paths = _route_paths(app)
    assert "/v2/workspaces/{workspace}/automodel/jobs" in paths
