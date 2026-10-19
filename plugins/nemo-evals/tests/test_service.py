# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the evals plugin service surface."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_evals.service import EvaluatorPluginService


def _route_paths(app: FastAPI) -> set[str]:
    """Collect all route paths, compatible with FastAPI 0.138+ _IncludedRouter."""
    paths: set[str] = set()
    queue = list(app.routes)
    while queue:
        route = queue.pop()
        if hasattr(route, "path"):
            path = route.path
            if isinstance(path, str):
                paths.add(path)
        fn = getattr(route, "effective_candidates", None)
        if callable(fn):
            queue.extend(fn())  # type: ignore[arg-type]
    return paths


def test_service_health_route_mounts_with_valid_prefix() -> None:
    app = FastAPI()
    service = EvaluatorPluginService()
    for spec in service.get_routers():
        app.include_router(spec.router, prefix=spec.prefix)

    response = TestClient(app).get("/v1/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "plugin": "evals",
        "status": "ok",
        "mode": "sdk-backed-job-scaffold",
        "jobs": ["evals.evaluate", "evals.agent-evaluate", "evals.retrieve-eval"],
    }


def test_service_mounts_evaluator_job_collection_at_sdk_route() -> None:
    app = FastAPI()
    service = EvaluatorPluginService()
    for spec in service.get_routers():
        app.include_router(spec.router, prefix=spec.prefix)

    route_paths = _route_paths(app)

    assert "/v2/workspaces/{workspace}/evaluate/jobs" in route_paths


def test_service_mounts_retrieve_eval_job_collection_at_sdk_route() -> None:
    app = FastAPI()
    service = EvaluatorPluginService()
    for spec in service.get_routers():
        app.include_router(spec.router, prefix=spec.prefix)

    assert "/v2/workspaces/{workspace}/retrieve-eval/jobs" in _route_paths(app)
