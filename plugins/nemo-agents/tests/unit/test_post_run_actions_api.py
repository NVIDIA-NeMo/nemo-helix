# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the installed post-run actions route."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agents_plugin.api.v2 import post_run_actions as post_run_actions_router_module


def _test_client() -> TestClient:
    app = FastAPI()
    app.include_router(post_run_actions_router_module.router, prefix="/apis/agents/v2")
    return TestClient(app, raise_server_exceptions=False)


def test_lists_the_installed_actions() -> None:
    resp = _test_client().get("/apis/agents/v2/post-run-actions")

    assert resp.status_code == 200
    listed = {action["kind"]: action for action in resp.json()}
    assert listed["agents.noop"] == {
        "kind": "agents.noop",
        "title": "No action",
        "description": "Do nothing after the agent runs; the job's results are the agent's own.",
    }
    assert listed["compass.persist"]["title"] == "Compass: save insights"
    assert listed["compass.preview"]["title"] == "Compass: preview insights"
