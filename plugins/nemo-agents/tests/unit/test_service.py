# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the agents plugin service wiring."""

from __future__ import annotations

from fastapi.routing import APIRoute
from nemo_agent_optimization_plugin.jobs.optimize import OptimizeJob
from nemo_agents_plugin.jobs.analyze_batch import AnalyzeBatchJob
from nemo_agents_plugin.jobs.evaluate_agent import EvaluateAgentJob
from nemo_agents_plugin.jobs.evaluate_suite import EvaluateSuiteJob
from nemo_agents_plugin.jobs.execute import ExecuteAgentJob
from nemo_agents_plugin.jobs.optimize_skills import OptimizeSkillsJob
from nemo_agents_plugin.service import AgentsService
from nemo_platform_plugin.scheduler import submit_path_for


def _mounted_routes() -> dict[str, set[str]]:
    service = AgentsService()
    routes: dict[str, set[str]] = {}
    for spec in service.get_routers():
        for route in spec.router.routes:
            if not isinstance(route, APIRoute):
                continue
            path = f"/apis/agents{spec.prefix}{route.path}".replace("{trailing_uri:path}", "{trailing_uri}")
            if route.methods is not None:
                routes.setdefault(path, set()).update(route.methods or set())
    return routes


def _mounted_post_paths() -> set[str]:
    """All POST paths mounted by AgentsService, regardless of which router owns them.

    Filters by HTTP method rather than description string, so copy-only docstring
    edits in service.py don't break route-shape tests.
    """
    return {path for path, methods in _mounted_routes().items() if "POST" in methods}


def test_evaluate_job_route_matches_generated_submit_path() -> None:
    assert submit_path_for(EvaluateAgentJob, workspace="{workspace}") in _mounted_post_paths()


def test_evaluate_suite_job_route_matches_generated_submit_path() -> None:
    assert submit_path_for(EvaluateSuiteJob, workspace="{workspace}") in _mounted_post_paths()


def test_execute_job_route_matches_generated_submit_path() -> None:
    assert submit_path_for(ExecuteAgentJob, workspace="{workspace}") in _mounted_post_paths()


def test_optimize_skills_job_route_matches_generated_submit_path() -> None:
    assert submit_path_for(OptimizeSkillsJob, workspace="{workspace}") in _mounted_post_paths()


def test_optimize_job_route_matches_generated_submit_path() -> None:
    assert submit_path_for(OptimizeJob, workspace="{workspace}") in _mounted_post_paths()


def test_analyze_job_route_matches_generated_submit_path() -> None:
    assert submit_path_for(AnalyzeBatchJob, workspace="{workspace}") in _mounted_post_paths()


def test_session_routes_are_mounted() -> None:
    routes = _mounted_routes()

    assert routes["/apis/agents/v2/workspaces/{workspace}/sessions"] == {"GET", "POST"}
    assert routes["/apis/agents/v2/workspaces/{workspace}/sessions/{name}"] == {"DELETE", "GET"}
    assert routes["/apis/agents/v2/workspaces/{workspace}/sessions/{name}/close"] == {"POST"}
