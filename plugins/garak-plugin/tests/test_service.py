# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the garak_plugin plugin service wiring."""

from __future__ import annotations

from unittest.mock import patch

from fastapi import APIRouter
from fastapi.routing import APIRoute
from garak_plugin.jobs.scan import ScanJob
from garak_plugin.service import GarakPluginService
from nemo_helix_plugin.scheduler import submit_path_for


def _mounted_post_paths() -> set[str]:
    service = GarakPluginService()
    paths: set[str] = set()
    for spec in service.get_routers():
        for route in spec.router.routes:
            if isinstance(route, APIRoute) and route.methods is not None and "POST" in route.methods:
                paths.add(f"/apis/garak-plugin{spec.prefix}{route.path}")
    return paths


def test_scan_job_submit_route_is_mounted() -> None:
    assert submit_path_for(ScanJob, workspace="{workspace}") in _mounted_post_paths()


def test_scan_job_routes_do_not_override_default_profile() -> None:
    """The execution profile scan jobs land on is controlled by the garak_plugin plugin
    config (see ScanJob.compile / garak_plugin.config), not by add_job_routes'
    default_profile — that would be dead code, since compile() always sets a profile."""
    with patch("garak_plugin.service.add_job_routes") as mock_add_job_routes:
        mock_add_job_routes.return_value = APIRouter()
        GarakPluginService().get_routers()

    mock_add_job_routes.assert_called_once()
    assert "default_profile" not in mock_add_job_routes.call_args.kwargs
