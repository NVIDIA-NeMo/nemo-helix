# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the auditor plugin service wiring."""

from __future__ import annotations

from unittest.mock import patch

from fastapi import APIRouter
from fastapi.routing import APIRoute
from nemo_auditor.jobs.audit import AuditJob
from nemo_auditor.service import AuditorPluginService
from nemo_helix_plugin.scheduler import submit_path_for


def _mounted_post_paths() -> set[str]:
    service = AuditorPluginService()
    paths: set[str] = set()
    for spec in service.get_routers():
        for route in spec.router.routes:
            if isinstance(route, APIRoute) and route.methods is not None and "POST" in route.methods:
                paths.add(f"/apis/auditor{spec.prefix}{route.path}")
    return paths


def test_audit_job_submit_route_is_mounted() -> None:
    assert submit_path_for(AuditJob, workspace="{workspace}") in _mounted_post_paths()


def test_audit_job_routes_do_not_override_default_profile() -> None:
    """The execution profile audit jobs land on is controlled by the auditor plugin
    config (see AuditJob.compile / nemo_auditor.config), not by add_job_routes'
    default_profile — that would be dead code, since compile() always sets a profile."""
    with patch("nemo_auditor.service.add_job_routes") as mock_add_job_routes:
        mock_add_job_routes.return_value = APIRouter()
        AuditorPluginService().get_routers()

    mock_add_job_routes.assert_called_once()
    assert "default_profile" not in mock_add_job_routes.call_args.kwargs
