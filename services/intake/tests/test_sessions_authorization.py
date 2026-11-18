# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session deletion authorization through the real HTTP middleware and WASM PDP."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from fastapi.testclient import TestClient
from nhx.core.auth.app.embedded_pdp import set_policy_data
from nhx.intake.config import ClickHouseConfig, IntakeConfig
from nhx.intake.service import IntakeService
from nhx.intake.spans.api.dependencies import get_spans_service, require_workspace_access
from nhx.testing import create_test_client


def test_session_delete_authorization_precedes_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    # This test exercises Intake's static rules, independent of optional plugins
    # installed in the developer's environment.
    monkeypatch.setattr("nemo_helix_plugin.authz_discovery.discover_plugin_authz", lambda: [])
    monkeypatch.setattr(IntakeService, "is_ready", AsyncMock(return_value=True))
    service = SimpleNamespace(delete_session=AsyncMock())
    config = IntakeConfig(clickhouse_config=ClickHouseConfig(url="http://127.0.0.1:1"))
    authz_path = Path(__file__).parents[2] / "core/auth/src/nhx/core/auth/assets/static-authz.yaml"
    policy = yaml.safe_load(authz_path.read_text())
    policy["authz"]["principals"] = {
        "viewer@test.com": {"workspaces": {"default": ["Viewer"]}},
        "editor@test.com": {"workspaces": {"default": ["Editor"]}},
        "admin@test.com": {"workspaces": {"default": ["Admin"]}},
        "platform-admin@test.com": {"workspaces": {"system": ["HelixAdmin"]}},
    }
    policy["authz"]["workspaces"] = {"default": {}, "other": {}}

    with create_test_client(
        IntakeService,
        client_type=TestClient,
        auth_enabled=True,
        auth_bundle_cache_seconds=60,
        auth_policy_data_refresh_interval=60,
        service_configs={IntakeService: config},
        dependency_overrides={get_spans_service: lambda: service, require_workspace_access: lambda: None},
    ) as client:
        cases = [
            ("viewer@test.com", "default", "intake:write", 403),
            ("editor@test.com", "default", "intake:write", 403),
            ("editor@test.com", "default", "platform:write", 403),
            ("admin@test.com", "default", "intake:read", 403),
            ("admin@test.com", "default", "platform:read", 403),
            ("admin@test.com", "default", "jobs:write", 403),
            ("admin@test.com", "other", "intake:write", 403),
            ("admin@test.com", "default", "intake:write", 204),
            ("admin@test.com", "default", "platform:write", 204),
            ("platform-admin@test.com", "other", "intake:write", 204),
            ("platform-admin@test.com", "other", "intake:read", 403),
        ]
        for principal, workspace, scope, expected in cases:
            # Refresh from the fixture for each request, independent of the auth
            # service's background policy-data refresh.
            set_policy_data(policy)
            service.delete_session.reset_mock()
            response = client.delete(
                f"/apis/intake/v2/workspaces/{workspace}/sessions/target",
                headers={"X-NHX-Principal-Id": principal, "X-NHX-Scopes": scope},
            )
            assert response.status_code == expected, (principal, workspace, scope, response.text)
            if expected == 204:
                service.delete_session.assert_awaited_once_with(workspace=workspace, session_id="target")
            else:
                assert response.json() == {"detail": "Forbidden"}
                service.delete_session.assert_not_awaited()
