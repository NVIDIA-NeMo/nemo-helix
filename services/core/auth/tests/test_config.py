# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for auth service configuration."""

import pytest
from pydantic import ValidationError


def test_auth_config_defaults():
    """Test default auth config."""
    from nhx.core.auth.config import AuthServiceConfig

    cfg = AuthServiceConfig()

    # Shared config defaults
    assert cfg.enabled is False
    assert cfg.policy_decision_point_base_url == "http://localhost:8080"

    # Auth service specific defaults
    assert cfg.policy_decision_point_provider == "embedded"
    assert cfg.policy_decision_point_request_timeout_seconds == 5
    assert cfg.bundle_cache_seconds == 5
    assert cfg.admin_email is None
    assert cfg.default_workspace == "default"
    assert cfg.opa_startup_timeout_seconds == 15
    assert cfg.opa_request_timeout_seconds == 3
    assert cfg.opa_shutdown_timeout_seconds == 5


def test_default_workspace_custom():
    """Test custom default workspace."""
    from nhx.core.auth.config import AuthServiceConfig

    cfg = AuthServiceConfig(default_workspace="my-workspace")

    assert cfg.default_workspace == "my-workspace"


@pytest.mark.parametrize(
    "field", ["opa_startup_timeout_seconds", "opa_request_timeout_seconds", "opa_shutdown_timeout_seconds"]
)
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), 301])
def test_managed_opa_timeouts_are_positive_and_bounded(field, value):
    from nhx.core.auth.config import AuthServiceConfig

    with pytest.raises(ValidationError):
        AuthServiceConfig(**{field: value})
