# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for auth service configuration."""


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
    assert cfg.embedded_pdp_auto_build_wasm is True
    assert cfg.bundle_cache_seconds == 5
    assert cfg.admin_email is None
    assert cfg.default_workspace == "default"


def test_default_workspace_custom():
    """Test custom default workspace."""
    from nhx.core.auth.config import AuthServiceConfig

    cfg = AuthServiceConfig(default_workspace="my-workspace")

    assert cfg.default_workspace == "my-workspace"
