# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for permission vs scope format validation."""

from pathlib import Path

import pytest
import yaml
from nemo_helix_plugin.authz_format import (
    is_valid_nhx_scope_id,
    is_valid_permission_id,
    is_wildcard_permission,
    validate_nhx_scope_strings_for_config,
    validate_permission_strings,
    validate_runtime_authorize_scopes,
    validate_static_authz_data,
)
from nhx.common.auth.exceptions import InvalidPermissionFormatError, InvalidScopeFormatError


class TestPermissionPatterns:
    def test_valid_permissions(self) -> None:
        assert is_valid_permission_id("secrets.read")
        assert is_valid_permission_id("models.create")
        assert is_valid_permission_id("data-designer.jobs.read")
        assert is_valid_permission_id("inference.gateway.model.exec")
        assert is_valid_permission_id("models.trust-remote-code.set")

    def test_invalid_permissions(self) -> None:
        assert not is_valid_permission_id("secrets:read")
        assert not is_valid_permission_id("read")
        assert not is_valid_permission_id("")
        assert not is_valid_permission_id("foo..bar")


class TestScopePatterns:
    def test_valid_nhx_scopes(self) -> None:
        assert is_valid_nhx_scope_id("secrets:read")
        assert is_valid_nhx_scope_id("platform:write")
        assert is_valid_nhx_scope_id("data-designer:read")

    def test_invalid_nhx_scopes(self) -> None:
        assert not is_valid_nhx_scope_id("secrets.read")
        assert not is_valid_nhx_scope_id("read")
        assert not is_valid_nhx_scope_id("")


class TestValidatePermissionStrings:
    def test_rejects_colon_syntax(self) -> None:
        with pytest.raises(InvalidPermissionFormatError, match="dots"):
            validate_permission_strings(["secrets:read"], context="test")

    def test_rejects_malformed(self) -> None:
        with pytest.raises(InvalidPermissionFormatError, match="dot-separated"):
            validate_permission_strings(["invalid"], context="test")

    def test_accepts_valid_list(self) -> None:
        validate_permission_strings(["secrets.read", "models.create"], context="test")

    def test_accepts_role_wildcard_star(self) -> None:
        """ServiceSystem / OPA uses '*' alone to mean all permissions (see static-authz.yaml)."""
        validate_permission_strings(["*"], context="test")
        validate_permission_strings(["secrets.read", "*"], context="test")
        assert is_wildcard_permission("*")
        assert not is_wildcard_permission("**")


class TestValidateScopeStringsForConfig:
    def test_rejects_permission_syntax(self) -> None:
        with pytest.raises(InvalidScopeFormatError, match="colons"):
            validate_nhx_scope_strings_for_config(["secrets.read"], context="test")

    def test_rejects_bad_scope(self) -> None:
        with pytest.raises(InvalidScopeFormatError, match="area:action"):
            validate_nhx_scope_strings_for_config(["badscope"], context="test")

    def test_accepts_valid(self) -> None:
        validate_nhx_scope_strings_for_config(["secrets:read", "platform:read"], context="test")


class TestValidateRuntimeAuthorizeScopes:
    def test_allows_oidc_style_scopes(self) -> None:
        validate_runtime_authorize_scopes(["openid", "email"])

    def test_rejects_permission_like_values(self) -> None:
        with pytest.raises(InvalidScopeFormatError, match="permission syntax"):
            validate_runtime_authorize_scopes(["secrets.read"])

    def test_none_or_empty(self) -> None:
        validate_runtime_authorize_scopes(None)
        validate_runtime_authorize_scopes([])


class TestValidateStaticAuthzData:
    def test_valid_minimal_structure(self) -> None:
        data = {
            "authz": {
                "roles": {
                    "Viewer": {
                        "permissions": ["secrets.read"],
                    }
                },
                "endpoints": {
                    "/api/x": {
                        "get": {
                            "permissions": ["secrets.read"],
                            "scopes": ["secrets:read", "platform:read"],
                        }
                    }
                },
            }
        }
        validate_static_authz_data(data)

    def test_invalid_role_permission(self) -> None:
        data = {
            "authz": {
                "roles": {"Viewer": {"permissions": ["secrets:read"]}},
                "endpoints": {},
            }
        }
        with pytest.raises(InvalidPermissionFormatError):
            validate_static_authz_data(data)

    def test_valid_caller_kinds_pass(self) -> None:
        data = {
            "authz": {
                "roles": {},
                "endpoints": {
                    "/apis/x/v2/thing": {
                        "get": {"permissions": ["x.read"], "callers": ["principal", "service_principal"]},
                    }
                },
            }
        }
        validate_static_authz_data(data)

    def test_invalid_caller_kind_raises(self) -> None:
        # A hand-edited static-authz.yaml with an unknown caller kind is caught at load/build,
        # rather than failing silently in policy checks (the caller validator is now wired in).
        data = {
            "authz": {
                "roles": {},
                "endpoints": {
                    "/apis/x/v2/thing": {
                        "get": {"permissions": ["x.read"], "callers": ["anon"]},
                    }
                },
            }
        }
        with pytest.raises(ValueError, match="Invalid caller kind"):
            validate_static_authz_data(data)


def test_shipped_static_authz_yaml_passes_validation() -> None:
    """Regression: real static-authz.yaml must satisfy format checks."""
    path = Path(__file__).resolve().parents[4] / "services/core/auth/src/nhx/core/auth/assets/static-authz.yaml"
    assert path.is_file(), f"expected {path}"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    validate_static_authz_data(data)
