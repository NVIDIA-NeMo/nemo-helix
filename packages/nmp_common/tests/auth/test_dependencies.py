# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for auth dependencies."""

import json

import pytest
from nmp.common.auth import get_principal_auth_headers, principal_from_env
from nmp.common.auth.models import NMP_PRINCIPAL_ENVVAR
from nmp.common.config import AuthConfig, Configuration


@pytest.fixture
def clean_env(monkeypatch):
    """Ensure principal env var is cleared for tests (auto-restored after)."""
    monkeypatch.delenv(NMP_PRINCIPAL_ENVVAR, raising=False)


def test_principal_from_env_parses_json(clean_env, monkeypatch):
    """Test that principal_from_env parses principal from JSON env var."""
    monkeypatch.setenv(
        NMP_PRINCIPAL_ENVVAR,
        json.dumps(
            {
                "id": "job-creator@example.com",
                "email": "job-creator@example.com",
                "groups": ["engineering", "platform-team"],
            }
        ),
    )

    principal = principal_from_env()
    assert principal is not None
    headers = principal.get_headers()

    assert headers["X-NMP-Principal-Id"] == "job-creator@example.com"
    assert headers["X-NMP-Principal-Email"] == "job-creator@example.com"
    assert headers["X-NMP-Principal-Groups"] == "engineering,platform-team"


def test_principal_from_env_handles_id_only(clean_env, monkeypatch):
    """Test that principal_from_env works with only principal ID set."""
    monkeypatch.setenv(
        NMP_PRINCIPAL_ENVVAR,
        json.dumps(
            {
                "id": "service-account",
            }
        ),
    )

    principal = principal_from_env()
    assert principal is not None
    headers = principal.get_headers()

    assert headers["X-NMP-Principal-Id"] == "service-account"
    assert "X-NMP-Principal-Email" not in headers
    assert "X-NMP-Principal-Groups" not in headers


def test_principal_from_env_returns_none_when_not_set(clean_env):
    """Test that principal_from_env returns None when env var not set."""
    # No env vars set
    principal = principal_from_env()
    assert principal is None


def test_principal_from_env_does_not_modify_context(clean_env, monkeypatch):
    """Test that principal_from_env does not modify auth context var."""
    monkeypatch.setenv(
        NMP_PRINCIPAL_ENVVAR,
        json.dumps(
            {
                "id": "test-user",
            }
        ),
    )

    principal = principal_from_env()
    assert principal is not None
    headers = principal.get_headers()
    assert headers["X-NMP-Principal-Id"] == "test-user"

    # The headers set in the context are not changed
    assert get_principal_auth_headers() == {}


def test_principal_from_env_handles_empty_groups(clean_env, monkeypatch):
    """Test that empty groups list results in no groups header."""
    monkeypatch.setenv(
        NMP_PRINCIPAL_ENVVAR,
        json.dumps(
            {
                "id": "user",
                "groups": [],
            }
        ),
    )

    principal = principal_from_env()
    assert principal is not None
    headers = principal.get_headers()

    assert headers["X-NMP-Principal-Id"] == "user"
    assert "X-NMP-Principal-Groups" not in headers


def test_principal_from_env_invalid_json_raises(clean_env, monkeypatch):
    """Test that invalid JSON raises ValueError."""
    monkeypatch.setenv(NMP_PRINCIPAL_ENVVAR, "not-valid-json")

    with pytest.raises(ValueError, match="Invalid JSON"):
        principal_from_env()


def test_principal_from_env_missing_id_returns_none(clean_env, monkeypatch):
    """Test that JSON without id returns None."""
    monkeypatch.setenv(
        NMP_PRINCIPAL_ENVVAR,
        json.dumps(
            {
                "email": "user@example.com",
            }
        ),
    )

    principal = principal_from_env()
    assert principal is None


@pytest.fixture
def auth_enabled():
    """Enable auth for the duration of the test."""
    original = Configuration._overrides.get(AuthConfig)
    Configuration.set_override(AuthConfig(enabled=True))
    yield
    # Restore original (or clear override)
    if original is not None:
        Configuration._overrides[AuthConfig] = original
    else:
        Configuration._overrides.pop(AuthConfig, None)


def test_get_platform_sdk_includes_auth_headers_from_env(clean_env, monkeypatch, auth_enabled):
    """Test that get_platform_sdk() includes auth headers from principal_from_env() context.

    This verifies the full chain: NMP_PRINCIPAL env var → principal_from_env() →
    get_principal_auth_headers() → get_platform_sdk() default_headers.

    This is critical for job tasks - when principal_from_env() is active, SDK calls
    must include the job creator's auth headers.
    """
    from nmp.common.sdk_factory import get_platform_sdk

    monkeypatch.setenv(
        NMP_PRINCIPAL_ENVVAR,
        json.dumps(
            {
                "id": "task-user@example.com",
                "email": "task-user@example.com",
                "groups": ["data-science", "ml-ops"],
            }
        ),
    )

    sdk = get_platform_sdk()

    # Verify the SDK has auth headers set
    default_headers = sdk.default_headers
    assert default_headers is not None, "SDK should have default headers set"
    assert default_headers.get("X-NMP-Principal-Id") == "task-user@example.com"
    assert default_headers.get("X-NMP-Principal-Email") == "task-user@example.com"
    assert default_headers.get("X-NMP-Principal-Groups") == "data-science,ml-ops"


def test_get_platform_sdk_no_headers_without_principal(clean_env, auth_enabled):
    """Test that get_platform_sdk() has no auth headers when NMP_PRINCIPAL is not set."""
    from nmp.common.sdk_factory import get_platform_sdk

    # No NMP_PRINCIPAL env var - should have no auth headers
    sdk = get_platform_sdk()

    default_headers = sdk.default_headers or {}
    assert "X-NMP-Principal-Id" not in default_headers
    assert "X-NMP-Principal-Email" not in default_headers
    assert "X-NMP-Principal-Groups" not in default_headers


class TestDependencyProviderEntityClient:
    """Tests for DependencyProvider.get_entity_client() on-behalf-of behavior."""

    def test_entity_client_uses_service_principal(self):
        """Entity client should authenticate as service:{name}, not the user."""
        from unittest.mock import patch

        from nmp.common.auth import auth_client_context
        from nmp.common.auth.client import AuthClient
        from nmp.common.auth.models import Principal
        from nmp.common.config import AuthConfig
        from nmp.common.service.base import DependencyProvider

        dp = DependencyProvider()

        user_principal = Principal(id="user@example.com", email="user@example.com")
        auth_client = AuthClient(principal=user_principal, config=AuthConfig(), http_client=None)
        token = auth_client_context.set(auth_client)

        try:
            with patch.object(dp, "get_sdk_client") as mock_sdk:
                mock_base_sdk = mock_sdk.return_value

                dp._get_entity_sdk_on_behalf_of()

                mock_base_sdk.with_options.assert_called_once()
                call_kwargs = mock_base_sdk.with_options.call_args
                headers = call_kwargs.kwargs.get("set_default_headers") or call_kwargs[1].get("set_default_headers")

                assert headers["X-NMP-Principal-Id"] == "service:platform"
                assert headers["X-NMP-Principal-On-Behalf-Of"] == "user@example.com"
        finally:
            auth_client_context.reset(token)

    def test_entity_client_no_on_behalf_of_without_user_context(self):
        """Without user context, on-behalf-of should not be set."""
        from unittest.mock import patch

        from nmp.common.service.base import DependencyProvider

        dp = DependencyProvider()

        with patch.object(dp, "get_sdk_client") as mock_sdk:
            mock_base_sdk = mock_sdk.return_value

            dp._get_entity_sdk_on_behalf_of()

            call_kwargs = mock_base_sdk.with_options.call_args
            headers = call_kwargs.kwargs.get("set_default_headers") or call_kwargs[1].get("set_default_headers")

            assert headers["X-NMP-Principal-Id"] == "service:platform"
            assert "X-NMP-Principal-On-Behalf-Of" not in headers


class TestPrincipalOtlpHeaders:
    """Tests for Principal.get_otlp_headers_value()."""

    def test_otlp_headers_with_all_fields(self):
        """Test OTLP headers format includes all principal fields."""
        from nmp.common.auth.models import Principal

        principal = Principal(
            id="user@example.com",
            email="user@example.com",
            groups=["team-a", "team-b"],
        )

        headers = principal.get_otlp_headers_value()

        # Values should be URL-encoded, commas in groups become %2C
        assert "X-NMP-Principal-Id=user%40example.com" in headers
        assert "X-NMP-Principal-Email=user%40example.com" in headers
        assert "X-NMP-Principal-Groups=team-a%2Cteam-b" in headers

    def test_otlp_headers_id_only(self):
        """Test OTLP headers with only ID (no email, no groups)."""
        from nmp.common.auth.models import Principal

        principal = Principal(id="service-account")

        headers = principal.get_otlp_headers_value()

        assert headers == "X-NMP-Principal-Id=service-account"
        assert "Email" not in headers
        assert "Groups" not in headers

    def test_otlp_headers_special_characters_encoded(self):
        """Test that special characters are URL-encoded in OTLP headers."""
        from nmp.common.auth.models import Principal

        principal = Principal(
            id="user+special=chars@example.com",
            email="user+special=chars@example.com",
            groups=["group,with,commas", "group=with=equals"],
        )

        headers = principal.get_otlp_headers_value()

        # @ becomes %40, + becomes %2B, = becomes %3D, , becomes %2C
        assert "user%2Bspecial%3Dchars%40example.com" in headers
        assert "group%2Cwith%2Ccommas%2Cgroup%3Dwith%3Dequals" in headers


class TestBuildServicePrincipalHeadersDelegation:
    """build_service_principal_headers must forward the acting user's id, email, and groups."""

    def test_propagates_effective_user_claims(self):
        from nmp.common.auth import auth_client_context, build_service_principal_headers
        from nmp.common.auth.client import AuthClient
        from nmp.common.auth.models import Principal
        from nmp.common.config import AuthConfig

        user = Principal(
            id="user@example.com",
            email="user@example.com",
            groups=["team-a", "team-b"],
        )
        token = auth_client_context.set(AuthClient(principal=user, config=AuthConfig()))
        try:
            h = build_service_principal_headers("guardrails")
            assert h["X-NMP-Principal-Id"] == "service:guardrails"
            assert h["X-NMP-Principal-On-Behalf-Of"] == "user@example.com"
            assert h["X-NMP-Principal-On-Behalf-Of-Email"] == "user@example.com"
            assert h["X-NMP-Principal-On-Behalf-Of-Groups"] == "team-a,team-b"
        finally:
            auth_client_context.reset(token)

    def test_no_on_behalf_headers_when_only_service_context(self):
        from nmp.common.auth import auth_client_context, build_service_principal_headers
        from nmp.common.auth.client import AuthClient
        from nmp.common.auth.models import Principal
        from nmp.common.config import AuthConfig

        svc = Principal(id="service:batch", email=None, groups=[])
        token = auth_client_context.set(AuthClient(principal=svc, config=AuthConfig()))
        try:
            h = build_service_principal_headers("jobs")
            assert h == {"X-NMP-Principal-Id": "service:jobs"}
        finally:
            auth_client_context.reset(token)

    def test_id_only_when_no_auth_context(self):
        from nmp.common.auth import build_service_principal_headers

        assert build_service_principal_headers("x") == {"X-NMP-Principal-Id": "service:x"}


class TestGetPrincipalAuthHeadersDelegation:
    def test_includes_on_behalf_of_groups_and_email_on_principal(self):
        from nmp.common.auth import auth_client_context, get_principal_auth_headers
        from nmp.common.auth.client import AuthClient
        from nmp.common.auth.models import Principal
        from nmp.common.config import AuthConfig

        p = Principal(
            id="service:jobs",
            on_behalf_of="creator@example.com",
            on_behalf_of_email="creator@example.com",
            on_behalf_of_groups=["data-science"],
        )
        token = auth_client_context.set(AuthClient(principal=p, config=AuthConfig()))
        try:
            h = get_principal_auth_headers()
            assert h["X-NMP-Principal-On-Behalf-Of-Groups"] == "data-science"
            assert h["X-NMP-Principal-On-Behalf-Of-Email"] == "creator@example.com"
        finally:
            auth_client_context.reset(token)


def test_principal_from_env_serializes_on_behalf_of_headers(clean_env, monkeypatch):
    """Job principal JSON with delegation rehydrates to headers downstream services expect."""
    monkeypatch.setenv(
        NMP_PRINCIPAL_ENVVAR,
        json.dumps(
            {
                "id": "service:customizer",
                "on_behalf_of": "creator@example.com",
                "on_behalf_of_email": "creator@example.com",
                "on_behalf_of_groups": ["ml-team"],
            }
        ),
    )
    from nmp.common.auth import principal_from_env

    principal = principal_from_env()
    assert principal is not None
    headers = principal.get_headers()
    assert headers["X-NMP-Principal-On-Behalf-Of-Groups"] == "ml-team"
    assert headers["X-NMP-Principal-On-Behalf-Of-Email"] == "creator@example.com"
