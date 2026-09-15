# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for auth models."""

import json
import os

import pytest
from nmp.common.auth.exceptions import InvalidPrincipalHeader
from nmp.common.auth.models import (
    MAX_EMAIL_LENGTH,
    MAX_GROUP_LENGTH,
    MAX_GROUPS_COUNT,
    MAX_PRINCIPAL_ID_LENGTH,
    NMP_PRINCIPAL_ENVVAR,
    AuthContext,
    Principal,
)


class TestPrincipalFromHeaders:
    """Tests for Principal.from_headers() method."""

    def test_from_nmp_headers_basic(self):
        """Test creating Principal from basic NeMo Platform headers."""
        headers = {
            "x-nmp-principal-id": "user@example.com",
        }

        principal = Principal.from_headers(headers)

        assert principal is not None
        assert principal.id == "user@example.com"
        assert principal.email is None
        assert principal.groups == []

    def test_from_nmp_headers_full(self):
        """Test creating Principal from full NeMo Platform headers."""
        headers = {
            "x-nmp-principal-id": "user@example.com",
            "x-nmp-principal-email": "user@example.com",
            "x-nmp-principal-groups": "admin, users, developers",
            "x-nmp-principal-on-behalf-of": "other@example.com",
        }

        principal = Principal.from_headers(headers)

        assert principal is not None
        assert principal.id == "user@example.com"
        assert principal.email == "user@example.com"
        assert principal.groups == ["admin", "users", "developers"]
        assert principal.on_behalf_of == "other@example.com"
        # On-behalf-of header present: empty obo groups/email headers parse to [] / None
        assert principal.on_behalf_of_groups == []
        assert principal.on_behalf_of_email is None

    def test_from_headers_account_context_roundtrip(self):
        headers = {
            "x-nmp-principal-id": "service:worker",
            "x-nmp-actor-account-id": "account-service",
            "x-nmp-actor-aliases": "service:worker,service:worker,legacy-worker",
            "x-nmp-principal-on-behalf-of": "user@example.com",
            "x-nmp-subject-account-id": "account-user",
            "x-nmp-subject-aliases": "legacy-user,user@example.com",
        }

        principal = Principal.from_headers(headers)

        assert principal is not None
        assert principal.account_id == "account-service"
        assert principal.authz_aliases == ["service:worker", "legacy-worker"]
        assert principal.caller_kind == "service_principal"
        assert principal.on_behalf_of_account_id == "account-user"
        assert principal.on_behalf_of_authz_aliases == ["legacy-user", "user@example.com"]
        assert Principal.from_headers({k.lower(): v for k, v in principal.get_headers().items()}) == principal

    def test_from_headers_ignores_caller_kind_header(self):
        headers = {
            "x-nmp-principal-id": "user@example.com",
            "x-nmp-caller-kind": "service_principal",
        }

        principal = Principal.from_headers(headers)

        assert principal is not None
        assert principal.caller_kind == "principal"
        assert "X-NMP-Caller-Kind" not in principal.get_headers()

    def test_from_headers_rejects_wildcard_alias(self):
        headers = {
            "x-nmp-principal-id": "user@example.com",
            "x-nmp-actor-aliases": "team-a,*",
        }

        with pytest.raises(InvalidPrincipalHeader, match="wildcard"):
            Principal.from_headers(headers)

    def test_from_headers_rejects_delegated_account_without_delegation(self):
        headers = {
            "x-nmp-principal-id": "user@example.com",
            "x-nmp-subject-account-id": "account-user",
        }

        with pytest.raises(InvalidPrincipalHeader, match="On-Behalf-Of is required"):
            Principal.from_headers(headers)

    def test_from_nmp_headers_with_on_behalf_of_groups_and_email(self):
        """On-behalf-of groups and email are parsed when the on-behalf-of header is present."""
        headers = {
            "x-nmp-principal-id": "service:evaluator",
            "x-nmp-principal-on-behalf-of": "user@example.com",
            "x-nmp-principal-on-behalf-of-groups": "team-a, team-b",
            "x-nmp-principal-on-behalf-of-email": "user@example.com",
        }
        principal = Principal.from_headers(headers)
        assert principal is not None
        assert principal.id == "service:evaluator"
        assert principal.on_behalf_of == "user@example.com"
        assert principal.on_behalf_of_groups == ["team-a", "team-b"]
        assert principal.on_behalf_of_email == "user@example.com"

    def test_from_nmp_headers_empty_groups(self):
        """Test that empty groups header results in empty list."""
        headers = {
            "x-nmp-principal-id": "user@example.com",
            "x-nmp-principal-groups": "",
        }

        principal = Principal.from_headers(headers)

        assert principal is not None
        assert principal.groups == []

    def test_from_nmp_headers_strips_whitespace(self):
        """Test that whitespace is stripped from values."""
        headers = {
            "x-nmp-principal-id": "  user@example.com  ",
            "x-nmp-principal-groups": " admin ,  users , developers ",
        }

        principal = Principal.from_headers(headers)

        assert principal is not None
        assert principal.id == "user@example.com"
        assert principal.groups == ["admin", "users", "developers"]

    def test_from_headers_no_principal_id(self):
        """Test that None is returned when no principal ID is present."""
        headers = {
            "x-nmp-principal-email": "user@example.com",
        }

        principal = Principal.from_headers(headers)

        assert principal is None

    def test_from_headers_empty_principal_id(self):
        """Test that None is returned for empty principal ID."""
        headers = {
            "x-nmp-principal-id": "",
        }

        principal = Principal.from_headers(headers)

        assert principal is None

    def test_from_headers_whitespace_only_principal_id(self):
        """Test that None is returned for whitespace-only principal ID."""
        headers = {
            "x-nmp-principal-id": "   ",
        }

        principal = Principal.from_headers(headers)

        assert principal is None

    def test_from_headers_empty_dict(self):
        """Test that None is returned for empty headers."""
        principal = Principal.from_headers({})

        assert principal is None

    def test_from_headers_case_insensitive_keys(self):
        """HTTP header keys are case-insensitive."""
        headers = {
            "X-NMP-Principal-Id": "user@example.com",
            "X-NMP-Principal-Email": "user@example.com",
        }

        principal = Principal.from_headers(headers)

        assert principal is not None
        assert principal.id == "user@example.com"
        assert principal.email == "user@example.com"


class TestPrincipalEffectiveIdentity:
    """Tests for effective_id / effective_email / effective_groups / effective_principal."""

    def test_effective_id_prefers_on_behalf_of(self):
        p = Principal(id="service:x", on_behalf_of="human@example.com")
        assert p.effective_id == "human@example.com"

    def test_effective_email_is_delegate_only_when_delegating(self):
        """When delegating, service email must not mask a missing delegate email."""
        p = Principal(
            id="service:x",
            email="noreply@svc.example.com",
            on_behalf_of="human@example.com",
            on_behalf_of_email=None,
        )
        assert p.effective_email is None

    def test_effective_email_uses_delegate_when_set(self):
        p = Principal(
            id="service:x",
            email="noreply@svc.example.com",
            on_behalf_of="human@example.com",
            on_behalf_of_email="human@example.com",
        )
        assert p.effective_email == "human@example.com"

    def test_effective_groups_empty_delegate_list_does_not_fallback_to_service(self):
        p = Principal(
            id="service:x",
            groups=["platform-admin"],
            on_behalf_of="human@example.com",
            on_behalf_of_groups=[],
        )
        assert p.effective_groups == []

    def test_effective_groups_uses_delegate_when_set(self):
        p = Principal(
            id="service:x",
            groups=["platform-admin"],
            on_behalf_of="human@example.com",
            on_behalf_of_groups=["workspace-editors"],
        )
        assert p.effective_groups == ["workspace-editors"]

    def test_effective_principal_matches_effective_fields(self):
        p = Principal(
            id="service:x",
            email="svc@example.com",
            groups=["svc-group"],
            on_behalf_of="human@example.com",
            on_behalf_of_email="human@example.com",
            on_behalf_of_groups=["human-group"],
        )
        eff = p.effective_principal
        assert eff.id == "human@example.com"
        assert eff.email == "human@example.com"
        assert eff.groups == ["human-group"]
        assert eff.on_behalf_of is None

    @pytest.mark.parametrize(
        ("principal_id", "expected"),
        [
            ("user@example.com", False),
            ("service:auth", True),
            ("service-account:otel-collector", True),
        ],
    )
    def test_is_service_identity(self, principal_id: str, expected: bool):
        assert Principal(id=principal_id).is_service_identity() is expected


class TestPrincipalEffectiveId:
    """Tests for Principal.effective_id property."""

    def test_returns_id_when_no_on_behalf_of(self):
        principal = Principal(id="user@example.com")
        assert principal.effective_id == "user@example.com"

    def test_returns_on_behalf_of_when_set(self):
        principal = Principal(id="service:evaluator", on_behalf_of="user@example.com")
        assert principal.effective_id == "user@example.com"

    def test_returns_id_when_on_behalf_of_is_none(self):
        principal = Principal(id="service:auth", on_behalf_of=None)
        assert principal.effective_id == "service:auth"

    def test_returns_empty_string_when_both_empty(self):
        principal = Principal(id="", on_behalf_of=None)
        assert principal.effective_id == ""


class TestPrincipalGetHeaders:
    """Tests for Principal.get_headers() method."""

    def test_get_headers_basic(self):
        """Test getting headers from a basic Principal."""
        principal = Principal(id="user@example.com")

        headers = principal.get_headers()

        assert headers == {
            "X-NMP-Principal-Id": "user@example.com",
        }

    def test_get_headers_full(self):
        """Test getting headers from a full Principal."""
        principal = Principal(
            id="user@example.com",
            email="user@example.com",
            groups=["admin", "users"],
            on_behalf_of="other@example.com",
        )

        headers = principal.get_headers()

        assert headers == {
            "X-NMP-Principal-Id": "user@example.com",
            "X-NMP-Principal-Email": "user@example.com",
            "X-NMP-Principal-Groups": "admin,users",
            "X-NMP-Principal-On-Behalf-Of": "other@example.com",
        }

    def test_get_headers_includes_account_context(self):
        principal = Principal(
            id="service:worker",
            account_id="account-123",
            authz_aliases=["service:worker"],
            on_behalf_of="user@example.com",
            on_behalf_of_account_id="account-456",
            on_behalf_of_authz_aliases=["legacy-user"],
        )

        headers = principal.get_headers()

        assert headers["X-NMP-Actor-Account-Id"] == "account-123"
        assert headers["X-NMP-Actor-Aliases"] == "service:worker"
        assert headers["X-NMP-Subject-Account-Id"] == "account-456"
        assert headers["X-NMP-Subject-Aliases"] == "legacy-user"

    def test_get_headers_includes_on_behalf_of_groups_and_email(self):
        principal = Principal(
            id="service:worker",
            on_behalf_of="user@example.com",
            on_behalf_of_groups=["g1", "g2"],
            on_behalf_of_email="user@example.com",
        )
        headers = principal.get_headers()
        assert headers["X-NMP-Principal-Id"] == "service:worker"
        assert headers["X-NMP-Principal-On-Behalf-Of"] == "user@example.com"
        assert headers["X-NMP-Principal-On-Behalf-Of-Groups"] == "g1,g2"
        assert headers["X-NMP-Principal-On-Behalf-Of-Email"] == "user@example.com"

    def test_get_headers_empty_groups(self):
        """Test that empty groups are not included in headers."""
        principal = Principal(id="user@example.com", groups=[])

        headers = principal.get_headers()

        assert "X-NMP-Principal-Groups" not in headers


class TestPrincipalToOnBehalfOf:
    """Tests for Principal.to_on_behalf_of() method."""

    def test_to_on_behalf_of_success(self):
        """Delegated principal carries on-behalf-of id, email, and groups only."""
        principal = Principal(
            id="service:evaluator",
            email="svc@example.com",
            groups=["service-metadata"],
            on_behalf_of="user@example.com",
            on_behalf_of_email="user@example.com",
            on_behalf_of_groups=["team-a"],
        )

        delegated = principal.to_on_behalf_of()

        assert delegated.id == "user@example.com"
        assert delegated.email == "user@example.com"
        assert delegated.groups == ["team-a"]
        assert delegated.on_behalf_of is None

    def test_to_on_behalf_of_without_delegation(self):
        """Test that error is raised when on_behalf_of is not set."""
        principal = Principal(id="user@example.com")

        with pytest.raises(ValueError, match="Cannot create on-behalf-of principal"):
            principal.to_on_behalf_of()


class TestPrincipalFromEnvVar:
    """Tests for Principal.from_env_var() method."""

    def test_from_env_var_success(self):
        """Test creating Principal from environment variable."""
        os.environ[NMP_PRINCIPAL_ENVVAR] = '{"id": "user@example.com", "email": "user@example.com"}'

        try:
            principal = Principal.from_env_var()

            assert principal is not None
            assert principal.id == "user@example.com"
            assert principal.email == "user@example.com"
        finally:
            del os.environ[NMP_PRINCIPAL_ENVVAR]

    def test_from_env_var_not_set(self):
        """Test that None is returned when env var is not set."""
        # Ensure env var is not set
        os.environ.pop(NMP_PRINCIPAL_ENVVAR, None)

        principal = Principal.from_env_var()

        assert principal is None

    def test_from_env_var_invalid_json(self):
        """Test that ValueError is raised for invalid JSON."""
        os.environ[NMP_PRINCIPAL_ENVVAR] = "not valid json"

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                Principal.from_env_var()
        finally:
            del os.environ[NMP_PRINCIPAL_ENVVAR]

    def test_from_env_var_empty_id(self):
        """Test that None is returned for empty ID."""
        os.environ[NMP_PRINCIPAL_ENVVAR] = '{"id": "", "email": "user@example.com"}'

        try:
            principal = Principal.from_env_var()
            assert principal is None
        finally:
            del os.environ[NMP_PRINCIPAL_ENVVAR]

    def test_from_env_var_custom_name(self):
        """Test using a custom environment variable name."""
        custom_env_var = "CUSTOM_PRINCIPAL"
        os.environ[custom_env_var] = '{"id": "custom-user"}'

        try:
            principal = Principal.from_env_var(custom_env_var)

            assert principal is not None
            assert principal.id == "custom-user"
        finally:
            del os.environ[custom_env_var]

    def test_from_env_var_includes_on_behalf_of_claims(self):
        payload = {
            "id": "service:jobs",
            "on_behalf_of": "creator@example.com",
            "on_behalf_of_email": "creator@example.com",
            "on_behalf_of_groups": ["data-science"],
        }
        os.environ[NMP_PRINCIPAL_ENVVAR] = json.dumps(payload)
        try:
            principal = Principal.from_env_var()
            assert principal is not None
            assert principal.id == "service:jobs"
            assert principal.on_behalf_of == "creator@example.com"
            assert principal.on_behalf_of_email == "creator@example.com"
            assert principal.on_behalf_of_groups == ["data-science"]
            assert principal.effective_groups == ["data-science"]
        finally:
            del os.environ[NMP_PRINCIPAL_ENVVAR]


class TestAuthContextPrincipalRoundTrip:
    """AuthContext should preserve delegation fields for persistence and SDK rehydration."""

    def test_from_principal_and_to_principal_round_trip(self):
        p = Principal(
            id="service:x",
            email="svc@example.com",
            groups=["g-svc"],
            on_behalf_of="user@example.com",
            on_behalf_of_email="user@example.com",
            on_behalf_of_groups=["g-user"],
        )
        ctx = AuthContext.from_principal(p)
        assert ctx.principal_id == "service:x"
        assert ctx.principal_email == "svc@example.com"
        assert ctx.principal_groups == ["g-svc"]
        assert ctx.principal_on_behalf_of == "user@example.com"
        assert ctx.principal_on_behalf_of_email == "user@example.com"
        assert ctx.principal_on_behalf_of_groups == ["g-user"]

        restored = ctx.to_principal()
        assert restored.model_dump() == p.model_dump()


class TestPrincipalGetOtlpHeadersValue:
    """Tests for Principal.get_otlp_headers_value() method."""

    def test_otlp_headers_basic(self):
        """Test OTLP headers with basic principal."""
        principal = Principal(id="user@example.com")

        value = principal.get_otlp_headers_value()

        assert value == "X-NMP-Principal-Id=user%40example.com"

    def test_otlp_headers_full(self):
        """Test OTLP headers with full principal."""
        principal = Principal(
            id="user@example.com",
            email="user@example.com",
            groups=["admin", "users"],
            on_behalf_of="other@example.com",
        )

        value = principal.get_otlp_headers_value()

        # Values should be URL-encoded
        assert "X-NMP-Principal-Id=user%40example.com" in value
        assert "X-NMP-Principal-Email=user%40example.com" in value
        assert "X-NMP-Principal-Groups=admin%2Cusers" in value
        assert "X-NMP-Principal-On-Behalf-Of=other%40example.com" in value

    def test_otlp_headers_includes_on_behalf_of_groups_and_email(self):
        principal = Principal(
            id="service:worker",
            on_behalf_of="user@example.com",
            on_behalf_of_groups=["a", "b"],
            on_behalf_of_email="user@example.com",
        )
        value = principal.get_otlp_headers_value()
        assert "X-NMP-Principal-On-Behalf-Of-Groups=a%2Cb" in value
        assert "X-NMP-Principal-On-Behalf-Of-Email=user%40example.com" in value

    def test_otlp_headers_special_characters(self):
        """Test OTLP headers with special characters are properly encoded."""
        principal = Principal(
            id="user+test@example.com",
            groups=["group,with,commas", "group with spaces"],
        )

        value = principal.get_otlp_headers_value()

        # Special characters should be encoded
        assert "%2B" in value  # + encoded
        assert "%2C" in value  # , encoded
        assert "%20" in value  # space encoded


class TestPrincipalHeaderValidation:
    """Tests for input validation in Principal.from_headers()."""

    # --- Principal ID validation ---

    def test_valid_email_principal_id(self):
        headers = {"x-nmp-principal-id": "user@example.com"}
        principal = Principal.from_headers(headers)
        assert principal is not None
        assert principal.id == "user@example.com"

    def test_valid_service_principal_id(self):
        headers = {"x-nmp-principal-id": "service:my-service"}
        principal = Principal.from_headers(headers)
        assert principal is not None
        assert principal.id == "service:my-service"

    def test_valid_oidc_subject_principal_id(self):
        headers = {"x-nmp-principal-id": "auth0/abc123-def456"}
        principal = Principal.from_headers(headers)
        assert principal is not None
        assert principal.id == "auth0/abc123-def456"

    def test_principal_id_at_max_length(self):
        value = "a" * MAX_PRINCIPAL_ID_LENGTH
        headers = {"x-nmp-principal-id": value}
        principal = Principal.from_headers(headers)
        assert principal is not None
        assert principal.id == value

    def test_principal_id_exceeds_max_length(self):
        value = "a" * (MAX_PRINCIPAL_ID_LENGTH + 1)
        headers = {"x-nmp-principal-id": value}
        with pytest.raises(InvalidPrincipalHeader, match="exceeds maximum length"):
            Principal.from_headers(headers)

    def test_principal_id_control_characters(self):
        headers = {"x-nmp-principal-id": "user\x00@example.com"}
        with pytest.raises(InvalidPrincipalHeader, match="invalid characters"):
            Principal.from_headers(headers)

    def test_principal_id_newline(self):
        headers = {"x-nmp-principal-id": "user\n@example.com"}
        with pytest.raises(InvalidPrincipalHeader, match="invalid characters"):
            Principal.from_headers(headers)

    def test_principal_id_path_traversal_with_backslash(self):
        headers = {"x-nmp-principal-id": "..\\..\\etc\\passwd"}
        with pytest.raises(InvalidPrincipalHeader, match="invalid characters"):
            Principal.from_headers(headers)

    def test_principal_id_forward_slash_allowed_for_oidc(self):
        """Forward slashes are allowed because OIDC subjects can contain them."""
        headers = {"x-nmp-principal-id": "../../etc/passwd"}
        principal = Principal.from_headers(headers)
        assert principal is not None

    def test_principal_id_spaces(self):
        headers = {"x-nmp-principal-id": "user name"}
        with pytest.raises(InvalidPrincipalHeader, match="invalid characters"):
            Principal.from_headers(headers)

    def test_principal_id_sql_injection(self):
        headers = {"x-nmp-principal-id": "'; DROP TABLE users;--"}
        with pytest.raises(InvalidPrincipalHeader, match="invalid characters"):
            Principal.from_headers(headers)

    def test_principal_id_unicode(self):
        headers = {"x-nmp-principal-id": "user\u00e9@example.com"}
        with pytest.raises(InvalidPrincipalHeader, match="invalid characters"):
            Principal.from_headers(headers)

    # --- Email validation ---

    def test_valid_email(self):
        headers = {
            "x-nmp-principal-id": "user@example.com",
            "x-nmp-principal-email": "user@example.com",
        }
        principal = Principal.from_headers(headers)
        assert principal is not None
        assert principal.email == "user@example.com"

    def test_email_plus_addressing(self):
        headers = {
            "x-nmp-principal-id": "user@example.com",
            "x-nmp-principal-email": "user+tag@example.com",
        }
        principal = Principal.from_headers(headers)
        assert principal is not None
        assert principal.email == "user+tag@example.com"

    def test_email_exceeds_max_length(self):
        long_email = "a" * (MAX_EMAIL_LENGTH - 10) + "@example.com"
        headers = {
            "x-nmp-principal-id": "user@example.com",
            "x-nmp-principal-email": long_email,
        }
        with pytest.raises(InvalidPrincipalHeader, match="exceeds maximum length"):
            Principal.from_headers(headers)

    def test_email_missing_at(self):
        headers = {
            "x-nmp-principal-id": "user@example.com",
            "x-nmp-principal-email": "not-an-email",
        }
        with pytest.raises(InvalidPrincipalHeader, match="not a valid email"):
            Principal.from_headers(headers)

    def test_email_empty_local_part(self):
        headers = {
            "x-nmp-principal-id": "user@example.com",
            "x-nmp-principal-email": "@example.com",
        }
        with pytest.raises(InvalidPrincipalHeader, match="not a valid email"):
            Principal.from_headers(headers)

    def test_email_empty_domain(self):
        headers = {
            "x-nmp-principal-id": "user@example.com",
            "x-nmp-principal-email": "user@",
        }
        with pytest.raises(InvalidPrincipalHeader, match="not a valid email"):
            Principal.from_headers(headers)

    def test_email_whitespace_only_treated_as_none(self):
        headers = {
            "x-nmp-principal-id": "user@example.com",
            "x-nmp-principal-email": "   ",
        }
        principal = Principal.from_headers(headers)
        assert principal is not None
        assert principal.email is None

    # --- Groups validation ---

    def test_valid_groups(self):
        headers = {
            "x-nmp-principal-id": "user@example.com",
            "x-nmp-principal-groups": "admin,users,developers",
        }
        principal = Principal.from_headers(headers)
        assert principal is not None
        assert principal.groups == ["admin", "users", "developers"]

    def test_groups_exceeds_max_count(self):
        groups = ",".join(f"group{i}" for i in range(MAX_GROUPS_COUNT + 1))
        headers = {
            "x-nmp-principal-id": "user@example.com",
            "x-nmp-principal-groups": groups,
        }
        with pytest.raises(InvalidPrincipalHeader, match="exceeds maximum"):
            Principal.from_headers(headers)

    def test_groups_at_max_count(self):
        groups = ",".join(f"group{i}" for i in range(MAX_GROUPS_COUNT))
        headers = {
            "x-nmp-principal-id": "user@example.com",
            "x-nmp-principal-groups": groups,
        }
        principal = Principal.from_headers(headers)
        assert principal is not None
        assert len(principal.groups) == MAX_GROUPS_COUNT

    def test_group_exceeds_max_length(self):
        long_group = "g" * (MAX_GROUP_LENGTH + 1)
        headers = {
            "x-nmp-principal-id": "user@example.com",
            "x-nmp-principal-groups": long_group,
        }
        with pytest.raises(InvalidPrincipalHeader, match="exceeding"):
            Principal.from_headers(headers)

    def test_group_with_invalid_characters(self):
        headers = {
            "x-nmp-principal-id": "user@example.com",
            "x-nmp-principal-groups": "valid-group,bad group!",
        }
        with pytest.raises(InvalidPrincipalHeader, match="invalid characters"):
            Principal.from_headers(headers)

    def test_group_with_control_characters(self):
        headers = {
            "x-nmp-principal-id": "user@example.com",
            "x-nmp-principal-groups": "admin,group\x00name",
        }
        with pytest.raises(InvalidPrincipalHeader, match="invalid characters"):
            Principal.from_headers(headers)

    # --- On-behalf-of validation ---

    def test_valid_on_behalf_of(self):
        headers = {
            "x-nmp-principal-id": "admin@example.com",
            "x-nmp-principal-on-behalf-of": "user@example.com",
        }
        principal = Principal.from_headers(headers)
        assert principal is not None
        assert principal.on_behalf_of == "user@example.com"

    def test_on_behalf_of_exceeds_max_length(self):
        value = "a" * (MAX_PRINCIPAL_ID_LENGTH + 1)
        headers = {
            "x-nmp-principal-id": "admin@example.com",
            "x-nmp-principal-on-behalf-of": value,
        }
        with pytest.raises(InvalidPrincipalHeader, match="exceeds maximum length"):
            Principal.from_headers(headers)

    def test_on_behalf_of_invalid_characters(self):
        headers = {
            "x-nmp-principal-id": "admin@example.com",
            "x-nmp-principal-on-behalf-of": "user; DROP TABLE",
        }
        with pytest.raises(InvalidPrincipalHeader, match="invalid characters"):
            Principal.from_headers(headers)

    def test_on_behalf_of_whitespace_only_treated_as_none(self):
        headers = {
            "x-nmp-principal-id": "admin@example.com",
            "x-nmp-principal-on-behalf-of": "   ",
        }
        principal = Principal.from_headers(headers)
        assert principal is not None
        assert principal.on_behalf_of is None

    def test_blank_on_behalf_of_rejects_subject_account_headers(self):
        headers = {
            "x-nmp-principal-id": "admin@example.com",
            "x-nmp-principal-on-behalf-of": "   ",
            "x-nmp-subject-account-id": "account-user",
        }
        with pytest.raises(InvalidPrincipalHeader, match="On-Behalf-Of is required"):
            Principal.from_headers(headers)

    # --- Combined validation ---

    def test_all_headers_valid(self):
        headers = {
            "x-nmp-principal-id": "user@example.com",
            "x-nmp-principal-email": "user@example.com",
            "x-nmp-principal-groups": "admin,users",
            "x-nmp-principal-on-behalf-of": "service:backend",
        }
        principal = Principal.from_headers(headers)
        assert principal is not None
        assert principal.id == "user@example.com"
        assert principal.email == "user@example.com"
        assert principal.groups == ["admin", "users"]
        assert principal.on_behalf_of == "service:backend"

    def test_invalid_principal_id_short_circuits(self):
        """Invalid principal ID is caught before other headers are checked."""
        headers = {
            "x-nmp-principal-id": "bad\x00id",
            "x-nmp-principal-email": "also-invalid",
        }
        with pytest.raises(InvalidPrincipalHeader, match="X-NMP-Principal-Id"):
            Principal.from_headers(headers)
