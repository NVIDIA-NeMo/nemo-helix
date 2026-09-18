# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the AuthClient class."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from nemo_platform import NeMoPlatform
from nmp.common.auth.client import AuthClient
from nmp.common.auth.exceptions import InvalidPermissionFormatError, InvalidScopeFormatError
from nmp.common.auth.models import Principal
from nmp.common.config import AuthConfig
from nmp.common.sdk_factory import get_sdk_on_behalf_of


@pytest.fixture
def auth_config():
    """Create an AuthConfig instance for testing."""
    return AuthConfig(
        enabled=True,
        policy_decision_point_base_url="http://localhost:8181",
    )


@pytest.fixture
def auth_config_disabled():
    """Create an AuthConfig instance with auth disabled."""
    return AuthConfig(
        enabled=False,
        policy_decision_point_base_url="http://localhost:8181",
    )


@pytest.fixture
def principal():
    """Create a test principal."""
    return Principal(
        id="test-user@example.com",
        email="test-user@example.com",
        groups=["test-group"],
        on_behalf_of=None,
    )


@pytest.fixture
def principal_with_delegate():
    """Create a test principal with an on-behalf-of field."""
    return Principal(
        id="admin-user@example.com",
        email="admin-user@example.com",
        groups=["admin-group"],
        on_behalf_of="delegated-user@example.com",
    )


@pytest.fixture
def principal_service_delegating():
    """Service principal acting on behalf of a user with delegate-specific claims."""
    return Principal(
        id="service:evaluator",
        email="svc@example.com",
        groups=["platform-admin"],
        on_behalf_of="user@example.com",
        on_behalf_of_email="user@example.com",
        on_behalf_of_groups=["workspace-editors"],
    )


class TestHasPermissionsFormatValidation:
    """Runtime validation: invalid strings raise; valid permission/scope syntax proceeds to PDP."""

    @pytest.mark.asyncio
    async def test_has_permissions_accepts_dot_syntax_and_calls_pdp(self, auth_config, principal):
        """Valid permissions pass format checks and reach the has_permissions endpoint."""
        mock_http_client = httpx.AsyncClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"allowed": True}}
        mock_response.raise_for_status = MagicMock()
        with patch.object(mock_http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            auth_client = AuthClient(principal=principal, config=auth_config, http_client=mock_http_client)
            result = await auth_client.has_permissions("ws", ["secrets.read", "models.create"])
        assert result is True
        mock_post.assert_called_once()
        body = mock_post.call_args[1]["json"]["input"]
        assert body["permissions"] == ["secrets.read", "models.create"]

    @pytest.mark.asyncio
    async def test_authorize_request_accepts_nmp_scopes_and_calls_pdp(self, auth_config, principal):
        """Valid NeMo Platform-style scopes pass format checks and are sent to the allow endpoint."""
        mock_http_client = httpx.AsyncClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"allowed": True}}
        mock_response.raise_for_status = MagicMock()
        with patch.object(mock_http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            auth_client = AuthClient(principal=principal, config=auth_config, http_client=mock_http_client)
            out = await auth_client.authorize_request(
                "GET",
                "/apis/models/v2/workspaces/ws/models",
                scopes=["platform:read", "models:read"],
                http_client=mock_http_client,
            )
        assert out.allowed is True
        body = mock_post.call_args[1]["json"]["input"]
        assert body["scopes"] == ["platform:read", "models:read"]

    @pytest.mark.asyncio
    async def test_rejects_scope_syntax_in_permissions(self, auth_config, principal):
        auth_client = AuthClient(principal=principal, config=auth_config)
        with pytest.raises(InvalidPermissionFormatError, match="dots"):
            await auth_client.has_permissions("ws", ["secrets:read"])

    @pytest.mark.asyncio
    async def test_authorize_request_rejects_permission_like_scopes(self, auth_config, principal):
        auth_client = AuthClient(principal=principal, config=auth_config)
        with pytest.raises(InvalidScopeFormatError, match="permission syntax"):
            await auth_client.authorize_request("GET", "/x", scopes=["secrets.read"])


class TestHasPermissionsPdpPayloadWithDelegation:
    """has_permissions must send the acting user's email/groups (delegate), not the service's."""

    @pytest.mark.asyncio
    async def test_has_permissions_sends_delegate_claims_when_delegating(
        self, auth_config, principal_service_delegating
    ):
        mock_http_client = httpx.AsyncClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"allowed": True}}
        mock_response.raise_for_status = MagicMock()

        with patch.object(mock_http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            auth_client = AuthClient(
                principal=principal_service_delegating,
                config=auth_config,
                http_client=mock_http_client,
            )
            assert await auth_client.has_permissions("ws", ["models.read"]) is True
        body = mock_post.call_args[1]["json"]["input"]
        assert body["principal_id"] == "service:evaluator"
        assert body["on_behalf_of_principal_id"] == "user@example.com"
        assert body["principal_email"] == "user@example.com"
        assert body["principal_groups"] == ["workspace-editors"]

    @pytest.mark.asyncio
    async def test_has_permissions_service_permitted_delegate_not_sent_to_pdp(self, auth_config):
        """PDP input must reflect delegate claims only: service 'admin' groups must not mask delegate."""
        principal = Principal(
            id="service:worker",
            email="svc@example.com",
            groups=["platform-admin"],
            on_behalf_of="user@example.com",
            on_behalf_of_email="user@example.com",
            on_behalf_of_groups=["no-access-group"],
        )
        mock_http_client = httpx.AsyncClient()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        async def decide_post(url, **kwargs):
            inp = kwargs["json"]["input"]
            # Same list as on_behalf_of_groups above — not workspace-editors (other tests) or
            # the service's platform-admin groups.
            assert inp.get("principal_groups") == ["no-access-group"]
            mock_response.json.return_value = {"result": {"allowed": False}}
            return mock_response

        with patch.object(mock_http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = decide_post
            auth_client = AuthClient(principal=principal, config=auth_config, http_client=mock_http_client)
            assert await auth_client.has_permissions("ws", ["models.read"]) is False

    @pytest.mark.asyncio
    async def test_has_permissions_service_without_claims_delegate_permitted(self, auth_config):
        """Service with no email/groups: permission check uses delegate groups/email only."""
        principal = Principal(
            id="service:worker",
            on_behalf_of="user@example.com",
            on_behalf_of_email="user@example.com",
            on_behalf_of_groups=["workspace-editors"],
        )
        mock_http_client = httpx.AsyncClient()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        async def decide_post(url, **kwargs):
            inp = kwargs["json"]["input"]
            allowed = (
                inp["principal_id"] == "service:worker"
                and inp.get("on_behalf_of_principal_id") == "user@example.com"
                and inp.get("principal_groups") == ["workspace-editors"]
                and inp.get("principal_email") == "user@example.com"
            )
            mock_response.json.return_value = {"result": {"allowed": allowed}}
            return mock_response

        with patch.object(mock_http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = decide_post
            auth_client = AuthClient(principal=principal, config=auth_config, http_client=mock_http_client)
            assert await auth_client.has_permissions("ws", ["models.read"]) is True


class TestHasRole:
    @pytest.mark.asyncio
    async def test_has_role_sends_effective_delegate_claims(self, auth_config, principal_service_delegating):
        mock_http_client = httpx.AsyncClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"has_role": True}}
        mock_response.raise_for_status = MagicMock()

        with patch.object(mock_http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            auth_client = AuthClient(
                principal=principal_service_delegating,
                config=auth_config,
                http_client=mock_http_client,
            )
            assert await auth_client.has_role("system", "PlatformAdmin") is True

        body = mock_post.call_args.kwargs["json"]["input"]
        assert body == {
            "principal_id": "user@example.com",
            "principal_email": "user@example.com",
            "principal_groups": ["workspace-editors"],
            "on_behalf_of_principal_id": "user@example.com",
            "workspace": "system",
            "role": "PlatformAdmin",
        }

    @pytest.mark.asyncio
    async def test_has_role_maps_pdp_transport_failure_to_service_unavailable(self, auth_config, principal):
        mock_http_client = httpx.AsyncClient()
        request = httpx.Request("POST", "http://localhost:8181/v1/data/authz/has_role")

        with patch.object(
            mock_http_client,
            "post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("PDP unavailable", request=request),
        ):
            auth_client = AuthClient(principal=principal, config=auth_config, http_client=mock_http_client)
            with pytest.raises(HTTPException) as exc_info:
                await auth_client.has_role("system", "PlatformAdmin")

        assert exc_info.value.status_code == 503
        assert exc_info.value.detail == "Role check service unavailable"

    @pytest.mark.asyncio
    async def test_has_role_maps_invalid_pdp_response_to_internal_error(self, auth_config, principal):
        mock_http_client = httpx.AsyncClient()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.side_effect = ValueError("invalid JSON")

        with patch.object(mock_http_client, "post", new_callable=AsyncMock, return_value=mock_response):
            auth_client = AuthClient(principal=principal, config=auth_config, http_client=mock_http_client)
            with pytest.raises(HTTPException) as exc_info:
                await auth_client.has_role("system", "PlatformAdmin")

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Internal role check error"


class TestAuthorizeRequestPdpPayloadWithDelegation:
    @pytest.mark.asyncio
    async def test_authorize_request_sends_delegate_claims(self, auth_config, principal_service_delegating):
        mock_http_client = httpx.AsyncClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"allowed": True}}
        mock_response.raise_for_status = MagicMock()

        with patch.object(mock_http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            auth_client = AuthClient(
                principal=principal_service_delegating,
                config=auth_config,
                http_client=mock_http_client,
            )
            out = await auth_client.authorize_request("GET", "/v2/workspaces/ws/models", http_client=mock_http_client)
        assert out.allowed is True
        body = mock_post.call_args[1]["json"]["input"]
        assert body["principal_email"] == "user@example.com"
        assert body["principal_groups"] == ["workspace-editors"]
        assert body["on_behalf_of_principal_id"] == "user@example.com"


class TestWaitRole:
    @pytest.mark.asyncio
    async def test_wait_role_requires_pdp_url_before_creating_client(self, principal):
        auth_config = AuthConfig(enabled=True, policy_decision_point_base_url="")
        auth_client = AuthClient(principal=principal, config=auth_config)

        with patch.object(auth_client, "_new_pdp_http_client") as new_client:
            with pytest.raises(RuntimeError, match="Policy Decision Point URL not configured"):
                await auth_client.wait_role(
                    "user@example.com",
                    "test-workspace",
                    "Viewer",
                    timeout=0.01,
                    poll_interval=0.01,
                )

        new_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_wait_role_reuses_provided_http_client(self, auth_config, principal):
        mock_http_client = httpx.AsyncClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"has_role": True}}
        mock_response.raise_for_status = MagicMock()

        with (
            patch.object(mock_http_client, "post", new_callable=AsyncMock, return_value=mock_response),
            patch.object(mock_http_client, "aclose", new_callable=AsyncMock) as close,
        ):
            auth_client = AuthClient(principal=principal, config=auth_config)
            assert await auth_client.wait_role(
                "user@example.com",
                "test-workspace",
                "Viewer",
                timeout=0.01,
                poll_interval=0.01,
                http_client=mock_http_client,
            )

        close.assert_not_called()


class TestOnBehalfOfHasPermissions:
    """Tests for the on_behalf_of_has_permissions method."""

    @pytest.mark.asyncio
    async def test_auth_disabled_returns_true(self, auth_config_disabled, principal_with_delegate):
        """Test that when auth is disabled, method returns True."""
        auth_client = AuthClient(
            principal=principal_with_delegate,
            config=auth_config_disabled,
        )

        result = await auth_client.on_behalf_of_has_permissions(
            workspace_id="test-workspace",
            permissions=["secrets.read"],
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_on_behalf_of_with_permissions_returns_true(self, auth_config, principal_with_delegate):
        """Test that when delegated user has permissions, method returns True."""
        # Create a real AsyncClient but mock its post method
        mock_http_client = httpx.AsyncClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"allowed": True}}
        mock_response.raise_for_status = MagicMock()

        with patch.object(mock_http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            auth_client = AuthClient(
                principal=principal_with_delegate,
                config=auth_config,
                http_client=mock_http_client,
            )

            result = await auth_client.on_behalf_of_has_permissions(
                workspace_id="test-workspace",
                permissions=["secrets.read"],
            )

            assert result is True

            # Verify the mock was called with the delegated principal
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            request_json = call_args[1]["json"]

            # The delegated principal ID should be used
            assert request_json["input"]["principal_id"] == "delegated-user@example.com"
            assert request_json["input"]["workspace"] == "test-workspace"
            assert request_json["input"]["permissions"] == ["secrets.read"]

    @pytest.mark.asyncio
    async def test_on_behalf_of_without_permissions_returns_false(self, auth_config, principal_with_delegate):
        """Test that when delegated user lacks permissions, method returns False."""
        # Create a real AsyncClient but mock its post method
        mock_http_client = httpx.AsyncClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"allowed": False}}
        mock_response.raise_for_status = MagicMock()

        with patch.object(mock_http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            auth_client = AuthClient(
                principal=principal_with_delegate,
                config=auth_config,
                http_client=mock_http_client,
            )

            result = await auth_client.on_behalf_of_has_permissions(
                workspace_id="test-workspace",
                permissions=["secrets.read"],
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_on_behalf_of_has_permissions_sends_delegate_claims_when_present(
        self, auth_config, principal_service_delegating
    ):
        """Delegated PDP check uses on-behalf-of email/groups when set on the outer principal."""
        mock_http_client = httpx.AsyncClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"allowed": True}}
        mock_response.raise_for_status = MagicMock()

        with patch.object(mock_http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            auth_client = AuthClient(
                principal=principal_service_delegating,
                config=auth_config,
                http_client=mock_http_client,
            )

            await auth_client.on_behalf_of_has_permissions(
                workspace_id="test-workspace",
                permissions=["secrets.read"],
            )

            request_json = mock_post.call_args[1]["json"]
            assert request_json["input"]["principal_id"] == "user@example.com"
            assert request_json["input"]["principal_email"] == "user@example.com"
            assert request_json["input"]["principal_groups"] == ["workspace-editors"]
            assert "on_behalf_of_principal_id" not in request_json["input"]

    @pytest.mark.asyncio
    async def test_on_behalf_of_multiple_permissions(self, auth_config, principal_with_delegate):
        """Test checking multiple permissions for delegated user."""
        mock_http_client = httpx.AsyncClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"allowed": True}}
        mock_response.raise_for_status = MagicMock()

        with patch.object(mock_http_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            auth_client = AuthClient(
                principal=principal_with_delegate,
                config=auth_config,
                http_client=mock_http_client,
            )

            result = await auth_client.on_behalf_of_has_permissions(
                workspace_id="test-workspace",
                permissions=["secrets.read", "secrets.write"],
            )

            assert result is True

            # Verify all permissions were checked
            call_args = mock_post.call_args
            request_json = call_args[1]["json"]
            assert request_json["input"]["permissions"] == ["secrets.read", "secrets.write"]


class TestGetSdkOnBehalfOf:
    """Tests for the get_sdk_on_behalf_of SDK factory helper."""

    def test_adds_on_behalf_of_header_to_sync_sdk(self):
        """Test that get_sdk_on_behalf_of adds the on-behalf-of header to a sync SDK."""
        base_sdk = NeMoPlatform(
            base_url="http://testserver",
            default_headers={"X-NMP-Principal-Id": "service:my-service"},
        )

        delegated_sdk = get_sdk_on_behalf_of(base_sdk, "user@example.com")

        # Verify the SDK is a new instance with on-behalf-of configured
        assert delegated_sdk is not base_sdk
        # Verify default headers include the on-behalf-of header
        assert "X-NMP-Principal-On-Behalf-Of" in delegated_sdk.default_headers
        assert delegated_sdk.default_headers["X-NMP-Principal-On-Behalf-Of"] == "user@example.com"

    def test_get_sdk_on_behalf_of_with_principal_includes_email_and_groups(self):
        """Test that Principal delegation includes delegated email and groups headers."""
        base_sdk = NeMoPlatform(
            base_url="http://testserver",
            default_headers={"X-NMP-Principal-Id": "service:my-service"},
        )

        delegated_sdk = get_sdk_on_behalf_of(
            base_sdk,
            Principal(
                id="user@example.com",
                account_id="account-user",
                email="user@example.com",
                groups=["workspace-editors", "ml-team"],
                authz_aliases=["legacy-user", "user@example.com"],
            ),
        )

        assert delegated_sdk.default_headers["X-NMP-Principal-Id"] == "service:my-service"
        assert delegated_sdk.default_headers["X-NMP-Principal-On-Behalf-Of"] == "user@example.com"
        assert delegated_sdk.default_headers["X-NMP-Principal-On-Behalf-Of-Email"] == "user@example.com"
        assert delegated_sdk.default_headers["X-NMP-Principal-On-Behalf-Of-Groups"] == "workspace-editors,ml-team"
        assert delegated_sdk.default_headers["X-NMP-Subject-Account-Id"] == "account-user"
        assert delegated_sdk.default_headers["X-NMP-Subject-Aliases"] == "legacy-user,user@example.com"

    def test_preserves_original_sdk(self):
        """Test that get_sdk_on_behalf_of doesn't modify the original SDK."""
        base_sdk = NeMoPlatform(
            base_url="http://testserver",
            default_headers={"X-NMP-Principal-Id": "service:my-service"},
        )

        # Get original headers count
        original_headers = dict(base_sdk.default_headers)

        # Create delegated SDK
        delegated_sdk = get_sdk_on_behalf_of(base_sdk, "user@example.com")

        # Verify original SDK is unchanged
        assert base_sdk.default_headers == original_headers
        assert "X-NMP-Principal-On-Behalf-Of" not in base_sdk.default_headers

        # Verify delegated SDK has the new header
        assert "X-NMP-Principal-On-Behalf-Of" in delegated_sdk.default_headers

    def test_preserves_original_headers(self):
        """Test that get_sdk_on_behalf_of preserves all original headers."""
        base_sdk = NeMoPlatform(
            base_url="http://testserver",
            default_headers={
                "X-NMP-Principal-Id": "service:my-service",
                "X-Custom-Header": "custom-value",
                "Authorization": "Bearer token123",
            },
        )

        delegated_sdk = get_sdk_on_behalf_of(base_sdk, "user@example.com")

        # Verify all original headers are preserved
        assert delegated_sdk.default_headers["X-NMP-Principal-Id"] == "service:my-service"
        assert delegated_sdk.default_headers["X-Custom-Header"] == "custom-value"
        assert delegated_sdk.default_headers["Authorization"] == "Bearer token123"
        # And the new header is added
        assert delegated_sdk.default_headers["X-NMP-Principal-On-Behalf-Of"] == "user@example.com"

    def test_can_chain_delegations(self):
        """Test that get_sdk_on_behalf_of can be used multiple times."""
        base_sdk = NeMoPlatform(
            base_url="http://testserver",
            default_headers={"X-NMP-Principal-Id": "service:my-service"},
        )

        delegated_sdk1 = get_sdk_on_behalf_of(base_sdk, "user1@example.com")
        delegated_sdk2 = get_sdk_on_behalf_of(base_sdk, "user2@example.com")

        # Verify each delegation is independent and preserves original headers
        assert delegated_sdk1.default_headers["X-NMP-Principal-Id"] == "service:my-service"
        assert delegated_sdk1.default_headers["X-NMP-Principal-On-Behalf-Of"] == "user1@example.com"
        assert delegated_sdk2.default_headers["X-NMP-Principal-Id"] == "service:my-service"
        assert delegated_sdk2.default_headers["X-NMP-Principal-On-Behalf-Of"] == "user2@example.com"
        assert delegated_sdk1 is not delegated_sdk2
