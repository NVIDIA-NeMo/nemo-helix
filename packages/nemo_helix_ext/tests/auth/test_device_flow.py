# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for OAuth device flow."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from nemo_helix_ext.auth.device_flow import (
    DeviceCodeResponse,
    DeviceFlow,
    DeviceFlowError,
    TokenResponse,
    authenticate_with_password_grant,
    refresh_access_token,
)
from nemo_helix_ext.client.tls import NHX_CLIENT_SSL_CERT_FILE_ENVVAR


class TestDeviceCodeResponse:
    """Tests for DeviceCodeResponse dataclass."""

    def test_device_code_response_creation(self):
        """Test creating a DeviceCodeResponse instance."""
        response = DeviceCodeResponse(
            device_code="abc123",
            user_code="XYZ-789",
            verification_uri="https://sso.example.com/device",
            verification_uri_complete="https://sso.example.com/device?user_code=XYZ-789",
            expires_in=900,
            interval=5,
        )

        assert response.device_code == "abc123"
        assert response.user_code == "XYZ-789"
        assert response.verification_uri == "https://sso.example.com/device"
        assert response.verification_uri_complete == "https://sso.example.com/device?user_code=XYZ-789"
        assert response.expires_in == 900
        assert response.interval == 5


class TestTokenResponse:
    """Tests for TokenResponse dataclass."""

    def test_token_response_creation(self):
        """Test creating a TokenResponse instance."""
        response = TokenResponse(
            access_token="access_token_123",
            id_token=None,
            refresh_token="refresh_token_456",
            token_type="Bearer",
            expires_in=3600,
            scope="openid email profile",
        )

        assert response.access_token == "access_token_123"
        assert response.refresh_token == "refresh_token_456"
        assert response.token_type == "Bearer"
        assert response.expires_in == 3600
        assert response.scope == "openid email profile"

    def test_token_response_without_refresh_token(self):
        """Test TokenResponse without refresh token."""
        response = TokenResponse(
            access_token="access_token_123",
            id_token=None,
            refresh_token=None,
            token_type="Bearer",
            expires_in=3600,
            scope=None,
        )

        assert response.access_token == "access_token_123"
        assert response.refresh_token is None

    def test_token_for_nhx_selects_configured_id_token(self):
        response = TokenResponse(
            access_token="opaque-access-token",
            id_token="signed-id-token",
            refresh_token="refresh-token",
            token_type="Bearer",
            expires_in=3600,
            scope="openid",
            bearer_token_source="id_token",
        )

        assert response.token_for_nhx == "signed-id-token"

    def test_token_for_nhx_requires_configured_id_token(self):
        response = TokenResponse(
            access_token="opaque-access-token",
            id_token=None,
            refresh_token=None,
            token_type="Bearer",
            expires_in=3600,
            scope="openid",
            bearer_token_source="id_token",
        )

        with pytest.raises(DeviceFlowError, match="did not return one"):
            _ = response.token_for_nhx

    def test_token_for_nhx_rejects_unknown_source(self):
        response = TokenResponse(
            access_token="opaque-access-token",
            id_token="signed-id-token",
            refresh_token=None,
            token_type="Bearer",
            expires_in=3600,
            scope="openid",
            bearer_token_source="refresh_token",  # type: ignore[arg-type]
        )

        with pytest.raises(ValueError, match="bearer_token_source"):
            _ = response.token_for_nhx


class TestDeviceFlow:
    """Tests for DeviceFlow class."""

    @pytest.fixture
    def device_flow(self):
        """Create a DeviceFlow instance for testing."""
        return DeviceFlow(
            device_authorization_endpoint="https://sso.example.com/device/code",
            token_endpoint="https://sso.example.com/oauth/token",
            client_id="test-client",
            scope="openid email profile",
        )

    @pytest.mark.asyncio
    async def test_start_device_authorization_success(self, device_flow):
        """Test successful device authorization request."""
        mock_response_data = {
            "device_code": "device_code_123",
            "user_code": "ABC-123",
            "verification_uri": "https://sso.example.com/device",
            "verification_uri_complete": "https://sso.example.com/device?code=ABC-123",
            "expires_in": 1800,
            "interval": 5,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await device_flow.start_device_authorization()

            assert result.device_code == "device_code_123"
            assert result.user_code == "ABC-123"
            assert result.verification_uri == "https://sso.example.com/device"
            assert result.expires_in == 1800
            assert result.interval == 5

            mock_client.post.assert_called_once_with(
                "https://sso.example.com/device/code",
                data={
                    "client_id": "test-client",
                    "scope": "openid email profile",
                },
                timeout=30.0,
            )
            mock_client_class.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_start_device_authorization_includes_stable_device_metadata(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        flow = DeviceFlow(
            device_authorization_endpoint="https://sso.example.com/device/code",
            token_endpoint="https://sso.example.com/oauth/token",
            client_id="test-client",
            scope="openid email profile",
            include_device_id=True,
            device_display_name="NeMo Helix CLI",
        )
        mock_response_data = {
            "device_code": "device_code_123",
            "user_code": "ABC-123",
            "verification_uri": "https://sso.example.com/device",
            "expires_in": 1800,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            await flow.start_device_authorization()

            request_data = mock_client.post.call_args.kwargs["data"]
            assert request_data["client_id"] == "test-client"
            assert request_data["scope"] == "openid email profile"
            assert request_data["device_id"] == flow.device_id
            assert request_data["display_name"] == "NeMo Helix CLI"
            assert flow.device_id is not None

        second_flow = DeviceFlow(
            device_authorization_endpoint="https://sso.example.com/device/code",
            token_endpoint="https://sso.example.com/oauth/token",
            client_id="test-client",
            include_device_id=True,
        )
        device_id_path = tmp_path / "nhx" / "oidc-device-id"
        assert second_flow.device_id == flow.device_id
        assert device_id_path.read_text(encoding="utf-8").strip() == flow.device_id
        assert device_id_path.stat().st_mode & 0o777 == 0o600

    @pytest.mark.asyncio
    async def test_start_device_authorization_uses_context_certificate_authority(self, monkeypatch):
        """Test device authorization uses an explicit context CA bundle."""
        monkeypatch.delenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)
        flow = DeviceFlow(
            device_authorization_endpoint="https://sso.example.com/device/code",
            token_endpoint="https://sso.example.com/oauth/token",
            client_id="test-client",
            certificate_authority="/tmp/context-ca.pem",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {
                "device_code": "device_code_123",
                "user_code": "ABC-123",
                "verification_uri": "https://sso.example.com/device",
                "expires_in": 1800,
            }
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            await flow.start_device_authorization()

            mock_client_class.assert_called_once_with(verify="/tmp/context-ca.pem")

    @pytest.mark.asyncio
    async def test_start_device_authorization_uses_nemo_scoped_ca_bundle(self, device_flow, monkeypatch):
        mock_response_data = {
            "device_code": "device_code_123",
            "user_code": "ABC-123",
            "verification_uri": "https://sso.example.com/device",
            "expires_in": 1800,
        }
        monkeypatch.setenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, "/tmp/nemo-ca.pem")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            await device_flow.start_device_authorization()

            mock_client_class.assert_called_once_with(verify="/tmp/nemo-ca.pem")

    @pytest.mark.asyncio
    async def test_start_device_authorization_default_interval(self, device_flow):
        """Test device authorization with default interval."""
        mock_response_data = {
            "device_code": "device_code_123",
            "user_code": "ABC-123",
            "verification_uri": "https://sso.example.com/device",
            "expires_in": 1800,
            # No interval provided
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await device_flow.start_device_authorization()

            assert result.interval == 5  # Default interval

    @pytest.mark.asyncio
    async def test_start_device_authorization_wraps_http_error(self, device_flow):
        request = httpx.Request("POST", "https://sso.example.com/device/code")
        response = httpx.Response(
            400,
            request=request,
            json={"error": "invalid_client", "error_description": "Device flow is disabled"},
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            with pytest.raises(DeviceFlowError, match="HTTP 400: Device flow is disabled"):
                await device_flow.start_device_authorization()

    @pytest.mark.asyncio
    async def test_poll_for_token_success(self, device_flow):
        """Test successful token polling."""
        mock_token_response = {
            "access_token": "access_123",
            "refresh_token": "refresh_456",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "openid email",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_token_response
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            with patch("nemo_helix_ext.auth.device_flow._async_pause", new_callable=AsyncMock):
                result = await device_flow.poll_for_token(
                    device_code="device_123",
                    interval=1,  # Short interval for testing
                    expires_in=60,
                )

            assert result.access_token == "access_123"
            assert result.refresh_token == "refresh_456"
            assert mock_client.post.call_args.kwargs["data"]["scope"] == "openid email profile"

    @pytest.mark.asyncio
    async def test_poll_for_token_can_omit_scope(self):
        flow = DeviceFlow(
            device_authorization_endpoint="https://sso.example.com/device/code",
            token_endpoint="https://sso.example.com/oauth/token",
            client_id="test-client",
            scope="openid email profile",
            include_scope_in_token_request=False,
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"access_token": "access_123"}
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            with patch("nemo_helix_ext.auth.device_flow._async_pause", new_callable=AsyncMock):
                await flow.poll_for_token(device_code="device_123", interval=1, expires_in=60)

        assert mock_client.post.call_args.kwargs["data"] == {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "client_id": "test-client",
            "device_code": "device_123",
        }

    @pytest.mark.asyncio
    async def test_poll_for_token_handles_bodyless_unauthorized_response(self, device_flow):
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.json.side_effect = ValueError("empty response")
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            with (
                patch("nemo_helix_ext.auth.device_flow._async_pause", new_callable=AsyncMock),
                pytest.raises(DeviceFlowError, match="invalid or expired"),
            ):
                await device_flow.poll_for_token(device_code="device_123", interval=1, expires_in=60)

    @pytest.mark.asyncio
    async def test_poll_for_token_authorization_pending(self, device_flow):
        """Test polling with authorization pending then success."""
        pending_response = MagicMock()
        pending_response.status_code = 400
        pending_response.json.return_value = {"error": "authorization_pending"}

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "access_token": "access_123",
            "token_type": "Bearer",
            "expires_in": 3600,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = [pending_response, success_response]
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            with patch("nemo_helix_ext.auth.device_flow._async_pause", new_callable=AsyncMock):
                result = await device_flow.poll_for_token(
                    device_code="device_123",
                    interval=1,
                    expires_in=60,
                )

            assert result.access_token == "access_123"
            assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_poll_for_token_expired(self, device_flow):
        """Test polling with expired token error."""
        error_response = MagicMock()
        error_response.status_code = 400
        error_response.json.return_value = {"error": "expired_token"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = error_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            with patch("nemo_helix_ext.auth.device_flow._async_pause", new_callable=AsyncMock):
                with pytest.raises(DeviceFlowError, match="Authorization request expired"):
                    await device_flow.poll_for_token(
                        device_code="device_123",
                        interval=1,
                        expires_in=60,
                    )

    @pytest.mark.asyncio
    async def test_poll_for_token_access_denied(self, device_flow):
        """Test polling with access denied error."""
        error_response = MagicMock()
        error_response.status_code = 400
        error_response.json.return_value = {"error": "access_denied"}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = error_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            with patch("nemo_helix_ext.auth.device_flow._async_pause", new_callable=AsyncMock):
                with pytest.raises(DeviceFlowError, match="User denied authorization"):
                    await device_flow.poll_for_token(
                        device_code="device_123",
                        interval=1,
                        expires_in=60,
                    )

    @pytest.mark.asyncio
    async def test_poll_for_token_slow_down(self, device_flow):
        """Test polling with slow_down response increases interval."""
        slow_down_response = MagicMock()
        slow_down_response.status_code = 400
        slow_down_response.json.return_value = {"error": "slow_down"}

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "access_token": "access_123",
            "token_type": "Bearer",
            "expires_in": 3600,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = [slow_down_response, success_response]
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            sleep_calls = []

            async def mock_pause(seconds):
                sleep_calls.append(seconds)

            with patch("nemo_helix_ext.auth.device_flow._async_pause", side_effect=mock_pause):
                result = await device_flow.poll_for_token(
                    device_code="device_123",
                    interval=5,
                    expires_in=60,
                )

            assert result.access_token == "access_123"
            # First sleep should be 5, second should be 10 (5 + 5 after slow_down)
            assert sleep_calls[0] == 5
            assert sleep_calls[1] == 10


class TestDeviceFlowError:
    """Tests for DeviceFlowError exception."""

    def test_device_flow_error_message(self):
        """Test DeviceFlowError with message."""
        error = DeviceFlowError("Test error message")
        assert str(error) == "Test error message"


class TestRefreshAccessToken:
    @pytest.mark.asyncio
    async def test_refresh_access_token_success(self):
        with patch("nemo_helix_ext.auth.device_flow.refresh_token_grant") as mock_refresh:
            mock_refresh.return_value = {
                "access_token": "new_access",
                "refresh_token": "new_refresh",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "openid profile",
            }

            response = await refresh_access_token(
                token_endpoint="https://idp/token",
                client_id="client-id",
                refresh_token="refresh-token",
                scope="openid profile",
            )

            assert response.access_token == "new_access"
            assert response.refresh_token == "new_refresh"
            assert response.scope == "openid profile"

            mock_refresh.assert_called_once_with(
                "https://idp/token",
                "client-id",
                "refresh-token",
                scope="openid profile",
                certificate_authority=None,
            )

    @pytest.mark.asyncio
    async def test_refresh_access_token_uses_context_certificate_authority(self):
        with patch("nemo_helix_ext.auth.device_flow.refresh_token_grant") as mock_refresh:
            mock_refresh.return_value = {
                "access_token": "new_access",
                "token_type": "Bearer",
                "expires_in": 3600,
            }

            await refresh_access_token(
                token_endpoint="https://idp/token",
                client_id="client-id",
                refresh_token="refresh-token",
                certificate_authority="/tmp/context-ca.pem",
            )

            assert mock_refresh.call_args.kwargs["certificate_authority"] == "/tmp/context-ca.pem"

    @pytest.mark.asyncio
    async def test_refresh_access_token_failure(self):
        with patch("nemo_helix_ext.auth.device_flow.refresh_token_grant") as mock_refresh:
            mock_refresh.side_effect = RuntimeError("Token refresh failed (HTTP 400): invalid_grant")

            with pytest.raises(DeviceFlowError, match="Token refresh failed"):
                await refresh_access_token(
                    token_endpoint="https://idp/token",
                    client_id="client-id",
                    refresh_token="refresh-token",
                )


class TestPasswordGrant:
    def test_authenticate_with_password_grant_success(self):
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "access_token": "access_123",
                "refresh_token": "refresh_456",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "openid profile email",
            }
            mock_client.post.return_value = mock_response
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = None
            mock_client_class.return_value = mock_client

            response = authenticate_with_password_grant(
                token_endpoint="https://idp/token",
                client_id="client-id",
                username="user",
                password="secret",
                scope="openid profile email",
            )

            assert response.access_token == "access_123"
            assert response.refresh_token == "refresh_456"
            assert response.scope == "openid profile email"

            mock_client.post.assert_called_once_with(
                "https://idp/token",
                data={
                    "grant_type": "password",
                    "client_id": "client-id",
                    "username": "user",
                    "password": "secret",
                    "scope": "openid profile email",
                },
                timeout=30.0,
            )
            mock_client_class.assert_called_once_with()

    def test_authenticate_with_password_grant_uses_nemo_scoped_ca_bundle(self, monkeypatch):
        monkeypatch.setenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, "/tmp/nemo-ca.pem")

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "access_token": "access_123",
                "token_type": "Bearer",
                "expires_in": 3600,
            }
            mock_client.post.return_value = mock_response
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = None
            mock_client_class.return_value = mock_client

            authenticate_with_password_grant(
                token_endpoint="https://idp/token",
                client_id="client-id",
                username="user",
                password="secret",
            )

            mock_client_class.assert_called_once_with(verify="/tmp/nemo-ca.pem")

    def test_authenticate_with_password_grant_uses_context_certificate_authority(self, monkeypatch):
        monkeypatch.delenv(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, raising=False)

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "access_token": "access_123",
                "token_type": "Bearer",
                "expires_in": 3600,
            }
            mock_client.post.return_value = mock_response
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = None
            mock_client_class.return_value = mock_client

            authenticate_with_password_grant(
                token_endpoint="https://idp/token",
                client_id="client-id",
                username="user",
                password="secret",
                certificate_authority="/tmp/context-ca.pem",
            )

            mock_client_class.assert_called_once_with(verify="/tmp/context-ca.pem")

    def test_authenticate_with_password_grant_failure(self):
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = {
                "error": "invalid_grant",
                "error_description": "Bad credentials",
            }
            mock_response.text = "Bad credentials"
            mock_client.post.return_value = mock_response
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = None
            mock_client_class.return_value = mock_client

            with pytest.raises(DeviceFlowError, match="Token request failed: invalid_grant - Bad credentials"):
                authenticate_with_password_grant(
                    token_endpoint="https://idp/token",
                    client_id="client-id",
                    username="user",
                    password="wrong",
                )
