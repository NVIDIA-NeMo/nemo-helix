# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MCP HTTP transport authentication and the destructive-tool gate.

These run entirely in-process and need no running NeMo Helix instance.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
from nhx.core.mcp import main as mcp_main
from nhx.core.mcp.auth import (
    AUTH_TOKEN_ENV_VAR,
    MCPAuthConfigError,
    SharedBearerTokenVerifier,
    is_loopback_host,
    resolve_http_auth,
)
from nhx.core.mcp.server import create_server

TOKEN = "s3cret-token-0123456789"
INITIALIZE_BODY = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
}
MCP_HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("::1", True),
        ("localhost", True),
        ("0.0.0.0", False),
        ("::", False),
        ("10.1.2.3", False),
        ("example.com", False),
    ],
)
def test_is_loopback_host(host: str, expected: bool) -> None:
    assert is_loopback_host(host) is expected


class TestResolveHttpAuth:
    def test_token_set_returns_verifier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(AUTH_TOKEN_ENV_VAR, TOKEN)
        assert isinstance(resolve_http_auth("0.0.0.0", allow_unauthenticated=False), SharedBearerTokenVerifier)

    def test_short_token_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(AUTH_TOKEN_ENV_VAR, "short")
        with pytest.raises(MCPAuthConfigError, match="at least"):
            resolve_http_auth("127.0.0.1", allow_unauthenticated=False)

    @pytest.mark.parametrize("token", [" " * 16, f" {TOKEN}", f"{TOKEN}\n"])
    def test_whitespace_padded_token_rejected(self, monkeypatch: pytest.MonkeyPatch, token: str) -> None:
        monkeypatch.setenv(AUTH_TOKEN_ENV_VAR, token)
        with pytest.raises(MCPAuthConfigError, match="whitespace"):
            resolve_http_auth("127.0.0.1", allow_unauthenticated=False)

    def test_no_token_refused_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(AUTH_TOKEN_ENV_VAR, raising=False)
        with pytest.raises(MCPAuthConfigError, match="requires authentication"):
            resolve_http_auth("127.0.0.1", allow_unauthenticated=False)

    def test_opt_out_allowed_on_loopback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(AUTH_TOKEN_ENV_VAR, raising=False)
        assert resolve_http_auth("127.0.0.1", allow_unauthenticated=True) is None

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.1.2.3", "example.com"])
    def test_opt_out_refused_off_loopback(self, monkeypatch: pytest.MonkeyPatch, host: str) -> None:
        monkeypatch.delenv(AUTH_TOKEN_ENV_VAR, raising=False)
        with pytest.raises(MCPAuthConfigError, match="loopback"):
            resolve_http_auth(host, allow_unauthenticated=True)


class TestSharedBearerTokenVerifier:
    @pytest.mark.asyncio
    async def test_accepts_configured_token(self) -> None:
        assert await SharedBearerTokenVerifier(TOKEN).verify_token(TOKEN) is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("presented", ["", "wrong", TOKEN + "x", TOKEN[:-1]])
    async def test_rejects_other_tokens(self, presented: str) -> None:
        assert await SharedBearerTokenVerifier(TOKEN).verify_token(presented) is None


class TestHttpTransport:
    @staticmethod
    async def _post_initialize(
        headers: dict[str, str],
        *,
        base_url: str = "http://test",
        allowed_hosts: list[str] | None = None,
    ) -> int:
        server = create_server("http://unused.example.com", auth=SharedBearerTokenVerifier(TOKEN))
        app = server.http_app(transport="streamable-http", host_origin_protection="auto", allowed_hosts=allowed_hosts)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url=base_url) as client:
                response = await client.post("/mcp", json=INITIALIZE_BODY, headers={**MCP_HEADERS, **headers})
        return response.status_code

    @pytest.mark.asyncio
    async def test_request_without_token_is_rejected(self) -> None:
        assert await self._post_initialize({}) == 401

    @pytest.mark.asyncio
    async def test_request_with_wrong_token_is_rejected(self) -> None:
        assert await self._post_initialize({"Authorization": "Bearer not-the-token-000000"}) == 401

    @pytest.mark.asyncio
    async def test_request_with_valid_token_is_accepted(self) -> None:
        assert await self._post_initialize({"Authorization": f"Bearer {TOKEN}"}) == 200

    @pytest.mark.asyncio
    async def test_untrusted_host_on_loopback_is_rejected(self) -> None:
        headers = {"Authorization": f"Bearer {TOKEN}", "Host": "rebind.example.com"}
        assert await self._post_initialize(headers, base_url="http://127.0.0.1") == 421

    @pytest.mark.asyncio
    async def test_untrusted_origin_on_loopback_is_rejected(self) -> None:
        headers = {"Authorization": f"Bearer {TOKEN}", "Origin": "https://evil.example.com"}
        assert await self._post_initialize(headers, base_url="http://127.0.0.1") == 403

    @pytest.mark.asyncio
    async def test_tls_proxy_public_host_requires_allowed_hosts(self) -> None:
        headers = {"Authorization": f"Bearer {TOKEN}", "Host": "mcp.example.com"}
        assert await self._post_initialize(headers, base_url="http://127.0.0.1") == 421
        assert (
            await self._post_initialize(headers, base_url="http://127.0.0.1", allowed_hosts=["mcp.example.com"]) == 200
        )


class TestDestructiveToolGate:
    @staticmethod
    async def _tool_names(enable_destructive_tools: bool = False) -> set[str]:
        server = create_server("http://unused.example.com", enable_destructive_tools=enable_destructive_tools)
        return {tool.name for tool in await server.list_tools()}

    @pytest.mark.asyncio
    async def test_delete_workspace_hidden_by_default(self) -> None:
        names = await self._tool_names()
        assert "delete_workspace" not in names
        assert {"list_workspaces", "create_workspace"} <= names

    @pytest.mark.asyncio
    async def test_delete_workspace_exposed_when_enabled(self) -> None:
        assert "delete_workspace" in await self._tool_names(enable_destructive_tools=True)


class TestCliStartup:
    @pytest.fixture
    def create_server_stub(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        """Replace the server factory so main() never builds or serves a real server."""
        stub = MagicMock()
        monkeypatch.setattr(mcp_main, "create_server", stub)
        monkeypatch.delenv(AUTH_TOKEN_ENV_VAR, raising=False)
        return stub

    @staticmethod
    def _run_main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
        monkeypatch.setattr("sys.argv", ["nemo-mcp", *argv])
        mcp_main.main()

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (["--transport", "streamable-http"], "requires authentication"),
            (
                ["--transport", "streamable-http", "--host", "0.0.0.0", "--allow-unauthenticated"],
                "only permitted on a loopback host",
            ),
            (
                ["--transport", "streamable-http", "--allow-unauthenticated", "--enable-destructive-tools"],
                "cannot be combined",
            ),
            (["--allow-unauthenticated"], "only applies to --transport streamable-http"),
        ],
    )
    def test_unsafe_http_combinations_exit_before_serving(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        create_server_stub: MagicMock,
        argv: list[str],
        message: str,
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            self._run_main(monkeypatch, *argv)
        assert exc_info.value.code == 2
        assert message in capsys.readouterr().err
        create_server_stub.assert_not_called()

    def test_allow_unauthenticated_with_token_set_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        create_server_stub: MagicMock,
    ) -> None:
        monkeypatch.setenv(AUTH_TOKEN_ENV_VAR, TOKEN)
        with pytest.raises(SystemExit) as exc_info:
            self._run_main(monkeypatch, "--transport", "streamable-http", "--allow-unauthenticated")
        assert exc_info.value.code == 2
        assert f"cannot be combined with {AUTH_TOKEN_ENV_VAR}" in capsys.readouterr().err
        create_server_stub.assert_not_called()

    def test_http_with_token_passes_verifier_to_server(
        self, monkeypatch: pytest.MonkeyPatch, create_server_stub: MagicMock
    ) -> None:
        monkeypatch.setenv(AUTH_TOKEN_ENV_VAR, TOKEN)
        self._run_main(monkeypatch, "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "9001")

        kwargs = create_server_stub.call_args.kwargs
        assert isinstance(kwargs["auth"], SharedBearerTokenVerifier)
        assert kwargs["enable_destructive_tools"] is False
        create_server_stub.return_value.run.assert_called_once_with(
            transport="streamable-http", host="0.0.0.0", port=9001, host_origin_protection="auto"
        )

    def test_enable_destructive_tools_reaches_server_when_authenticated(
        self, monkeypatch: pytest.MonkeyPatch, create_server_stub: MagicMock
    ) -> None:
        monkeypatch.setenv(AUTH_TOKEN_ENV_VAR, TOKEN)
        self._run_main(monkeypatch, "--transport", "streamable-http", "--enable-destructive-tools")
        assert create_server_stub.call_args.kwargs["enable_destructive_tools"] is True

    def test_loopback_opt_out_serves_without_auth(
        self, monkeypatch: pytest.MonkeyPatch, create_server_stub: MagicMock
    ) -> None:
        self._run_main(monkeypatch, "--transport", "streamable-http", "--allow-unauthenticated")
        assert create_server_stub.call_args.kwargs["auth"] is None
        create_server_stub.return_value.run.assert_called_once_with(
            transport="streamable-http", host="127.0.0.1", port=8000, host_origin_protection="auto"
        )

    def test_stdio_needs_no_token_and_no_auth(
        self, monkeypatch: pytest.MonkeyPatch, create_server_stub: MagicMock
    ) -> None:
        self._run_main(monkeypatch)
        assert create_server_stub.call_args.kwargs["auth"] is None
        create_server_stub.return_value.run.assert_called_once_with(transport="stdio")

    def test_http_transport_enables_host_origin_protection(
        self,
        monkeypatch: pytest.MonkeyPatch,
        create_server_stub: MagicMock,
    ) -> None:
        monkeypatch.setenv(AUTH_TOKEN_ENV_VAR, TOKEN)
        self._run_main(monkeypatch, "--transport", "streamable-http", "--port", "8080")
        create_server_stub.return_value.run.assert_called_once_with(
            transport="streamable-http", host="127.0.0.1", port=8080, host_origin_protection="auto"
        )
