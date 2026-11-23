# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP client for the OpenSandbox lifecycle API and a sandbox's execd API.

Lifecycle calls go to the OpenSandbox server with the API key. execd calls go through the ingress
gateway to port 44772 of the sandbox, with the sandbox's secure-access and execd tokens.
"""

from __future__ import annotations

import json
from types import TracebackType
from typing import Any, Self

import httpx
from nemo_deployments_plugin.backends.opensandbox.config import OpenSandboxExecutorConfig
from nemo_deployments_plugin.backends.opensandbox.endpoint import SandboxEndpointAuth

EXECD_PORT = 44772
EXECD_ACCESS_TOKEN_ENV = "EXECD_ACCESS_TOKEN"
EXECD_ACCESS_TOKEN_HEADER = "X-EXECD-ACCESS-TOKEN"
API_KEY_HEADER = "OPEN-SANDBOX-API-KEY"

_PAGE_SIZE = 100


class OpenSandboxError(RuntimeError):
    """An OpenSandbox or execd call failed. ``status_code`` is None when no response arrived."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _raise_for_status(response: httpx.Response, action: str) -> None:
    if response.is_success:
        return
    detail = response.text.strip()
    try:
        body = response.json()
        if isinstance(body, dict) and body.get("message"):
            detail = f"{body.get('code', '')}: {body['message']}".lstrip(": ")
    except ValueError:
        pass
    raise OpenSandboxError(
        f"{action} failed (HTTP {response.status_code}): {detail[:500]}", status_code=response.status_code
    )


class OpenSandboxClient:
    """Async context manager holding one HTTP client for the server and one for the gateway."""

    def __init__(self, config: OpenSandboxExecutorConfig, api_key: str) -> None:
        self._server_base = f"{config.protocol}://{config.domain}"
        self._gateway_base = f"{config.protocol}://{config.gateway_address}"
        timeout = httpx.Timeout(config.request_timeout_seconds)
        self._server = httpx.AsyncClient(
            base_url=f"{self._server_base}/v1", headers={API_KEY_HEADER: api_key}, timeout=timeout
        )
        self._gateway = httpx.AsyncClient(
            timeout=timeout,
            auth=SandboxEndpointAuth(
                server_base=self._server_base, gateway_base=self._gateway_base, api_key_env=config.api_key_env
            ),
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self._server.aclose()
        await self._gateway.aclose()

    # -- lifecycle API ---------------------------------------------------------------------------

    async def create_sandbox(self, body: dict[str, Any]) -> dict[str, Any]:
        response = await self._request(self._server, "POST", "/sandboxes", "create sandbox", json=body)
        return response.json()

    async def list_sandboxes(self, metadata: dict[str, str]) -> list[dict[str, Any]]:
        """Sandboxes whose metadata contains every pair in ``metadata``."""
        selector = "&".join(f"{key}={value}" for key, value in metadata.items())
        items: list[dict[str, Any]] = []
        page = 1
        while True:
            response = await self._request(
                self._server,
                "GET",
                "/sandboxes",
                "list sandboxes",
                params={"metadata": selector, "page": page, "pageSize": _PAGE_SIZE},
            )
            body = response.json()
            items.extend(body.get("items") or [])
            if not (body.get("pagination") or {}).get("hasNextPage"):
                return items
            page += 1

    async def patch_metadata(self, sandbox_id: str, patch: dict[str, str | None]) -> None:
        await self._request(
            self._server, "PATCH", f"/sandboxes/{sandbox_id}/metadata", "update sandbox metadata", json=patch
        )

    async def delete_sandbox(self, sandbox_id: str) -> None:
        """Delete a sandbox. A sandbox that is already gone is not an error."""
        try:
            await self._request(self._server, "DELETE", f"/sandboxes/{sandbox_id}", "delete sandbox")
        except OpenSandboxError as exc:
            if exc.status_code != 404:
                raise

    # -- execd API -------------------------------------------------------------------------------

    async def upload_file(self, sandbox_id: str, token: str, *, path: str, content: bytes, mode: int) -> None:
        """Write ``content`` to ``path``. ``mode`` is the permission bits, such as ``0o644``."""
        metadata = json.dumps({"path": path, "mode": int(f"{mode:o}")})
        await self._execd(
            sandbox_id,
            token,
            "POST",
            "/files/upload",
            "upload file",
            files={"metadata": ("metadata", metadata, "application/json"), "file": ("file", content)},
        )

    async def start_command(
        self, sandbox_id: str, token: str, argv: list[str], *, envs: dict[str, str] | None = None
    ) -> str:
        """Start ``argv`` as a background command and return its command id."""
        response = await self._execd(
            sandbox_id,
            token,
            "POST",
            "/command",
            "start command",
            json={"argv": argv, "background": True, "envs": envs or {}},
        )
        for line in response.text.splitlines():
            line = line.removeprefix("data:").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "init" and event.get("text"):
                return str(event["text"])
            if event.get("type") == "error":
                error = event.get("error") or {}
                raise OpenSandboxError(f"start command failed: {error.get('evalue') or error.get('ename')}")
        raise OpenSandboxError("start command failed: execd returned no command id")

    async def command_status(self, sandbox_id: str, token: str, command_id: str) -> dict[str, Any]:
        response = await self._execd(sandbox_id, token, "GET", f"/command/status/{command_id}", "read command status")
        return response.json()

    async def command_logs(self, sandbox_id: str, token: str, command_id: str) -> str:
        response = await self._execd(
            sandbox_id, token, "GET", f"/command/{command_id}/logs", "read command logs", params={"cursor": 0}
        )
        return response.text

    async def interrupt_command(self, sandbox_id: str, token: str, command_id: str) -> None:
        await self._execd(sandbox_id, token, "DELETE", "/command", "interrupt command", params={"id": command_id})

    async def http_get(self, sandbox_id: str, port: int, path: str, *, timeout: float) -> int:
        """Status code of ``GET path`` on ``port`` of the sandbox, through the gateway."""
        url = f"{self._gateway_base}/{sandbox_id}/{port}/{path.lstrip('/')}"
        try:
            response = await self._gateway.get(url, timeout=timeout)
        except httpx.HTTPError as exc:
            raise OpenSandboxError(f"GET {path} on port {port} failed: {type(exc).__name__}") from exc
        return response.status_code

    # -- transport -------------------------------------------------------------------------------

    async def _execd(
        self, sandbox_id: str, token: str, method: str, path: str, action: str, **kwargs: Any
    ) -> httpx.Response:
        url = f"{self._gateway_base}/{sandbox_id}/{EXECD_PORT}{path}"
        return await self._request(
            self._gateway, method, url, action, headers={EXECD_ACCESS_TOKEN_HEADER: token}, **kwargs
        )

    @staticmethod
    async def _request(client: httpx.AsyncClient, method: str, url: str, action: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            # Messages are stored on the deployment; the exception text can name internal addresses.
            raise OpenSandboxError(f"{action} failed: {type(exc).__name__}") from exc
        _raise_for_status(response, action)
        return response
