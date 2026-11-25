# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenSandbox SDK services for one executor.

Lifecycle calls go to the OpenSandbox server with the API key. execd calls go through the ingress
gateway to the execd port of the sandbox, with the gateway's routing header, the sandbox's endpoint
headers and its execd token.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from types import TracebackType
from typing import Self

import httpx
from nemo_deployments_plugin.backends.opensandbox.config import OpenSandboxExecutorConfig
from nemo_deployments_plugin.backends.opensandbox.endpoint import (
    INGRESS_HEADER,
    SandboxEndpointAuth,
    endpoint_headers,
    forget_endpoint_headers,
    parse_route_label,
    route_label,
    sandbox_endpoint_host,
)
from opensandbox.adapters.factory import AdapterFactory
from opensandbox.config import ConnectionConfig
from opensandbox.constants import DEFAULT_EXECD_PORT
from opensandbox.exceptions import SandboxApiException, SandboxException
from opensandbox.models.sandboxes import SandboxEndpoint, SandboxFilter, SandboxInfo
from opensandbox.services import Commands, Filesystem, Sandboxes
from opensandbox.transport import RetryPolicy

EXECD_ACCESS_TOKEN_ENV = "EXECD_ACCESS_TOKEN"
EXECD_ACCESS_TOKEN_HEADER = "X-EXECD-ACCESS-TOKEN"

_PAGE_SIZE = 100
_MAX_DETAIL_CHARS = 500


def describe(action: str, exc: SandboxException) -> str:
    """Message for a failed OpenSandbox call. A transport failure is reduced to its type name."""
    if isinstance(exc, SandboxApiException):
        detail = _api_error_detail(exc)[:_MAX_DETAIL_CHARS]
        summary = f"{action} failed (HTTP {exc.status_code})"
        return f"{summary}: {detail}" if detail else summary
    # Messages are stored on the deployment; the transport error text can name internal addresses.
    return f"{action} failed: {type(exc.__cause__ or exc).__name__}"


def _api_error_detail(exc: SandboxApiException) -> str:
    """The error code and message of an API error response."""
    if exc.error.message:
        return f"{exc.error.code}: {exc.error.message}"
    # The SDK parses the error body only for the status codes in the OpenSandbox API specification.
    body = (exc.response_body or b"").decode(errors="replace").strip()
    try:
        parsed = json.loads(body)
    except ValueError:
        return body
    if isinstance(parsed, dict) and parsed.get("message"):
        return ": ".join(str(part) for part in (parsed.get("code"), parsed["message"]) if part)
    return body


@dataclass(frozen=True)
class Execd:
    """SDK services for the execd API of one sandbox."""

    commands: Commands
    files: Filesystem


class OpenSandboxClient:
    """Async context manager over the OpenSandbox SDK services of one executor."""

    def __init__(self, config: OpenSandboxExecutorConfig, api_key: str) -> None:
        self._api_key = api_key
        self._server_base = f"{config.protocol}://{config.domain}"
        self._gateway_address = config.gateway_address
        self._gateway_base = f"{config.protocol}://{config.gateway_address}"
        self._connection = ConnectionConfig(
            api_key=api_key,
            domain=config.domain,
            protocol=config.protocol,
            request_timeout=timedelta(seconds=config.request_timeout_seconds),
            # The deployment reconciler polls; a failed call is retried by the next poll.
            retry_policy=RetryPolicy.disabled(),
            endpoint_cache_disabled=True,
            event_hooks={"response": [self._forget_headers_on_401]},
        ).with_transport_if_missing()
        self._factory = AdapterFactory(self._connection)
        self.sandboxes: Sandboxes = self._factory.create_sandbox_service()
        create_connection = self._connection.model_copy(
            update={"request_timeout": timedelta(seconds=config.create_timeout_seconds)}
        )
        self.sandbox_creator: Sandboxes = AdapterFactory(create_connection).create_sandbox_service()
        self._probe = httpx.AsyncClient(
            auth=SandboxEndpointAuth(
                server_base=self._server_base, gateway_base=self._gateway_base, api_key_env=config.api_key_env
            ),
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self._probe.aclose()
        await self._connection.close_transport_if_owned()

    async def list_sandboxes(self, metadata: dict[str, str]) -> list[SandboxInfo]:
        """Sandboxes whose metadata contains every pair in ``metadata``."""
        sandboxes: list[SandboxInfo] = []
        page = 1
        while True:
            result = await self.sandboxes.list_sandboxes(
                SandboxFilter(metadata=metadata, page=page, page_size=_PAGE_SIZE)
            )
            sandboxes.extend(result.sandbox_infos)
            if not result.pagination.has_next_page:
                return sandboxes
            page += 1

    async def delete_sandbox(self, sandbox_id: str) -> None:
        """Delete a sandbox. A sandbox that is already gone is not an error."""
        try:
            await self.sandboxes.kill_sandbox(sandbox_id)
        except SandboxApiException as exc:
            if exc.status_code != 404:
                raise

    async def execd(self, sandbox_id: str, token: str) -> Execd:
        """execd services of a sandbox, authorized with its endpoint headers and ``token``."""
        headers = await endpoint_headers((self._server_base, sandbox_id, DEFAULT_EXECD_PORT), self._api_key)
        endpoint = SandboxEndpoint(
            endpoint=self._gateway_address,
            headers={
                **headers,
                INGRESS_HEADER: route_label(sandbox_id, DEFAULT_EXECD_PORT),
                EXECD_ACCESS_TOKEN_HEADER: token,
            },
        )
        return Execd(
            commands=self._factory.create_command_service(endpoint),
            files=self._factory.create_filesystem_service(endpoint),
        )

    async def http_get(self, sandbox_id: str, port: int, path: str, *, timeout: float) -> int | None:
        """Status code of ``GET path`` on ``port`` of the sandbox, or None when no response arrived."""
        try:
            response = await self._probe.get(
                f"{self._gateway_base}/{path.lstrip('/')}",
                headers={"Host": sandbox_endpoint_host(sandbox_id, port)},
                timeout=timeout,
            )
        except (httpx.HTTPError, ValueError):
            return None
        return response.status_code

    async def _forget_headers_on_401(self, response: httpx.Response) -> None:
        request = response.request
        route = parse_route_label(request.headers.get(INGRESS_HEADER, ""))
        if response.status_code == 401 and route is not None and str(request.url).startswith(f"{self._gateway_base}/"):
            forget_endpoint_headers((self._server_base, *route))
