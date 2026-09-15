# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent gateway proxy routes.

Two proxy routes:

``/v2/workspaces/{workspace}/agents/{name}/-/{trailing_uri}``
    Proxy by **agent name** — gateway resolves the active deployment and
    forwards the request.  This is the primary user-facing path, analogous to
    how IGW routes by model name.

``/v2/workspaces/{workspace}/deployments/{name}/-/{trailing_uri}``
    Proxy by **deployment name** — for direct targeting of a specific
    deployment (e.g. A/B testing).

The ``/-/`` separator prevents URL conflicts with the CRUD routes
(``/agents/{name}`` and ``/deployments/{name}``).  This mirrors the pattern
used by the Inference Gateway.

Streaming and SSE are supported: the response is streamed back to the client
chunk by chunk.  ``text/event-stream`` responses bypass buffering.

Failures under the OpenAI-compatible ``/-/v1/*`` surface are rendered with an
OpenAI ``error`` envelope beside FastAPI's ``detail`` (see ``openai_errors``);
every other proxied path keeps ``detail`` alone.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime, timedelta
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from nemo_agents_plugin.api.v2._perms import GatewayPerms
from nemo_agents_plugin.api.v2.dependencies import get_entity_client
from nemo_agents_plugin.api.v2.openai_errors import (
    UpstreamAgentError,
    augment_upstream_error_body,
    is_openai_compatible_uri,
    openai_error_response,
)
from nemo_agents_plugin.api.v2.session_access import get_owned_session_by_id
from nemo_agents_plugin.authz import scope
from nemo_agents_plugin.deployment_routing import (
    get_deployment_endpoint,
    get_deployment_transport,
    is_deployment_routable,
)
from nemo_agents_plugin.entities import Agent, AgentDeployment, AgentSession, SessionStatus
from nemo_agents_plugin.fabric.session_manager import DEFAULT_IDLE_SESSION_TIMEOUT_SECONDS
from nemo_agents_plugin.session_lifecycle import session_expiration_is_due
from nemo_agents_plugin.session_protocol import SESSION_ID_HEADER
from nemo_deployments_plugin.endpoint_transport import DIRECT, EndpointTransport
from nemo_platform_plugin.authz import CallerKind, path_rule
from nemo_platform_plugin.dependencies import get_effective_principal_id
from nemo_platform_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Platform-created Fabric deployments currently use the serving default. If the
# timeout becomes deployment-configurable, this value must come from the deployment.
_SESSION_IDLE_TIMEOUT = timedelta(seconds=DEFAULT_IDLE_SESSION_TIMEOUT_SECONDS)

# HTTP methods forwarded through the proxy, split by authorization scope. Read-like methods
# require only agents:read; mutating methods require agents:write. This mirrors the Inference
# Gateway's proxy precedent (its GET proxy is scoped inference:read), so a read-scoped token is
# not denied on read-only proxy calls. Both groups still require the same agents.gateway.invoke
# permission.
_PROXY_READ_METHODS = ["GET", "HEAD", "OPTIONS"]
_PROXY_WRITE_METHODS = ["POST", "PUT", "PATCH", "DELETE"]

_HOP_BY_HOP_HEADERS = {
    "host",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

# Platform-internal headers should not leak to the agent process
_PLATFORM_INTERNAL_HEADERS = {
    "x-nmp-principal-id",
    "x-nmp-principal-on-behalf-of",
}

# Headers we strip before forwarding to the agent process (hop-by-hop + platform-internal + session ID)
_REQUEST_HEADERS_TO_STRIP = _HOP_BY_HOP_HEADERS | _PLATFORM_INTERNAL_HEADERS | {SESSION_ID_HEADER.lower()}

# Headers we strip before forwarding to the client (hop-by-hop + platform-internal)
_RESPONSE_HEADERS_TO_STRIP = _HOP_BY_HOP_HEADERS | _PLATFORM_INTERNAL_HEADERS

# ---------------------------------------------------------------------------
# Endpoint resolution — subprocess vs. container mode
# ---------------------------------------------------------------------------
#
# STOP-GAP: the shared deployment_routing helpers are the *only* place the agents
# plugin learns where a deployment lives. For subprocess deployments that is the loopback
# ``AgentDeployment.endpoint`` the agents plugin bakes in at spawn. For container
# deployments (docker/k8s) the real address (k8s Service DNS, docker host:port)
# is known only to the deployments plugin and projected onto ``endpoints`` by the
# agents controller. The rest of the proxy (streaming, SSE, header stripping, the
# SSRF origin guard in ``_proxy``) is mode-agnostic and untouched.
#
# POSSIBLE FUTURE DIRECTION: one option being considered is folding agent routing
# into the Inference Gateway (so IGW also serves as the agents gateway). If that
# direction is taken, this bespoke proxy and its deployment-routing helpers would likely retire.
# That re-architecture is not committed to here; this stop-gap stands on its own.

# Container modes (docker/k8s) resolve their address from the projected ``endpoints``
# rather than the loopback ``endpoint``. The agents controller maps the deployments-plugin
# Deployment.status (READY/...) onto the agents-local status, so a routable container
# deployment reads as "running" — the same value subprocess uses.


async def _serve_agent_proxy(
    workspace: str,
    name: str,
    trailing_uri: str,
    request: Request,
    entity_client: NemoEntitiesClient,
    effective_principal_id: str,
) -> Response:
    """Resolve the target deployment for the named agent and forward the request to it.

    A persisted session makes its bound deployment authoritative. Without a session, this keeps
    the existing first-routable-deployment behavior. Shared by the read/write route handlers,
    which differ only in their authorization scope (``agents:read`` vs ``agents:write``).
    """
    try:
        return await _resolve_and_proxy_agent(
            workspace,
            name,
            trailing_uri,
            request,
            entity_client,
            effective_principal_id,
        )
    except HTTPException as exc:
        if not is_openai_compatible_uri(trailing_uri):
            raise
        return openai_error_response(exc)


async def _resolve_and_proxy_agent(
    workspace: str,
    name: str,
    trailing_uri: str,
    request: Request,
    entity_client: NemoEntitiesClient,
    effective_principal_id: str,
) -> StreamingResponse:
    """Pick the deployment for *name* — session-bound if one was supplied — and forward."""
    session = await _resolve_request_session(request, workspace, entity_client, effective_principal_id)
    if session is None:
        deployment = await _resolve_agent_deployment(name, workspace, entity_client)
    else:
        deployment = await _resolve_session_deployment(session, workspace, entity_client)
        if deployment.agent != name:
            raise _session_deployment_mismatch(
                session,
                detail=(f"its deployment belongs to agent '{deployment.agent}', not requested agent '{name}'"),
            )

    return await _proxy_deployment(
        request,
        deployment,
        trailing_uri,
        model_name=name,
        entity_client=entity_client,
        session=session,
    )


@router.api_route(
    "/agents/{name}/-/{trailing_uri:path}",
    methods=_PROXY_READ_METHODS,
    tags=["Agent Gateway"],
    include_in_schema=False,
)
@scope.read
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[GatewayPerms.INVOKE],
)
async def proxy_by_agent_name_read(
    workspace: str,
    name: str,
    trailing_uri: str,
    request: Request,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
    effective_principal_id: str = Depends(get_effective_principal_id),
) -> Response:
    """Read-scoped (GET/HEAD/OPTIONS) proxy to the active deployment for *agent name*."""
    return await _serve_agent_proxy(
        workspace,
        name,
        trailing_uri,
        request,
        entity_client,
        effective_principal_id,
    )


@router.api_route(
    "/agents/{name}/-/{trailing_uri:path}",
    methods=_PROXY_WRITE_METHODS,
    tags=["Agent Gateway"],
    include_in_schema=False,
)
@scope.write
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[GatewayPerms.INVOKE],
)
async def proxy_by_agent_name_write(
    workspace: str,
    name: str,
    trailing_uri: str,
    request: Request,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
    effective_principal_id: str = Depends(get_effective_principal_id),
) -> Response:
    """Write-scoped (POST/PUT/PATCH/DELETE) proxy to the active deployment for *agent name*."""
    return await _serve_agent_proxy(
        workspace,
        name,
        trailing_uri,
        request,
        entity_client,
        effective_principal_id,
    )


async def _serve_deployment_proxy(
    workspace: str,
    name: str,
    trailing_uri: str,
    request: Request,
    entity_client: NemoEntitiesClient,
    effective_principal_id: str,
) -> Response:
    """Proxy a request directly to the named deployment.

    Returns ``404`` if the deployment doesn't exist, ``503`` if it isn't currently running.
    Shared by the read/write route handlers, which differ only in authorization scope.
    """
    try:
        return await _resolve_and_proxy_deployment(
            workspace,
            name,
            trailing_uri,
            request,
            entity_client,
            effective_principal_id,
        )
    except HTTPException as exc:
        if not is_openai_compatible_uri(trailing_uri):
            raise
        return openai_error_response(exc)


async def _resolve_and_proxy_deployment(
    workspace: str,
    name: str,
    trailing_uri: str,
    request: Request,
    entity_client: NemoEntitiesClient,
    effective_principal_id: str,
) -> StreamingResponse:
    """Look the deployment up, check it against any supplied session, and forward."""
    try:
        dep = await entity_client.get(AgentDeployment, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"Deployment '{name}' not found in workspace '{workspace}'."
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    session = await _resolve_request_session(request, workspace, entity_client, effective_principal_id)
    if session is not None and session.deployment_id != dep.id:
        raise _session_deployment_mismatch(
            session,
            detail=f"it does not belong to requested deployment '{name}'",
        )

    return await _proxy_deployment(
        request,
        dep,
        trailing_uri,
        model_name=name,
        entity_client=entity_client,
        session=session,
    )


async def _proxy_deployment(
    request: Request,
    deployment: AgentDeployment,
    trailing_uri: str,
    *,
    model_name: str,
    entity_client: NemoEntitiesClient,
    session: AgentSession | None = None,
) -> StreamingResponse:
    """Validate and proxy to an already-resolved deployment entity."""
    if not is_deployment_routable(deployment):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Deployment '{deployment.name}' is not routable "
                f"(mode='{deployment.deployment_mode}', status='{deployment.status}')."
            ),
        )

    endpoint = get_deployment_endpoint(deployment)
    # is_deployment_routable guarantees a non-None endpoint, but narrow for the type checker.
    if endpoint is None:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=503,
            detail=f"Deployment '{deployment.name}' has no routable endpoint (status='{deployment.status}').",
        )

    activity_session = session

    async def persist_start_activity() -> None:
        nonlocal activity_session
        if activity_session is not None:
            activity_session = await _require_session_activity_update(entity_client, activity_session)

    async def persist_finish_activity() -> None:
        if activity_session is not None:
            await _best_effort_session_activity_update(entity_client, activity_session)

    tracks_session_activity = session is not None and request.method.upper() in _PROXY_WRITE_METHODS

    return await _proxy(
        request,
        endpoint,
        trailing_uri,
        model_name=model_name,
        session_id=session.id if session is not None else None,
        transport=get_deployment_transport(deployment),
        on_start=persist_start_activity if tracks_session_activity else None,
        on_complete=persist_finish_activity if tracks_session_activity else None,
    )


@router.api_route(
    "/deployments/{name}/-/{trailing_uri:path}",
    methods=_PROXY_READ_METHODS,
    tags=["Agent Gateway"],
    include_in_schema=False,
)
@scope.read
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[GatewayPerms.INVOKE],
)
async def proxy_by_deployment_name_read(
    workspace: str,
    name: str,
    trailing_uri: str,
    request: Request,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
    effective_principal_id: str = Depends(get_effective_principal_id),
) -> Response:
    """Read-scoped (GET/HEAD/OPTIONS) proxy directly to the named deployment."""
    return await _serve_deployment_proxy(
        workspace,
        name,
        trailing_uri,
        request,
        entity_client,
        effective_principal_id,
    )


@router.api_route(
    "/deployments/{name}/-/{trailing_uri:path}",
    methods=_PROXY_WRITE_METHODS,
    tags=["Agent Gateway"],
    include_in_schema=False,
)
@scope.write
@path_rule(
    callers=[CallerKind.PRINCIPAL],
    permissions=[GatewayPerms.INVOKE],
)
async def proxy_by_deployment_name_write(
    workspace: str,
    name: str,
    trailing_uri: str,
    request: Request,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
    effective_principal_id: str = Depends(get_effective_principal_id),
) -> Response:
    """Write-scoped (POST/PUT/PATCH/DELETE) proxy directly to the named deployment."""
    return await _serve_deployment_proxy(
        workspace,
        name,
        trailing_uri,
        request,
        entity_client,
        effective_principal_id,
    )


async def _resolve_agent_deployment(
    name: str,
    workspace: str,
    entity_client: NemoEntitiesClient,
) -> AgentDeployment:
    """Find the first routable deployment entity for the given agent."""
    try:
        await entity_client.get(Agent, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found in workspace '{workspace}'.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        result = await entity_client.list(AgentDeployment, workspace=workspace)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    running = [d for d in result.data if d.agent == name and is_deployment_routable(d)]
    if not running:
        raise HTTPException(
            status_code=503,
            detail=f"No running deployment found for agent '{name}' in workspace '{workspace}'.",
        )
    # first-match, no load-balancing across running deployments (out of scope).
    return running[0]


async def _resolve_request_session(
    request: Request,
    workspace: str,
    entity_client: NemoEntitiesClient,
    effective_principal_id: str,
) -> AgentSession | None:
    """Resolve and validate the persisted session supplied on a gateway request."""
    session_id = request.headers.get(SESSION_ID_HEADER)
    if session_id is None:
        return None
    if not session_id:
        raise HTTPException(status_code=400, detail=f"Header '{SESSION_ID_HEADER}' must not be empty.")

    session = await get_owned_session_by_id(
        entity_client,
        workspace=workspace,
        session_id=session_id,
        effective_principal_id=effective_principal_id,
    )
    if session.status is not SessionStatus.ACTIVE:
        raise _session_inactive(session)
    if session.expires_at is not None and session_expiration_is_due(session, at=_utc_now()):
        raise _session_expired(session)
    return session


async def _resolve_session_deployment(
    session: AgentSession,
    workspace: str,
    entity_client: NemoEntitiesClient,
) -> AgentDeployment:
    """Resolve the exact deployment bound to a validated session."""
    try:
        deployment = await entity_client.find_one(
            AgentDeployment,
            workspace=workspace,
            filter_obj={"id": session.deployment_id},
        )
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Deployment ID '{session.deployment_id}' for session ID "
                f"'{session.id}' was not found in workspace '{workspace}'."
            ),
        ) from exc
    except Exception as exc:
        logger.exception("Failed to look up deployment ID '%s'", session.deployment_id)
        raise HTTPException(status_code=500, detail="Failed to look up deployment.") from exc

    if deployment.id != session.deployment_id or deployment.workspace != workspace:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Deployment ID '{session.deployment_id}' for session ID "
                f"'{session.id}' was not found in workspace '{workspace}'."
            ),
        )
    return deployment


def _session_deployment_mismatch(session: AgentSession, *, detail: str) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail=f"Session ID '{session.id}' is bound to deployment ID '{session.deployment_id}'; {detail}.",
    )


def _session_inactive(session: AgentSession) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail=(f"Session ID '{session.id}' has status '{session.status.value}' and cannot be invoked."),
    )


def _session_expired(session: AgentSession) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail=f"Session ID '{session.id}' has expired and cannot be invoked.",
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


async def _persist_session_activity(
    entity_client: NemoEntitiesClient,
    session: AgentSession,
    *,
    activity_at: datetime | None = None,
) -> AgentSession:
    """Advance persisted activity timestamps, retrying one optimistic conflict."""
    timestamp = (activity_at or _utc_now()).astimezone(UTC)
    session_to_update = session

    for attempt in range(2):
        if session_to_update.status is not SessionStatus.ACTIVE:
            raise _session_inactive(session_to_update)
        if session_expiration_is_due(session_to_update, at=timestamp):
            raise _session_expired(session_to_update)

        last_active_at = session_to_update.last_active_at
        if session_to_update.first_active_at is None and last_active_at is None:
            session_to_update.first_active_at = timestamp
        if last_active_at is not None:
            if last_active_at.tzinfo is None or last_active_at.utcoffset() is None:
                last_active_at = last_active_at.replace(tzinfo=UTC)
            if last_active_at >= timestamp:
                return session_to_update

        session_to_update.last_active_at = timestamp
        session_to_update.expires_at = timestamp + _SESSION_IDLE_TIMEOUT
        try:
            return await entity_client.update(session_to_update)
        except NemoEntityConflictError:
            if attempt == 1:
                raise
            session_to_update = await entity_client.get_by_id(AgentSession, session.id)

    raise RuntimeError("Session activity update retry loop exited unexpectedly.")  # pragma: no cover


async def _require_session_activity_update(
    entity_client: NemoEntitiesClient,
    session: AgentSession,
) -> AgentSession:
    """Persist required invocation-start activity or reject the invocation."""
    try:
        return await _persist_session_activity(entity_client, session)
    except HTTPException:
        raise
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Session ID '{session.id}' was deleted before invocation.",
        ) from exc
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Session ID '{session.id}' is being modified concurrently.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to persist invocation-start activity for session ID '%s'.", session.id)
        raise HTTPException(
            status_code=503,
            detail=f"Could not persist activity for session ID '{session.id}'.",
        ) from exc


async def _best_effort_session_activity_update(
    entity_client: NemoEntitiesClient,
    session: AgentSession,
) -> None:
    """Persist invocation-finish activity without changing an emitted response."""
    try:
        await _persist_session_activity(entity_client, session)
    except HTTPException:
        logger.debug(
            "Session ID '%s' became non-active before finish activity was persisted.",
            session.id,
        )
    except NemoEntityNotFoundError:
        logger.debug("Session ID '%s' was deleted before finish activity was persisted.", session.id)
    except Exception:
        logger.warning(
            "Failed to persist invocation-finish activity for session ID '%s'.",
            session.id,
            exc_info=True,
        )


async def _proxy(
    request: Request,
    endpoint: str,
    trailing_uri: str,
    *,
    model_name: str | None = None,
    session_id: str | None = None,
    transport: EndpointTransport = DIRECT,
    on_start: Callable[[], Awaitable[None]] | None = None,
    on_complete: Callable[[], Awaitable[None]] | None = None,
) -> StreamingResponse:
    """Forward *request* to ``{endpoint}/{trailing_uri}`` and stream the response.

    *transport* picks the address actually dialed for that URL; the origin guard below
    runs on the advertised URL, before that swap.

    Error handling policy:
    - **4xx** from the agent: status and body passed through (client error, agent's
      response). On the OpenAI-compatible surface a body that no OpenAI client could
      read gains an ``error`` envelope beside its existing keys; one that already
      carries ``error.message`` is forwarded untouched.
    - **5xx** from the agent: translated to **502 Bad Gateway** (upstream fault).
    - **Connection failure** (httpx.RequestError): 502 Bad Gateway.

    All responses are streamed, including SSE (``text/event-stream``).
    ``content-length`` is stripped from forwarded headers because chunked
    transfer encoding makes the original value invalid.
    """
    # The SSRF guard below is origin-relative, not a host allow-list: it only
    # rejects a trailing_uri that escapes the *resolved* endpoint's origin. That
    # means it works unchanged for container-mode targets — a k8s in-cluster
    # Service DNS name (``<svc>.<ns>.svc.cluster.local:<port>``) or a docker
    # host:port — exactly as it does for subprocess loopback, as long as the
    # resolved endpoint is a well-formed ``scheme://netloc`` (checked next).
    endpoint_parsed = urlparse(endpoint)
    if not endpoint_parsed.scheme or not endpoint_parsed.netloc:
        raise HTTPException(status_code=500, detail="Deployment endpoint is misconfigured.")
    joined = urlparse(urljoin(endpoint.rstrip("/") + "/", trailing_uri))
    if joined.scheme != endpoint_parsed.scheme or joined.netloc != endpoint_parsed.netloc:
        raise HTTPException(status_code=400, detail="Invalid proxy target URI.")
    target_url = urlunparse(
        (endpoint_parsed.scheme, endpoint_parsed.netloc, joined.path, joined.params, joined.query, "")
    )
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"
    target_url, routing_headers = transport.target(target_url)

    # Build forwarded headers — strip hop-by-hop and platform-internal headers
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _REQUEST_HEADERS_TO_STRIP}
    headers.update(routing_headers)
    if session_id is not None:
        headers[SESSION_ID_HEADER] = session_id

    body = await request.body()

    activity_started = False
    activity_completed = False
    if on_start is not None:
        await on_start()
        activity_started = True

    async def _complete_activity_once() -> None:
        nonlocal activity_completed
        if not activity_started or activity_completed or on_complete is None:
            return
        activity_completed = True
        await on_complete()

    # We need the upstream response headers before we can construct StreamingResponse
    # (to forward content-type, etc.).  Use a two-phase approach:
    # 1. Open the stream and capture headers — this triggers the HTTP round-trip.
    # 2. Prime the generator with one __anext__() call so headers are populated.
    # 3. Wrap in _buffered() to re-yield the primed chunk before continuing the stream.
    response_headers: dict[str, str] = {}
    status_code_holder: list[int] = [200]

    async def _stream_with_headers() -> AsyncIterator[bytes]:
        read_timeout = float(os.environ.get("NEMO_AGENTS_GATEWAY_READ_TIMEOUT", "300"))
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=read_timeout, write=60.0, pool=10.0),
            # SSRF defense in depth: never let an agent's 3xx response redirect
            # us off the validated origin.
            follow_redirects=False,
            **transport.client_kwargs(),
        ) as client:
            async with client.stream(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
            ) as response:
                status_code_holder[0] = response.status_code
                for k, v in response.headers.items():
                    if k.lower() not in _RESPONSE_HEADERS_TO_STRIP:
                        response_headers[k] = v
                # Translate agent 5xx responses into 502 Bad Gateway.
                # aread() consumes the full body before raising so the connection
                # is cleanly closed rather than reset mid-stream.
                if response.status_code >= 500:
                    raise UpstreamAgentError(response.status_code, await response.aread())
                if response.status_code >= 400 and is_openai_compatible_uri(trailing_uri):
                    error_body = await response.aread()
                    reshaped = augment_upstream_error_body(error_body, response.status_code)
                    if reshaped is None:
                        yield error_body
                    else:
                        response_headers["content-type"] = "application/json"
                        yield reshaped
                    return
                async for chunk in response.aiter_bytes():
                    yield chunk

    stream_gen = _stream_with_headers()
    chunks: list[bytes] = []

    async def _buffered() -> AsyncIterator[bytes]:
        try:
            for c in chunks:
                yield c
            async for c in stream_gen:
                yield c
        finally:
            await _complete_activity_once()

    # Prime: triggers the HTTP request, populates response_headers / status_code_holder,
    # and catches the most common failure modes before we commit to a StreamingResponse.
    try:
        first_chunk = await stream_gen.__anext__()
        chunks.append(first_chunk)
    except StopAsyncIteration:
        pass  # empty body — still valid (e.g. 204)
    except HTTPException:
        await _complete_activity_once()
        raise  # 5xx → 502 translation raised inside the generator; propagate as-is
    except httpx.RequestError as exc:
        await _complete_activity_once()
        raise HTTPException(status_code=502, detail=f"Could not connect to agent: {exc}") from exc
    except BaseException:
        await _complete_activity_once()
        raise

    content_type = response_headers.get("content-type", "application/json")

    # NAT's ChatResponse.from_string() defaults model to "unknown-model" when
    # the agent wrapper doesn't supply one (the wrapper code lives in
    # nvidia-nat-core and doesn't have access to the platform entity name).
    # For non-streaming JSON responses, patch the model field to the
    # agent/deployment name the client addressed.  This is a gateway-level
    # workaround; the proper upstream fix belongs in nvidia-nat-core's
    # NemoAgentWrapperFunction.convert_to_chat_response where the LLM's
    # response_metadata carries the real model name.
    if model_name and not content_type.startswith("text/event-stream"):
        try:
            async for remaining in stream_gen:
                chunks.append(remaining)
        except BaseException:
            await _complete_activity_once()
            raise
        raw = b"".join(chunks)
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("model") == "unknown-model":
                data["model"] = model_name
                raw = json.dumps(data).encode()
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        chunks = [raw]

    return StreamingResponse(
        _buffered(),
        status_code=status_code_holder[0],
        headers={k: v for k, v in response_headers.items() if k.lower() != "content-length"},
        media_type=content_type,
    )
