# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI dependencies and utilities for authentication."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Dict, Generator, Optional

from fastapi import HTTPException, Request

if TYPE_CHECKING:
    from .client import AuthClient

# Context variable to store the current auth client (set by middleware or tasks runtime)
auth_client_context: ContextVar[Optional["AuthClient"]] = ContextVar("auth_client_context", default=None)


def get_auth_client(request: Request) -> "AuthClient":
    """Get the authorization client for the current request.

    This is the primary FastAPI dependency for all auth-related needs.
    It provides:

    - auth_client.principal: The authenticated principal (id, email, groups)
    - auth_client.authorize_request(method, path): Check if request is allowed
    - auth_client.has_permissions(workspace, perms): Check if principal has permissions
    - auth_client.wait_permissions(workspace, perms): Poll until permissions granted
    - auth_client.wait_role(principal, workspace, role): Poll until role granted/revoked

    Args:
        request: The FastAPI request object (unused, kept for FastAPI Depends signature)

    Returns:
        The AuthClient object with principal and authorization methods

    Raises:
        HTTPException: If no auth context is found (middleware not configured)

    Example:
        ```python
        @router.get("/v2/workspaces/{workspace}/models")
        async def list_models(
            workspace: str,
            auth_client: AuthClient = Depends(get_auth_client)
        ):
            # Access principal
            principal_id = auth_client.principal.id

            # Check permissions
            if await auth_client.has_permissions(workspace, ["models.read"]):
                ...
        ```
    """
    auth_client = auth_client_context.get()
    if auth_client is None:
        raise HTTPException(
            status_code=500,
            detail="No auth context found. Authorization middleware may not be configured.",
        )
    return auth_client


def get_principal_auth_headers() -> Dict[str, str]:
    """Get principal authentication headers from the current auth context.

    This is a utility function for services that need to forward principal
    authentication headers when making HTTP calls to other NeMo Helix services.
    It retrieves the current auth client from the context variable and
    extracts the principal headers for identity propagation.

    Returns:
        Dictionary of principal authentication headers to forward, or empty dict
        if no auth context. Headers include:

        - X-NHX-Principal-Id: The principal's unique identifier
        - X-NHX-Principal-Email: The principal's email (if available)
        - X-NHX-Principal-Groups: Comma-separated list of groups (if available)
        - X-NHX-Principal-On-Behalf-Of: On-behalf-of principal (if available)
        - X-NHX-Principal-On-Behalf-Of-Groups: On-behalf-of principal groups (if available)
        - X-NHX-Principal-On-Behalf-Of-Email: On-behalf-of principal email (if available)

    Example:
        ```python
        import httpx
        from nhx.common.auth import get_principal_auth_headers

        async def fetch_from_entities_service(resource_id: str):
            headers = get_principal_auth_headers()
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{entities_url}/v2/models/{resource_id}",
                    headers=headers
                )
                return response.json()
        ```
    """
    auth_client = auth_client_context.get()
    if auth_client and auth_client.principal:
        return auth_client.principal.get_headers()
    return {}


def build_service_principal_headers(service_name: str) -> Dict[str, str]:
    """Build NeMo Helix auth headers for outbound service-to-service calls.

    Returns:
    - `X-NHX-Principal-Id: service:<service_name>` so the downstream service
      can authorize the call.
    - When the current auth context is a non-service principal, also forwards
      `X-NHX-Principal-On-Behalf-Of`, `-Email`, and `-Groups` from
      ``Principal.effective_principal`` so downstream PDP checks evaluate the
      acting user (not the elevated service row alone).

    Args:
        service_name: The calling service's name (ex. "guardrails").

    Returns:
        Header dictionary ready to merge into an outbound request.
    """
    headers: Dict[str, str] = {
        "X-NHX-Principal-Id": f"service:{service_name}",
        "X-NHX-Actor-Aliases": f"service:{service_name}",
    }

    auth_client = auth_client_context.get()
    if auth_client is None or not auth_client.principal or not auth_client.principal.id:
        return headers

    effective = auth_client.principal.effective_principal
    if effective.caller_kind == "service_principal":
        return headers

    headers["X-NHX-Principal-On-Behalf-Of"] = effective.id
    if effective.email:
        headers["X-NHX-Principal-On-Behalf-Of-Email"] = effective.email
    if effective.groups:
        headers["X-NHX-Principal-On-Behalf-Of-Groups"] = ",".join(effective.groups)
    if effective.account_id:
        headers["X-NHX-Subject-Account-Id"] = effective.account_id
    if effective.authz_aliases:
        headers["X-NHX-Subject-Aliases"] = ",".join(effective.authz_aliases)

    return headers


@contextmanager
def auth_as_service(service: Optional[str] = None) -> Generator[None, None, None]:
    """Context manager to run code with service principal credentials.

    Creates a new AuthClient with service principal credentials and sets it
    in the context variable. This isolates the elevated credentials to the
    current async context without affecting concurrent code.

    The service principal (e.g., "service:auth") has elevated permissions in the
    OPA policy, allowing internal service-to-service calls.

    This can be used both within request handlers (where auth context exists)
    and in background tasks/startup code (where it creates a fresh context).

    Note:
        Currently all service principals are treated equally with full permissions.
        In the future, service principals may be restricted based on scope - e.g.,
        "service:evaluator" would not have access to data designer resources.

    Args:
        service: Service name for the principal (e.g., "evaluator").
                 Defaults to the current client's configured service name, or
                 "unknown" if no context exists and no service is specified.

    Example:
        ```python
        from nhx.common.auth import auth_as_service

        async def refresh_policy_data(entities_client):
            with auth_as_service():
                # Uses default service name from config
                data = await entities_client.list(RoleBindingEntity, ...)

        async def cross_service_call():
            with auth_as_service(service="evaluator"):
                # Uses "service:evaluator" as principal
                data = await other_client.fetch(...)
        ```
    """
    from nhx.common.config import get_auth_config

    from .client import AuthClient
    from .models import Principal

    # Get current auth client if it exists
    current_client = auth_client_context.get()

    # Determine service name and config
    if current_client is not None:
        auth_config = current_client.config
    else:
        # No existing context - create fresh config from global settings
        auth_config = get_auth_config()

    # Use provided service name, or default to "unknown"
    service_name = service if service is not None else "unknown"

    # Create new AuthClient with service principal
    service_principal = Principal(
        id=f"service:{service_name}",
        email=None,
        groups=[],
        authz_aliases=[f"service:{service_name}"],
    )
    service_client = AuthClient(principal=service_principal, config=auth_config)

    # Set new client in context (isolated to this async context)
    token = auth_client_context.set(service_client)
    try:
        yield
    finally:
        auth_client_context.reset(token)


def get_request_authorizer(request: Request):
    """Bind authorization to the authenticated caller and original scopes."""
    client = get_auth_client(request)
    if client.resolved_bearer_token is not None:
        scopes = client.resolved_bearer_token.scopes
    else:
        # Internal principal headers are authenticated by middleware before dependency resolution.
        scopes = request.headers.get("x-nhx-scopes", "").split() or None

    async def authorize(method: str, path: str) -> None:
        result = await client.authorize_request(method, path, scopes=scopes)
        if not result.allowed:
            raise HTTPException(status_code=403, detail="Access to taskset member denied")

    return authorize
