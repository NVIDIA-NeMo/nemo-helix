# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auth client for NeMo Platform services."""

import asyncio
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional

import httpx
from nmp.common.config import AuthConfig
from nmp.common.platform_endpoint import parse_platform_endpoint
from pydantic import BaseModel, Field

from .authz_format import validate_permission_strings, validate_runtime_authorize_scopes
from .exceptions import InvalidPermissionFormatError
from .models import Principal
from .token_resolver import ResolvedBearerToken

logger = logging.getLogger(__name__)


@dataclass
class AuthorizationResult:
    """Result of an authorization check.

    Attributes:
        allowed: Whether the request is authorized
        reason: Optional explanation for the decision (e.g., "health_check", "platform_admin")
    """

    allowed: bool
    reason: Optional[str] = None


class AuthClient(BaseModel):
    """Client for authorization operations.

    This model provides the complete authorization context including the principal
    and methods for permission and role checks. All PDP (Policy Decision Point)
    calls are centralized here.

    Methods:
        authorize_request: Check if a request (method + path) is allowed
        has_permissions: Check if principal has specific permissions in a workspace
        wait_role: Poll until a principal has (or lacks) a role (PDP ``has_role`` entrypoint)
    """

    principal: "Principal" = Field(..., description="The authenticated principal for this request")
    config: "AuthConfig" = Field(..., description="Authorization configuration")
    http_client: Optional[httpx.AsyncClient] = Field(
        default=None,
        description="HTTP client for PDP calls. Injected by middleware for ASGI transport in tests.",
    )
    service_name: Optional[str] = Field(
        default=None,
        description="Name of the calling service. Used to build service principal headers for PDP requests.",
    )
    resolved_bearer_token: Optional[ResolvedBearerToken] = Field(
        default=None,
        description=(
            "The trusted bearer token this request was authenticated with, when authenticated via a "
            "Bearer token (as opposed to internal principal headers). Lets handlers distinguish a caller "
            "authenticated via a Scoped Access Key from an ordinary OIDC session and read its validated "
            "scope claims, e.g. to prevent a scope-restricted access key from minting a broader one."
        ),
    )

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": False}

    @property
    def auth_enabled(self) -> bool:
        """Whether authentication is enabled."""
        return self.config.enabled

    @property
    def policy_decision_point_base_url(self) -> Optional[str]:
        """Policy Decision Point (PDP) base URL for permission checks."""
        return self.config.policy_decision_point_base_url

    def _new_pdp_http_client(self) -> httpx.AsyncClient:
        endpoint = parse_platform_endpoint(self.config.policy_decision_point_base_url)
        return endpoint.async_http_client(timeout=self.config.policy_decision_point_request_timeout_seconds)

    @property
    def _pdp_request_headers(self) -> dict[str, str]:
        """Headers sent with every PDP HTTP request.

        Combines the internal-request marker (suppresses access logging) with a
        service principal so the receiving middleware auto-authorises the call
        instead of recursing back into the PDP.
        """
        from nmp.common.observability import MARK_INTERNAL_REQUEST_HEADERS

        return {
            **MARK_INTERNAL_REQUEST_HEADERS,
            "X-NMP-Principal-Id": f"service:{self.service_name or 'unknown'}",
        }

    async def authorize_request(
        self,
        method: str,
        path: str,
        scopes: Optional[List[str]] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> AuthorizationResult:
        """Check if the current principal is authorized to make a request.

        This is the main authorization entrypoint, calling the PDP's "allow" endpoint
        to determine if a request should proceed.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            path: Request path (e.g., "/v2/workspaces/my-ws/models")
            scopes: Optional OAuth2 scopes from the token
            http_client: Optional pre-configured HTTP client (for connection reuse)

        Returns:
            AuthorizationResult with allowed=True/False and optional reason

        Raises:
            InvalidScopeFormatError: If a scope string looks like a permission (dot-separated).
            httpx.ConnectError: If PDP cannot be reached
            httpx.TimeoutException: If PDP times out
            httpx.HTTPStatusError: If PDP returns an error response

        Example:
            ```python
            result = await auth_client.authorize_request("GET", "/v2/workspaces/my-ws/models")
            if not result.allowed:
                raise HTTPException(status_code=403, detail="Forbidden")
            ```
        """
        validate_runtime_authorize_scopes(scopes)

        if not self.auth_enabled:
            logger.debug("Auth disabled, allowing request %s %s", method, path)
            return AuthorizationResult(allowed=True, reason="auth_disabled")

        if not self.policy_decision_point_base_url:
            raise RuntimeError("Policy Decision Point URL not configured")

        # Build authorization input (email/groups reflect the acting user when delegating)
        auth_input = {
            "principal_id": self.principal.id,
            "method": method,
            "path": path,
        }
        if self.principal.effective_email:
            auth_input["principal_email"] = self.principal.effective_email
        if self.principal.effective_groups:
            auth_input["principal_groups"] = self.principal.effective_groups
        if self.principal.on_behalf_of:
            auth_input["on_behalf_of_principal_id"] = self.principal.on_behalf_of
        if scopes:
            auth_input["scopes"] = scopes

        auth_url = self.config.get_pdp_url("allow")

        logger.debug("Calling PDP: url=%s, input=%s", auth_url, auth_input)

        pdp_headers = self._pdp_request_headers

        # Use provided http_client, instance http_client (from middleware), or create a new one
        # See architecture/docs/http-client-injection.md for injection patterns.
        client = http_client or self.http_client
        if client:
            response = await client.post(auth_url, json={"input": auth_input}, headers=pdp_headers)
        else:
            async with self._new_pdp_http_client() as temp_client:
                response = await temp_client.post(auth_url, json={"input": auth_input}, headers=pdp_headers)

        response.raise_for_status()

        result = response.json().get("result", {})
        allowed = result.get("allowed", False)
        reason = result.get("reason")

        logger.debug(
            "PDP response: method=%s, path=%s, allowed=%s, reason=%s",
            method,
            path,
            allowed,
            reason,
        )

        return AuthorizationResult(allowed=allowed, reason=reason)

    async def _pdp_check(self, pdp_endpoint: str, auth_input: dict, result_key: str, check_name: str) -> bool:
        """Send a single-shot PDP check and extract a boolean result.

        Shared by has_permissions/has_role: both open/close the PDP http client the
        same way and map httpx errors to the same HTTPException pair, differing only
        in the endpoint, request payload, and which result field holds the answer.

        ``check_name`` is a lowercase phrase (e.g. "permission check", "role check")
        used to build both the log message and the HTTPException detail, matching the
        wording each caller used before this helper was factored out.
        """
        from fastapi import HTTPException

        client = self.http_client
        should_close = False
        if client is None:
            client = self._new_pdp_http_client()
            should_close = True

        try:
            response = await client.post(
                self.config.get_pdp_url(pdp_endpoint),
                json={"input": auth_input},
                headers=self._pdp_request_headers,
            )
            response.raise_for_status()
            return bool(response.json().get("result", {}).get(result_key, False))
        except httpx.HTTPError as exc:
            logger.error("Failed to contact auth endpoint for %s: %s", check_name, str(exc))
            raise HTTPException(status_code=503, detail=f"{check_name.capitalize()} service unavailable") from exc
        except Exception as exc:
            logger.error("Unexpected error during %s: %s", check_name, str(exc))
            raise HTTPException(status_code=500, detail=f"Internal {check_name} error") from exc
        finally:
            if should_close:
                await client.aclose()

    async def has_permissions(self, workspace_id: str, permissions: List[str]) -> bool:
        """Check if the current principal has specific permissions in a workspace.

        This method calls the auth endpoint to check if the principal has all the required
        permissions in the specified workspace.

        Args:
            workspace_id: The workspace to check permissions in
            permissions: List of permission strings to check

        Returns:
            True if principal has all required permissions, False otherwise

        Raises:
            InvalidPermissionFormatError: If a permission string uses scope syntax (colons) or is malformed.
            RuntimeError: If auth endpoint is not configured or call fails

        Example:
            ```python
            @router.post("/v2/workspaces/{workspace_id}/models")
            async def create_model(
                workspace_id: str,
                model: ModelCreate,
                auth_client: AuthClient = Depends(get_auth_client)
            ):
                if await auth_client.has_permissions(workspace_id, ["models.create"]):
                    # Allow model creation
                    pass
            ```
        """
        validate_permission_strings(permissions, context="AuthClient.has_permissions")

        # If auth is disabled, allow all permissions
        if not self.auth_enabled:
            logger.debug("Auth disabled, allowing all permissions")
            return True

        if not self.policy_decision_point_base_url:
            raise RuntimeError("Policy Decision Point URL not configured for permission checks")

        # Prepare input with all principal identifiers (acting user when delegating)
        auth_input = {
            "principal_id": self.principal.id,
            "workspace": workspace_id,
            "permissions": permissions,
        }
        if self.principal.effective_email:
            auth_input["principal_email"] = self.principal.effective_email
        if self.principal.effective_groups:
            auth_input["principal_groups"] = self.principal.effective_groups
        if self.principal.on_behalf_of:
            auth_input["on_behalf_of_principal_id"] = self.principal.on_behalf_of

        allowed = await self._pdp_check("has_permissions", auth_input, "allowed", "permission check")

        logger.info(
            "Permission check for principal=%s workspace=%s permissions=%s: %s",
            self.principal.id,
            workspace_id,
            permissions,
            "ALLOWED" if allowed else "DENIED",
        )

        return allowed

    async def has_role(self, workspace_id: str, role: str) -> bool:
        """Check whether the current effective principal has a role in a workspace."""
        if not self.auth_enabled:
            return True
        if not self.policy_decision_point_base_url:
            raise RuntimeError("Policy Decision Point URL not configured for role checks")

        auth_input = {
            "principal_id": self.principal.effective_id,
            "workspace": workspace_id,
            "role": role,
        }
        if self.principal.effective_email:
            auth_input["principal_email"] = self.principal.effective_email
        if self.principal.effective_groups:
            auth_input["principal_groups"] = self.principal.effective_groups
        if self.principal.on_behalf_of:
            auth_input["on_behalf_of_principal_id"] = self.principal.on_behalf_of

        return await self._pdp_check("has_role", auth_input, "has_role", "role check")

    async def on_behalf_of_has_permissions(self, workspace_id: str, permissions: List[str]) -> bool:
        """Check if the on-behalf-of principal has specific permissions in a workspace.

        This is a convenience method to check permissions for delegated users
        when the current principal has an on-behalf-of principal field set.

        Args:
            workspace_id: The workspace to check permissions in
            permissions: List of permission strings to check
        Returns:
            True if auth not enabled, or on-behalf-of principal has all required permissions, False otherwise
        """
        validate_permission_strings(permissions, context="AuthClient.on_behalf_of_has_permissions")

        # If auth is disabled, allow all permissions
        if not self.auth_enabled:
            logger.debug("Auth disabled, allowing all permissions on behalf of principal")
            return True

        delegated_auth_client = AuthClient(
            principal=self.principal.to_on_behalf_of(), config=self.config, http_client=self.http_client
        )
        return await delegated_auth_client.has_permissions(workspace_id, permissions)

    async def wait_permissions(
        self,
        workspace_id: str,
        permissions: List[str],
        timeout: float = 30.0,
        poll_interval: float | None = None,
    ) -> bool:
        """Wait for the principal to have specific permissions in a workspace.

        This method polls the auth endpoint repeatedly until the principal has the required
        permissions or the timeout is reached.

        Args:
            workspace_id: The workspace to check permissions in
            permissions: List of permission strings to check
            timeout: Maximum time to wait in seconds (default: 30.0)
            poll_interval: Time between checks in seconds (default: 1.0)

        Returns:
            True if principal has all required permissions within timeout, False otherwise

        Example:
            ```python
            @router.post("/v2/workspaces")
            async def create_workspace(
                workspace: WorkspaceCreate,
                auth_client: AuthClient = Depends(get_auth_client)
            ):
                # Wait for workspace admin permissions after creation
                if await auth_client.wait_permissions("new-workspace", ["workspaces.manage_members"]):
                    logger.info("Permissions granted for new workspace")
                else:
                    logger.warning("Timeout waiting for workspace permissions")
            ```
        """
        if poll_interval is None:
            poll_interval = self.config.propagation_poll_interval_seconds
        start_time = asyncio.get_event_loop().time()
        end_time = start_time + timeout

        while asyncio.get_event_loop().time() < end_time:
            try:
                if await self.has_permissions(workspace_id, permissions):
                    logger.debug(
                        "Permissions granted after %.2f seconds",
                        asyncio.get_event_loop().time() - start_time,
                    )
                    return True
            except InvalidPermissionFormatError:
                raise
            except Exception as e:
                # Log but continue waiting
                logger.debug("Permission check failed during wait: %s", str(e))

            # Wait before next check
            await asyncio.sleep(poll_interval)

        # Timeout reached
        logger.warning(
            "Timeout (%.2f seconds) reached waiting for permissions",
            timeout,
        )
        return False

    async def wait_role(
        self,
        principal: str,
        workspace_id: str,
        role: str,
        is_present: bool = True,
        timeout: float = 30.0,
        poll_interval: float | None = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> bool:
        """Wait for a principal to have (or not have) a specific role in a workspace.

        This method polls the auth endpoint repeatedly until the principal has the required
        role state or the timeout is reached.

        Args:
            principal: The principal ID to check roles for (e.g., "user@example.com")
            workspace_id: The workspace to check the role in
            role: The role to check for (e.g., "Admin", "Editor", "Viewer")
            is_present: If True, waits for role to be granted. If False, waits for role to be revoked
            timeout: Maximum time to wait in seconds (default: 30.0)
            poll_interval: Time between checks in seconds (default: 1.0)
            http_client: Optional HTTP client for test injection via DependencyProvider.
                        See architecture/docs/http-client-injection.md.

        Returns:
            True if principal has the required role state within timeout, False otherwise

        Example:
            ```python
            # Wait for role to be granted
            if await auth_client.wait_role("user@example.com", workspace_id, "Editor"):
                logger.info("Role granted")

            # Wait for role to be revoked
            if await auth_client.wait_role("user@example.com", workspace_id, "Editor", is_present=False):
                logger.info("Role revoked")
            ```
        """
        if not self.auth_enabled:
            return True

        if not self.policy_decision_point_base_url:
            raise RuntimeError("Policy Decision Point URL not configured for role checks")

        if poll_interval is None:
            poll_interval = self.config.propagation_poll_interval_seconds
        start_time = asyncio.get_event_loop().time()
        end_time = start_time + timeout

        # Use provided http_client, instance http_client (from middleware), or create a new one
        # See architecture/docs/http-client-injection.md for injection patterns.
        client = http_client or self.http_client or self._new_pdp_http_client()
        should_close = http_client is None and self.http_client is None

        try:
            while asyncio.get_event_loop().time() < end_time:
                try:
                    # Query auth endpoint for role check
                    auth_url = self.config.get_pdp_url("has_role")
                    payload = {
                        "input": {
                            "principal_id": principal,
                            "workspace": workspace_id,
                            "role": role,
                        }
                    }

                    response = await client.post(auth_url, json=payload, headers=self._pdp_request_headers)
                    response.raise_for_status()
                    result = response.json()
                    has_role = result.get("result", {}).get("has_role", False)

                    # Check if the role state matches what we're waiting for
                    if has_role == is_present:
                        action = "granted" if is_present else "revoked"
                        logger.debug(
                            "Role '%s' %s for principal '%s' in workspace '%s' after %.2f seconds",
                            role,
                            action,
                            principal,
                            workspace_id,
                            asyncio.get_event_loop().time() - start_time,
                        )
                        return True
                except Exception as e:
                    # Log but continue waiting
                    logger.debug("Role check failed during wait: %s", str(e))

                # Wait before next check
                await asyncio.sleep(poll_interval)
        finally:
            if should_close:
                await client.aclose()

        # Timeout reached
        action = "granted" if is_present else "revoked"
        logger.warning(
            "Timeout (%.2f seconds) reached waiting for role '%s' to be %s for principal '%s' in workspace '%s'",
            timeout,
            role,
            action,
            principal,
            workspace_id,
        )
        return False

    @contextmanager
    def as_service(self):
        """Temporarily impersonate a service principal for internal operations.

        This context manager temporarily changes the principal ID to a service principal
        (prefixed with "service:") which has unrestricted access in the OPA policy.

        Example:
            ```python
            # Check if namespace exists without requiring user permissions
            with auth_client.as_service():
                existing_obj = await db.get(Namespace, id=namespace_id)
            ```
        """
        # Import here to avoid circular dependency
        from .dependencies import auth_client_context

        # Save the current principal
        original_principal = self.principal
        current_context = auth_client_context.get(None)

        try:
            # Create a service principal (use "unknown" as service name since
            # AuthConfig doesn't track which service is running)
            service_principal = Principal(
                id="service:unknown",
                email=None,
                groups=[],
            )

            # Update the principal in this AuthClient
            self.principal = service_principal

            # Also update the context variable if present
            if current_context is not None:
                current_context.principal = service_principal
                auth_client_context.set(current_context)

            yield

        finally:
            # Restore the original principal
            self.principal = original_principal

            # Restore context if present
            if current_context is not None:
                current_context.principal = original_principal
                auth_client_context.set(current_context)
