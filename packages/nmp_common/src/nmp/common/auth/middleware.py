# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorization middleware for NeMo Platform services."""

import logging
from typing import Any, Callable, Optional

import httpx
from fastapi import Request, Response
from nmp.common.config import AuthConfig, get_auth_config
from nmp.common.observability.context import get_app_ctx
from nmp.common.platform_endpoint import parse_platform_endpoint
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from .access_key_lifecycle import (
    AccessKeyLifecycleAuthenticator,
    AccessKeyLifecycleUnavailableError,
)
from .bearer import MalformedBearerTokenError, parse_bearer_authorization_header
from .client import AuthClient
from .dependencies import auth_client_context
from .exceptions import InvalidPrincipalHeader, InvalidScopeFormatError
from .models import Principal
from .token_resolver import ResolvedBearerToken, resolve_bearer_token

logger = logging.getLogger(__name__)


class PrincipalExtractionError(RuntimeError):
    """Raised when principal extraction returns an impossible success state."""


def _require_principal(principal: Principal | None) -> Principal:
    """Return a principal from a successful extraction result or fail loudly."""
    if principal is None:
        raise PrincipalExtractionError("principal extraction succeeded without returning a principal")
    return principal


def _describe_pdp_failure(exc: BaseException) -> str:
    """Best-effort description for PDP client errors (handles empty str(exc))."""
    msg = str(exc).strip()
    if msg:
        return f"{type(exc).__name__}: {msg}"
    parts = [f"{type(exc).__name__} (empty message)"]
    if exc.__cause__ is not None:
        parts.append(f"cause={type(exc.__cause__).__name__}: {exc.__cause__}")
    if isinstance(exc, httpx.RequestError):
        req = getattr(exc, "request", None)
        if req is not None:
            parts.append(f"request={req.method} {req.url}")
    parts.append(f"repr={exc!r}")
    return "; ".join(parts)


def _embedded_pdp_base_url_hint(config: AuthConfig) -> str:
    """Explain common misconfiguration when embedded PDP cannot be reached."""
    if config.policy_decision_point_provider != "embedded":
        return ""
    base = (config.policy_decision_point_base_url or "").strip()
    return (
        " For embedded PDP, auth.policy_decision_point_base_url must be the typed endpoint where "
        "this process serves /apis/auth (same as platform base_url / NMP_BASE_URL; HTTP(S) or unix://). "
        f"Absolute PDP URLs ignore the injected ASGI client base_url. Current auth.policy_decision_point_base_url={base!r}."
    )


# Health/metrics check endpoints - always allowed without authentication
HEALTH_ENDPOINTS = {
    "/status",
    "/cluster-info",
    "/health/live",
    "/health/ready",
    "/metrics",
    "/apis/auth/discovery",  # Discovery endpoint for CLI/SDK
    "/apis/auth/authenticate",  # Direct bearer-token validation JSON API
    "/apis/auth/ext-authz",  # Envoy bearer-token validation callout
    "/apis/auth/jwks",  # NeMo-minted bearer-token signing keys
    "/apis/auth/token",  # Workload identity token exchange validates the subject token itself
}

# GET requests to these paths bypass authentication (e.g. / -> /studio redirect).
PUBLIC_GET_PATHS = {
    "/",
    "/apis/plugins",  # Studio plugin manifest — fetched by the SPA before login completes
}

# Path prefixes that bypass authorization
BYPASS_PREFIXES = (
    "/apis/auth/ext-authz/",  # Envoy ext_authz path_prefix callout includes the original protected path
    "/studio",  # Studio UI static files — the SPA handles its own OIDC login
    "/plugin-ui/",  # Studio plugin bundles — loaded via dynamic import(), cannot send Authorization
)


class AuthorizationMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce authorization for all endpoints.

    This middleware:
    1. Intercepts requests to all endpoints
    2. Checks authorization with policy decision point endpoint
    3. Creates a Principal object from headers
    4. Sets up auth_client_context for dependency injection via get_auth_client()

    Configuration is read from the shared AuthConfig (auth: key in config.yaml).

    Usage:
        Services access the principal via AuthClient dependency:
        ```python
        @router.get("/example")
        async def example(auth_client: AuthClient = Depends(get_auth_client)):
            principal = auth_client.principal
        ```
    """

    def __init__(
        self,
        app: ASGIApp,
        service_name: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        access_key_lifecycle_http_client: Optional[httpx.AsyncClient] = None,
    ):
        """Initialize the authorization middleware.

        Args:
            app: The FastAPI/Starlette application.
            service_name: Name of this service (used for logging). Default: None.
            http_client: Optional HTTP client for PDP calls. If not provided,
                        one will be created lazily on first use. This is used for
                        testing with ASGI transport - see architecture/docs/http-client-injection.md.
            access_key_lifecycle_http_client: Optional HTTP client for access-key
                        lifecycle calls. If not provided, a client using the platform
                        endpoint transport is created lazily.
        """
        super().__init__(app)
        self.config: AuthConfig = get_auth_config()
        self.service_name = service_name
        self._client: Optional[httpx.AsyncClient] = http_client
        self._access_key_lifecycle = AccessKeyLifecycleAuthenticator(
            self.config,
            http_client=access_key_lifecycle_http_client,
        )
        self._jwt_validator: Optional[Any] = None

        if self.config.allow_unsigned_jwt:
            logger.warning(
                "auth.allow_unsigned_jwt is enabled. Unsigned JWTs (`alg=none`) are accepted; use only for local/testing."
            )

    @staticmethod
    def _principal_from_headers(headers_dict: dict) -> tuple[Principal, None] | tuple[None, JSONResponse]:
        """Extract and validate a Principal from request headers.

        Returns (principal, None) on success or (None, error_response) on validation failure.
        """
        try:
            principal = Principal.from_headers(headers_dict) or Principal()
        except InvalidPrincipalHeader as e:
            logger.warning("Invalid principal header: %s", e)
            return None, JSONResponse(status_code=400, content={"detail": str(e)})
        return principal, None

    def _get_jwt_validator(self) -> Optional[Any]:
        """Get JWT validator if OIDC is configured or unsigned JWTs are allowed."""
        if not self.config.oidc.enabled and not self.config.allow_unsigned_jwt:
            return None
        if self._jwt_validator is None:
            from .jwt import JWTValidator

            self._jwt_validator = JWTValidator(self.config)
        return self._jwt_validator

    def _get_client(self, request: Request) -> httpx.AsyncClient:
        """Get or create the HTTP client for PDP calls.

        Uses lazy initialization to avoid creating connections until needed.
        The client is reused across requests for connection pooling.

        Args:
            request: The current request (unused, kept for potential future use)

        Returns:
            An async HTTP client configured with auth.policy_decision_point_request_timeout_seconds
        """
        if self._client is None:
            endpoint = parse_platform_endpoint(self.config.policy_decision_point_base_url)
            self._client = endpoint.async_http_client(timeout=self.config.policy_decision_point_request_timeout_seconds)
        return self._client

    def _update_auth_context(self, principal: Principal) -> None:
        """Update the observability AuthContext with principal info.

        This is called after extracting principal info from Bearer tokens
        to ensure logging and tracing include the authenticated user info.
        """
        app_ctx = get_app_ctx()
        if app_ctx is not None and app_ctx.auth_ctx is not None:
            auth_ctx = app_ctx.auth_ctx
            auth_ctx.principal_id = principal.id
            auth_ctx.email = principal.email
            auth_ctx.groups = ",".join(principal.groups) if principal.groups else None
            # Invalidate the cached_property so updated values are used in logs
            if "_fields" in auth_ctx.__dict__:
                del auth_ctx.__dict__["_fields"]

    async def _call_next_with_auth_client(
        self, request: Request, call_next: Callable, auth_client: AuthClient
    ) -> Response:
        """Call downstream handler with auth context set."""
        context_token = auth_client_context.set(auth_client)
        try:
            return await call_next(request)
        finally:
            auth_client_context.reset(context_token)

    async def _call_pdp(
        self,
        auth_client: AuthClient,
        request: Request,
        scopes: list[str] | None,
        *,
        anonymous_denial_is_401: bool,
    ) -> Response | None:
        """Call PDP and return an error response, or None when authorized."""
        try:
            result = await auth_client.authorize_request(
                method=request.method,
                path=request.url.path,
                scopes=scopes,
                http_client=auth_client.http_client,
            )
        except httpx.ConnectError as e:
            logger.error(
                "Cannot connect to PDP at %s: %s (service: %s)%s",
                self.config.auth_url,
                _describe_pdp_failure(e),
                self.service_name or "unknown",
                _embedded_pdp_base_url_hint(self.config),
            )
            return JSONResponse(
                status_code=503,
                content={"detail": "Authorization service unavailable"},
            )
        except httpx.TimeoutException as e:
            logger.error(
                "PDP timeout at %s: %s (service: %s)",
                self.config.auth_url,
                _describe_pdp_failure(e),
                self.service_name or "unknown",
            )
            return JSONResponse(
                status_code=504,
                content={"detail": "Authorization service timeout"},
            )
        except httpx.HTTPStatusError as e:
            logger.error(
                "PDP error response from %s: HTTP %s (service: %s) body=%r",
                self.config.auth_url,
                e.response.status_code,
                self.service_name or "unknown",
                (e.response.text or "")[:500],
            )
            return JSONResponse(
                status_code=502,
                content={"detail": "Authorization service error"},
            )
        except InvalidScopeFormatError as e:
            logger.warning(
                "Invalid OAuth scope format (service: %s): %s",
                self.service_name or "unknown",
                str(e),
            )
            return JSONResponse(
                status_code=400,
                content={"detail": str(e)},
            )
        except Exception as e:
            logger.exception(
                "Unexpected error during authorization (service: %s): %s",
                self.service_name or "unknown",
                _describe_pdp_failure(e),
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal authorization error"},
            )

        if result.allowed:
            return None

        principal_id = auth_client.principal.id
        status_code = 401 if anonymous_denial_is_401 and not principal_id else 403
        logger.warning(
            "Authorization denied for %s %s (principal: %s, service: %s, reason: %s)",
            request.method,
            request.url.path,
            principal_id or "anonymous",
            self.service_name or "unknown",
            result.reason,
        )
        return JSONResponse(
            status_code=status_code,
            content={"detail": "Unauthorized" if status_code == 401 else "Forbidden"},
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main entry point - routes requests through the appropriate authorization flow.

        The authorization decision follows this priority order:

        1. Health endpoints (/health, /ready, etc.) - always allowed, no auth
        2. Auth disabled (config.enabled=false) - allow all, extract principal
        3. Auth enabled - call PDP for authorization decision

        In all cases where the request proceeds, this method sets up auth_client_context
        for dependency injection via get_auth_client().

        Args:
            request: The incoming HTTP request
            call_next: The next middleware/handler in the chain

        Returns:
            The response, either from downstream handlers or an error response
        """
        path = request.url.path

        # Skip authorization for health check endpoints
        if path in HEALTH_ENDPOINTS:
            return await call_next(request)

        if request.method in ("GET", "HEAD") and path in PUBLIC_GET_PATHS:
            return await call_next(request)

        # Skip authorization for bypass prefixes (e.g., /studio static files)
        if path.startswith(BYPASS_PREFIXES):
            return await call_next(request)

        # PDP HTTP entrypoints: only service principals may call them, and the middleware
        # must not recurse into authorize_request for these paths (AuthClient uses
        # X-NMP-Principal-Id: service:{name}; see _pdp_request_headers).
        if path.startswith("/apis/auth/v2/authz/"):
            headers_dict = dict(request.headers)
            principal_id = headers_dict.get("x-nmp-principal-id", "")
            if principal_id.startswith("service:"):
                return await self._handle_service_principal_request(request, call_next, headers_dict)
            status_code = 401 if not principal_id else 403
            return JSONResponse(
                status_code=status_code,
                content={"detail": "Unauthorized" if status_code == 401 else "Forbidden"},
            )

        headers_dict = dict(request.headers)

        # HF-compatible endpoints: restricted to service principals via Bearer token (HF_TOKEN).
        if path.startswith("/apis/files/v2/hf/"):
            return await self._handle_hf_compatible_request(request, call_next, headers_dict)

        # X-NMP-Principal-* headers (includes service principals — evaluated by PDP)
        if headers_dict.get("x-nmp-principal-id"):
            return await self._handle_principal_headers_request(request, call_next, headers_dict)

        # Skip authorization if auth is disabled, but still extract principal
        if not self.config.enabled:
            return await self._handle_auth_disabled_request(request, call_next)

        # Try to extract principal from Authorization: Bearer header (native OIDC or unsigned JWT)
        try:
            bearer_token = parse_bearer_authorization_header(headers_dict.get("authorization"))
        except MalformedBearerTokenError:
            return JSONResponse(status_code=401, content={"detail": "Invalid bearer token"})
        if bearer_token is not None:
            return await self._handle_bearer_token_request(request, call_next, bearer_token)

        # Perform authorization check with auth endpoint (allows PDP to decide for anonymous access)
        return await self._handle_auth_check(request, call_next)

    async def _handle_hf_compatible_request(
        self, request: Request, call_next: Callable, headers_dict: dict
    ) -> Response:
        """Handle requests to HuggingFace-compatible endpoints.

        These endpoints are restricted to service principals only. Auth is provided
        via Bearer token (HF_TOKEN) containing a service principal identifier like
        "service:nim". This allows huggingface-hub clients to authenticate by setting
        HF_TOKEN=service:<name>.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware/handler in the chain
            headers_dict: Pre-extracted request headers dictionary

        Returns:
            Response from downstream handlers, or 401 if not a valid service principal
        """
        try:
            bearer_token = parse_bearer_authorization_header(headers_dict.get("authorization"))
        except MalformedBearerTokenError:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        if bearer_token is not None and bearer_token.startswith("service:"):
            headers_dict["x-nmp-principal-id"] = bearer_token
            return await self._handle_principal_headers_request(request, call_next, headers_dict)

        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    async def _handle_service_principal_request(
        self, request: Request, call_next: Callable, headers_dict: dict
    ) -> Response:
        """Handle service-principal requests to PDP entrypoints only (`/apis/auth/v2/authz/`).

        These paths must not call `authorize_request`, which would POST to the same
        authz routes and recurse. All other routes use the PDP when `authorize_request` runs.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware/handler in the chain
            headers_dict: Pre-extracted request headers dictionary

        Returns:
            The response from downstream handlers (always proceeds, never denies)
        """
        logger.debug(
            "PDP entrypoint request for %s %s (principal: %s)",
            request.method,
            request.url.path,
            headers_dict.get("x-nmp-principal-id"),
        )

        principal, error_response = self._principal_from_headers(headers_dict)
        if error_response is not None:
            return error_response
        principal = _require_principal(principal)
        auth_client = AuthClient(
            principal=principal, config=self.config, http_client=self._client, service_name=self.service_name
        )
        return await self._call_next_with_auth_client(request, call_next, auth_client)

    async def _handle_principal_headers_request(
        self, request: Request, call_next: Callable, headers_dict: dict
    ) -> Response:
        """Handle requests that include X-NMP-Principal-* headers.

        1. Creates a Principal from the headers
        2. If auth is disabled, proceeds without PDP check
        3. If auth is enabled, performs PDP authorization check

        Args:
            request: The incoming HTTP request
            call_next: The next middleware/handler in the chain
            headers_dict: Pre-extracted request headers dictionary

        Returns:
            The response from downstream handlers or an error response
        """
        principal, error_response = self._principal_from_headers(headers_dict)
        if error_response is not None:
            return error_response
        principal = _require_principal(principal)

        if not self.config.enabled:
            # Auth disabled - just extract principal and proceed
            auth_client = AuthClient(
                principal=principal, config=self.config, http_client=self._client, service_name=self.service_name
            )
            return await self._call_next_with_auth_client(request, call_next, auth_client)

        # Auth enabled - perform PDP check with the parsed principal headers. The HF-compatible
        # flow synthesizes these from its Bearer service token without mutating request.headers.
        return await self._handle_auth_check(request, call_next, headers_dict)

    async def _handle_bearer_token_request(self, request: Request, call_next: Callable, token: str) -> Response:
        """Handle requests with Authorization: Bearer tokens through the shared resolver."""
        if self.config.access_keys.enabled:
            from .access_keys import is_access_key_token_candidate

            if is_access_key_token_candidate(token):
                resolved_or_error = await self._authenticate_access_key_lifecycle(token)
                if isinstance(resolved_or_error, Response):
                    return resolved_or_error
                return await self._handle_resolved_bearer_token(request, call_next, resolved_or_error)

        jwt_validator = self._get_jwt_validator()
        if jwt_validator is None and not self.config.access_keys.enabled:
            logger.warning("Bearer token provided but bearer token authentication is not configured")
            return JSONResponse(
                status_code=401,
                content={"detail": "Bearer token authentication not configured"},
            )

        from .jwt import UnsignedJWTRejectedError

        try:
            resolved = await resolve_bearer_token(
                self.config,
                token,
                jwt_validator=jwt_validator,
                skip_access_key_check=self.config.access_keys.enabled,
            )
        except UnsignedJWTRejectedError as exc:
            return JSONResponse(
                status_code=401,
                content={"detail": str(exc)},
            )

        if resolved is None:
            if jwt_validator is None:
                logger.warning(
                    "Bearer token rejected: OIDC is not configured and the token did not pass "
                    "the Scoped Access Key candidate check (service: %s)",
                    self.service_name or "unknown",
                )
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or expired token"},
            )

        return await self._handle_resolved_bearer_token(request, call_next, resolved)

    def _access_key_lifecycle_error_response(
        self,
        status_code: int,
        detail: str,
        *,
        retry_after: int | None = None,
    ) -> JSONResponse:
        headers = {"Retry-After": str(retry_after)} if retry_after is not None else None
        return JSONResponse(status_code=status_code, content={"detail": detail}, headers=headers)

    async def _authenticate_access_key_lifecycle(
        self,
        token: str,
    ) -> ResolvedBearerToken | Response:
        try:
            resolved = await self._access_key_lifecycle.authenticate(token)
        except AccessKeyLifecycleUnavailableError as exc:
            return self._access_key_lifecycle_error_response(
                exc.status_code,
                exc.detail,
                retry_after=exc.retry_after,
            )
        if resolved is None:
            return JSONResponse(status_code=401, content={"detail": "Invalid or expired token"})
        return resolved

    async def _handle_resolved_bearer_token(
        self,
        request: Request,
        call_next: Callable,
        resolved: ResolvedBearerToken,
    ) -> Response:
        """Authorize a request after a bearer token has produced trusted claims."""
        principal = resolved.principal

        # Update the observability context with principal info for logging
        self._update_auth_context(principal)

        logger.debug(
            "Authenticated via Bearer token: principal=%s, email=%s, groups=%s",
            principal.id,
            principal.email,
            principal.groups,
        )

        # If auth is disabled, just proceed with the principal
        if not self.config.enabled:
            auth_client = AuthClient(
                principal=principal,
                config=self.config,
                http_client=self._client,
                service_name=self.service_name,
                resolved_bearer_token=resolved,
            )
            return await self._call_next_with_auth_client(request, call_next, auth_client)

        # Perform authorization check with PDP
        auth_client = AuthClient(
            principal=principal,
            config=self.config,
            http_client=self._get_client(request),
            service_name=self.service_name,
            resolved_bearer_token=resolved,
        )

        # Extract scopes from token claims
        scopes = resolved.scopes if resolved.scopes else None

        if error_response := await self._call_pdp(
            auth_client,
            request,
            scopes,
            anonymous_denial_is_401=False,
        ):
            return error_response

        return await self._call_next_with_auth_client(request, call_next, auth_client)

    async def _handle_auth_disabled_request(self, request: Request, call_next: Callable) -> Response:
        """Handle requests when authorization is globally disabled (auth.enabled=false).

        When auth is disabled, all requests are allowed through without PDP checks.
        This method still:

        1. Extracts any Principal from headers (if present) for observability
        2. Creates an empty Principal if none provided (for backward compatibility)
        3. Sets up the AuthClient context so handlers can still access principal info

        This is useful for development, testing, or deployments where auth is
        handled externally.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware/handler in the chain

        Returns:
            The response from downstream handlers (always proceeds, never denies)
        """
        headers_dict = dict(request.headers)

        principal, error_response = self._principal_from_headers(headers_dict)
        if error_response is not None:
            return error_response
        principal = _require_principal(principal)

        logger.debug(
            "Auth disabled - principal for %s %s: id=%s, email=%s, groups=%s",
            request.method,
            request.url.path,
            principal.id,
            principal.email,
            principal.groups,
        )

        auth_client = AuthClient(
            principal=principal, config=self.config, http_client=self._client, service_name=self.service_name
        )
        return await self._call_next_with_auth_client(request, call_next, auth_client)

    async def _handle_auth_check(
        self, request: Request, call_next: Callable, headers_dict: dict | None = None
    ) -> Response:
        """Perform authorization check by calling the Policy Decision Point (PDP).

        This is the main authorization flow when auth is enabled and no bypass
        conditions apply. It delegates to AuthClient.authorize_request() for the
        actual PDP call, then handles the response appropriately.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware/handler in the chain
            headers_dict: Optional preprocessed headers. HF-compatible requests use this
                to pass the service principal synthesized from their Bearer token.

        Returns:
            - 401 Unauthorized: If no principal and request is denied
            - 403 Forbidden: If principal exists but lacks permission
            - 503 Service Unavailable: If PDP cannot be reached
            - 504 Gateway Timeout: If PDP times out
            - 502 Bad Gateway: If PDP returns an error response
            - 500 Internal Server Error: For unexpected PDP errors
            - Response from downstream: If authorized
        """
        if not headers_dict:
            headers_dict = dict(request.headers)

        principal, error_response = self._principal_from_headers(headers_dict)
        if error_response is not None:
            return error_response
        principal = _require_principal(principal)

        # Extract scopes from headers (space-separated list per OAuth2 standard)
        scopes_header = headers_dict.get("x-nmp-scopes", "")
        scopes = [s.strip() for s in scopes_header.split() if s.strip()] if scopes_header else None

        # Create AuthClient for the authorization check
        # Pass http_client for ASGI transport in tests - see architecture/docs/http-client-injection.md
        auth_client = AuthClient(
            principal=principal,
            config=self.config,
            http_client=self._get_client(request),
            service_name=self.service_name,
        )

        # Perform authorization check - only catch errors from the PDP call itself.
        # Errors from downstream handlers (call_next) should propagate normally.
        if error_response := await self._call_pdp(
            auth_client,
            request,
            scopes,
            anonymous_denial_is_401=True,
        ):
            return error_response

        # Authorization successful - set up context for downstream handlers.
        # This is outside the try/except so endpoint errors propagate normally
        # to FastAPI's exception handlers (not converted to "Internal authorization error").
        return await self._call_next_with_auth_client(request, call_next, auth_client)
