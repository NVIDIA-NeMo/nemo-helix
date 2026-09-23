# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Access log utilities for capturing HTTP requests in tests."""

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


@dataclass
class CapturedRequest:
    """A captured HTTP request for test verification."""

    method: str
    """HTTP method (GET, POST, etc.)."""

    path: str
    """Request path (without host, e.g., '/v2/workspaces/default/metrics')."""

    query_string: str
    """Query string (e.g., 'page=1&page_size=10')."""

    headers: dict[str, str]
    """Request headers (lowercase keys)."""

    @property
    def principal_id(self) -> str:
        """Get the X-NHX-Principal-Id header value."""
        return self.headers.get("x-nhx-principal-id", "")

    @property
    def principal_email(self) -> str:
        """Get the X-NHX-Principal-Email header value."""
        return self.headers.get("x-nhx-principal-email", "")

    @property
    def on_behalf_of(self) -> str:
        """Get the X-NHX-Principal-On-Behalf-Of header value."""
        return self.headers.get("x-nhx-principal-on-behalf-of", "")

    def matches(
        self,
        *,
        method: str | None = None,
        path_contains: str | None = None,
        path_startswith: str | None = None,
        principal_id: str | None = None,
    ) -> bool:
        """Check if this request matches the given criteria.

        Args:
            method: Match exact HTTP method (case-insensitive).
            path_contains: Match if path contains this substring.
            path_startswith: Match if path starts with this prefix.
            principal_id: Match exact principal ID.

        Returns:
            True if all specified criteria match.
        """
        if method is not None and self.method.upper() != method.upper():
            return False
        if path_contains is not None and path_contains not in self.path:
            return False
        if path_startswith is not None and not self.path.startswith(path_startswith):
            return False
        if principal_id is not None and self.principal_id != principal_id:
            return False
        return True


@dataclass
class AccessLog:
    """Collection of captured HTTP requests for test verification.

    Example usage:
        with create_test_client(MyService, access_log=True, client_type=ClientContext) as ctx:
            ctx.access_log.clear()  # Clear setup requests
            client_from_platform(ctx.sdk, WorkspacesClient).list_workspaces()
            assert ctx.access_log.has_request(path_contains="/workspaces", method="GET")
            for req in ctx.access_log.filter(path_contains="/entities/"):
                assert req.principal_id == "expected-user@example.com"
    """

    requests: list[CapturedRequest] = field(default_factory=list)
    """List of captured requests in chronological order."""

    def clear(self) -> None:
        """Clear all captured requests."""
        self.requests.clear()

    def filter(
        self,
        *,
        method: str | None = None,
        path_contains: str | None = None,
        path_startswith: str | None = None,
        principal_id: str | None = None,
    ) -> list[CapturedRequest]:
        """Filter captured requests by criteria.

        Args:
            method: Match exact HTTP method (case-insensitive).
            path_contains: Match if path contains this substring.
            path_startswith: Match if path starts with this prefix.
            principal_id: Match exact principal ID.

        Returns:
            List of matching requests.
        """
        return [
            req
            for req in self.requests
            if req.matches(
                method=method,
                path_contains=path_contains,
                path_startswith=path_startswith,
                principal_id=principal_id,
            )
        ]

    def has_request(
        self,
        *,
        method: str | None = None,
        path_contains: str | None = None,
        path_startswith: str | None = None,
        principal_id: str | None = None,
    ) -> bool:
        """Check if any captured request matches the criteria.

        Args:
            method: Match exact HTTP method (case-insensitive).
            path_contains: Match if path contains this substring.
            path_startswith: Match if path starts with this prefix.
            principal_id: Match exact principal ID.

        Returns:
            True if at least one request matches.
        """
        return (
            len(
                self.filter(
                    method=method,
                    path_contains=path_contains,
                    path_startswith=path_startswith,
                    principal_id=principal_id,
                )
            )
            > 0
        )

    def assert_has_request(
        self,
        *,
        method: str | None = None,
        path_contains: str | None = None,
        path_startswith: str | None = None,
        principal_id: str | None = None,
        message: str | None = None,
    ) -> CapturedRequest:
        """Assert that at least one request matches the criteria.

        Args:
            method: Match exact HTTP method (case-insensitive).
            path_contains: Match if path contains this substring.
            path_startswith: Match if path starts with this prefix.
            principal_id: Match exact principal ID.
            message: Optional custom error message.

        Returns:
            The first matching request.

        Raises:
            AssertionError: If no request matches.
        """
        matches = self.filter(
            method=method,
            path_contains=path_contains,
            path_startswith=path_startswith,
            principal_id=principal_id,
        )
        if not matches:
            criteria = {
                k: v
                for k, v in {
                    "method": method,
                    "path_contains": path_contains,
                    "path_startswith": path_startswith,
                    "principal_id": principal_id,
                }.items()
                if v is not None
            }
            paths = [r.path for r in self.requests]
            error_msg = message or f"No request matching {criteria}. Captured paths: {paths}"
            raise AssertionError(error_msg)
        return matches[0]


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Middleware that captures request details for test verification."""

    def __init__(self, app, access_log: AccessLog):
        super().__init__(app)
        self.access_log = access_log

    async def dispatch(self, request: Request, call_next):
        # Capture request details (use relative path, not full URL)
        captured = CapturedRequest(
            method=request.method,
            path=request.url.path,
            query_string=request.url.query or "",
            headers={k.lower(): v for k, v in request.headers.items()},
        )
        self.access_log.requests.append(captured)
        return await call_next(request)
