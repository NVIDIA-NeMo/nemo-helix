# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Testing utilities for mocking Policy Decision Point (PDP) in tests.

This module provides a clean interface for mocking PDP (e.g., OPA) API calls in tests using respx.

Key Features:
- Request matching: Mock specific PDP calls based on request content (partial matching supported)
- One-to-one mocking: Each mock call handles exactly one request (consumed on use)
- Call validation: Detects missing or extra calls with assert_all_called
- Lazy route registration: Routes only registered when first mock is added
- Debug logging: Detailed logs show which mocks matched and why

Implementation Notes:
The PDPMock class uses respx side_effect arrays to create one handler per mock call.
Routes are registered lazily when the first mock is added for that route.
Handlers are consumed in order (FIFO), so the order of mock setup must match the order
of actual calls.

If you need to handle the same request pattern multiple times, call the mock method
multiple times:

Example:
    with pdp_mock() as pdp:
        # Mock two separate permission checks
        pdp.mock_has_permissions(
            request={"input": {"principal_id": "user123"}},
            allowed=True
        )
        pdp.mock_has_permissions(
            request={"input": {"principal_id": "user123"}},
            allowed=True
        )

        # Make two requests - each consumes one mock
        auth.has_permissions("ws1", ["models.create"])
        auth.has_permissions("ws2", ["datasets.read"])
"""

import json
import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

import httpx
from nemo_helix_ext.auth.helpers import generate_unsigned_jwt as generate_unsigned_jwt_helper

# Some packages do not have respx as a dependency
try:
    import respx
except ImportError:
    respx = None

logger = logging.getLogger(__name__)


def generate_unsigned_jwt(
    principal_id: str,
    *,
    email: str | None = None,
    groups: list[str] | None = None,
    scopes: list[str] | None = None,
    expires_in_seconds: int | None = 3600,
    issued_at: int | None = None,
    audience: str | None = None,
    issuer: str | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Generate an unsigned JWT for local development and testing."""
    return generate_unsigned_jwt_helper(
        principal_id=principal_id,
        email=email,
        groups=groups,
        scopes=scopes,
        expires_in_seconds=expires_in_seconds,
        issued_at=issued_at,
        audience=audience,
        issuer=issuer,
        extra_claims=extra_claims,
    )


class PDPMock:
    """Helper class for mocking Policy Decision Point (PDP) API calls with a friendlier interface."""

    def __init__(self, respx_mock):
        self.respx_mock = respx_mock
        self._allow_route = None
        self._has_permissions_route = None
        self._allow_side_effects = []
        self._has_permissions_side_effects = []

    def mock_allow(
        self,
        method: Optional[str] = None,
        path: Optional[str] = None,
        principal_id: Optional[str] = None,
        principal_email: Optional[str] = None,
        groups: Optional[list] = None,
        scopes: Optional[list] = None,
        request: Optional[Dict[str, Any]] = None,
        status_code: int = 200,
        response: Optional[Dict[str, Any]] = None,
        allowed: bool = True,
    ):
        """Mock the OPA allow endpoint.

        Args:
            method: HTTP method (e.g., "GET", "POST") - used to build request
            path: Request path (e.g., "/v1/workspaces") - used to build request
            principal_id: Principal ID - used to build request
            principal_email: Principal email - used to build request
            groups: Principal groups list - used to build request
            scopes: OAuth2/OIDC scopes list - used to build request
            request: Optional partial JSON to match in the request body (overrides above params)
            status_code: HTTP status code to return
            response: Full response to return (overrides other params)
            allowed: Whether the request should be allowed
        """
        # Build the request from convenience parameters if not provided explicitly
        if request is None and any([method, path, principal_id, principal_email, groups, scopes]):
            request = {"input": {}}
            if method is not None:
                request["input"]["method"] = method
            if path is not None:
                request["input"]["path"] = path
            if principal_id is not None:
                request["input"]["principal_id"] = principal_id
            if principal_email is not None:
                request["input"]["principal_email"] = principal_email
            if groups is not None:
                request["input"]["principal_groups"] = groups
            if scopes is not None:
                request["input"]["scopes"] = scopes

        # Build the response if not provided
        if response is None:
            response = {"result": {"allowed": allowed, "headers": {"X-NHX-Authorized": "true" if allowed else "false"}}}

        # Create a handler function bound to this specific request/response
        def handler(req):
            try:
                content = req.content
                body = json.loads(content.decode()) if content else {}
                logger.debug("PDP mock (allow) checking request body: %s", json.dumps(body, indent=2))
            except Exception as e:
                logger.debug("Failed to parse request body: %s", e)
                return httpx.Response(500, json={"error": "Failed to parse request"})

            # Check if request matches expected pattern
            if request is None or PDPMock._json_contains(request, body):
                logger.debug("PDP mock matched! Returning allowed=%s", response["result"]["allowed"])
                return httpx.Response(status_code, json=response)
            else:
                # Request didn't match - raise descriptive error
                raise AssertionError(
                    f"PDP mock (allow): Request did not match expected pattern.\n"
                    f"Expected: {json.dumps(request, indent=2)}\n"
                    f"Got: {json.dumps(body, indent=2)}"
                )

        # Add handler to side effects array
        self._allow_side_effects.append(handler)

        # Register or update the route (using relative path since respx.mock has base_url)
        if self._allow_route is None:
            self._allow_route = self.respx_mock.post("/v1/data/authz/allow")
        self._allow_route.mock(side_effect=self._allow_side_effects)

    def mock_has_permissions(
        self,
        principal_id: Optional[str] = None,
        workspace: Optional[str] = None,
        permissions: Optional[list] = None,
        principal_email: Optional[str] = None,
        groups: Optional[list] = None,
        request: Optional[Dict[str, Any]] = None,
        status_code: int = 200,
        response: Optional[Dict[str, Any]] = None,
        allowed: bool = True,
    ):
        """Mock the auth has_permissions endpoint.

        Args:
            principal_id: Principal ID - used to build request
            workspace: Workspace ID - used to build request
            permissions: List of permissions to check - used to build request
            principal_email: Principal email - used to build request
            groups: Principal groups list - used to build request
            request: Optional partial JSON to match in the request body (overrides above params)
            status_code: HTTP status code to return
            response: Full response to return (overrides other params)
            allowed: Whether the permission check should pass
        """
        # Build the request from convenience parameters if not provided explicitly
        if request is None and any([principal_id, workspace, permissions, principal_email, groups]):
            request = {"input": {}}
            if principal_id is not None:
                request["input"]["principal_id"] = principal_id
            if workspace is not None:
                request["input"]["workspace"] = workspace
            if permissions is not None:
                request["input"]["permissions"] = permissions
            if principal_email is not None:
                request["input"]["principal_email"] = principal_email
            if groups is not None:
                request["input"]["principal_groups"] = groups

        # Build the response if not provided
        if response is None:
            response = {"result": {"allowed": allowed}}

        # Create a handler function bound to this specific request/response
        def handler(req):
            try:
                content = req.content
                body = json.loads(content.decode()) if content else {}
                logger.debug("PDP mock (has_permissions) checking request body: %s", json.dumps(body, indent=2))
            except Exception as e:
                logger.debug("Failed to parse request body: %s", e)
                return httpx.Response(500, json={"error": "Failed to parse request"})

            # Check if request matches expected pattern
            if request is None or PDPMock._json_contains(request, body):
                logger.debug("PDP mock matched! Returning allowed=%s", response["result"]["allowed"])
                return httpx.Response(status_code, json=response)
            else:
                # Request didn't match - raise descriptive error
                raise AssertionError(
                    f"PDP mock (has_permissions): Request did not match expected pattern.\n"
                    f"Expected: {json.dumps(request, indent=2)}\n"
                    f"Got: {json.dumps(body, indent=2)}"
                )

        # Add handler to side effects array
        self._has_permissions_side_effects.append(handler)

        # Register or update the route (using relative path since respx.mock has base_url)
        if self._has_permissions_route is None:
            self._has_permissions_route = self.respx_mock.post("/v1/data/authz/has_permissions")
        self._has_permissions_route.mock(side_effect=self._has_permissions_side_effects)

    @staticmethod
    def _json_contains(expected: Any, actual: Any) -> bool:
        """Return True if 'expected' is a subset of 'actual' JSON structure."""
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return False
            for key, exp_val in expected.items():
                if key not in actual:
                    return False
                if not PDPMock._json_contains(exp_val, actual[key]):
                    return False
            return True
        if isinstance(expected, list):
            if not isinstance(actual, list):
                return False
            # All expected items must be present in actual (order-agnostic)
            for item in expected:
                if not any(PDPMock._json_contains(item, act_item) for act_item in actual):
                    return False
            return True
        return expected == actual


@contextmanager
def pdp_mock(base_url: str = "http://opa:8181", assert_all_called: bool = True):
    """Context manager that provides a friendlier interface for mocking PDP calls.

    Usage:
        with pdp_mock() as pdp:
            # Simple usage without request matching
            pdp.mock_allow(allowed=True)
            pdp.mock_has_permissions(allowed=True)

            # Using convenience parameters for request matching
            pdp.mock_allow(
                method="POST",
                path="/v2/workspaces/test-ws/models",
                principal_id="user123",
                scopes=["models:write"],
                allowed=True,
            )

            pdp.mock_has_permissions(
                principal_id="user123",
                workspace="test-ws",
                permissions=["models.create"],
                groups=["developers"],
                allowed=True
            )

            # Or using explicit request dict (more control)
            pdp.mock_has_permissions(
                request={"input": {"principal_id": "user123", "workspace": "test-ws"}},
                allowed=True
            )

    Note: When using request matching, partial JSON matching is supported.
    Only the provided keys/values need to be present in the request body.

    Args:
        base_url: Base URL for PDP service (e.g., OPA). Default is "http://opa:8181".
                 All HTTP calls to this base URL must be mocked.
        assert_all_called: If True (default), assert that all mocked routes were called.
                          This helps catch missing or extra API calls in tests.
    """
    with respx.mock(base_url=base_url, assert_all_called=assert_all_called) as respx_mock:
        yield PDPMock(respx_mock)


# Backward compatibility aliases
OPAMock = PDPMock
opa_mock = pdp_mock
