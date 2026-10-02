# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Error hierarchy for the NemoClient.

All request/response failures derive from :class:`NemoClientError`.
Non-2xx responses additionally derive from :class:`NemoHTTPError` so callers
can distinguish an HTTP status from transport and response-validation errors.
"""

from __future__ import annotations

import httpx


class NemoClientError(Exception):
    """Base class for failures while executing or parsing a client request."""


class NemoTransportError(NemoClientError):
    """Raised when the HTTP transport fails after retries are exhausted."""

    def __init__(self, error: httpx.TransportError) -> None:
        self.error = error
        try:
            self.request = error.request
        except RuntimeError:
            self.request = None
        super().__init__(str(error))


class NemoResponseValidationError(NemoClientError):
    """Raised when a successful response does not match the endpoint contract."""

    def __init__(self, http_response: httpx.Response, error: Exception) -> None:
        self.http_response = http_response
        self.status_code = http_response.status_code
        self.body = self._extract_body(http_response)
        self.error = error
        super().__init__("Data returned by API is invalid for the expected schema")

    @staticmethod
    def _extract_body(resp: httpx.Response) -> object | None:
        try:
            return resp.json()
        except Exception:
            return None


class NemoHTTPError(NemoClientError):
    """Raised on non-2xx HTTP responses.

    Attributes:
        http_response: The raw httpx response.
        status_code: The HTTP status code.
        detail: A human-readable error message extracted from the response
            body (``{"detail": "..."}`` convention used by FastAPI / NeMo
            Platform), or the raw response text as a fallback.
        body: The parsed JSON response body, or None.
    """

    def __init__(self, http_response: httpx.Response) -> None:
        self.http_response = http_response
        self.status_code = http_response.status_code
        self.detail = self._extract_detail(http_response)
        self.body = self._extract_body(http_response)
        super().__init__(f"HTTP {self.status_code}: {self.detail}")

    @staticmethod
    def _extract_body(resp: httpx.Response) -> object | None:
        try:
            return resp.json()
        except Exception:
            return None

    @staticmethod
    def _extract_detail(resp: httpx.Response) -> str:
        try:
            body = resp.json()
            if isinstance(body, dict) and isinstance(body.get("detail"), str):
                return body["detail"]
        except Exception:
            pass
        try:
            return resp.text
        except Exception:
            return resp.reason_phrase or f"HTTP {resp.status_code}"


# ---------------------------------------------------------------------------
# Status-code-specific errors
# ---------------------------------------------------------------------------


class BadRequestError(NemoHTTPError):
    """HTTP 400"""


class AuthenticationError(NemoHTTPError):
    """HTTP 401"""


class PermissionDeniedError(NemoHTTPError):
    """HTTP 403"""


class NotFoundError(NemoHTTPError):
    """HTTP 404"""


class ConflictError(NemoHTTPError):
    """HTTP 409"""


class UnprocessableEntityError(NemoHTTPError):
    """HTTP 422"""


class RateLimitError(NemoHTTPError):
    """HTTP 429"""


class InternalServerError(NemoHTTPError):
    """HTTP 500+"""


_STATUS_CODE_TO_ERROR: dict[int, type[NemoHTTPError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    409: ConflictError,
    422: UnprocessableEntityError,
    429: RateLimitError,
    500: InternalServerError,
}


def raise_for_status(http_response: httpx.Response) -> None:
    """Raise status-code-specific NemoHTTPError subclass for non-2xx responses."""
    if 200 <= http_response.status_code < 300:
        return
    error_cls = _STATUS_CODE_TO_ERROR.get(http_response.status_code, NemoHTTPError)
    if error_cls is NemoHTTPError and http_response.status_code >= 500:
        error_cls = InternalServerError
    raise error_cls(http_response)
