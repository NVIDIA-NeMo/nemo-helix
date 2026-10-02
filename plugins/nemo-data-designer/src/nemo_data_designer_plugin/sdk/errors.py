# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import suppress

import httpx
from data_designer.errors import DataDesignerError


class DataDesignerClientError(DataDesignerError):
    """Base exception for Data Designer client errors.

    When the error originated from an HTTP response, the underlying status code
    is exposed as :attr:`status_code` so callers can branch on it cleanly
    instead of pattern-matching the message string.
    """

    def __init__(self, *args: object, status_code: int | None = None) -> None:
        super().__init__(*args)
        self.status_code = status_code


class DataDesignerConfigValidationError(DataDesignerClientError):
    """Exception raised when the Data Designer configuration is invalid."""


class DataDesignerPreviewError(DataDesignerClientError):
    """Raised for errors related to a Data Designer preview request."""


class DataDesignerJobError(DataDesignerClientError):
    """Raised for errors related to a Data Designer job."""


def extract_http_error_info(exc: httpx.HTTPStatusError) -> tuple[int, str]:
    """Pull the status code and a human-readable detail string out of an httpx error.

    Tries to parse the response body as JSON and use its ``detail`` field (the
    convention used by FastAPI / NeMo Helix); falls back to the raw body
    text if that isn't available.
    """
    response = exc.response
    try:
        response.read()
    except Exception:
        pass

    detail = response.text
    body = None
    with suppress(Exception):
        body = response.json()
    if isinstance(body, dict) and isinstance(body.get("detail"), str):
        detail = body["detail"]

    return response.status_code, detail
