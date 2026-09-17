# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Error compatibility helpers for SDK resources mounted on NeMoPlatform."""

from collections.abc import Iterator
from contextlib import contextmanager

import httpx
from nemo_platform_plugin.client.errors import NemoHTTPError


@contextmanager
def httpx_status_errors() -> Iterator[None]:
    """Preserve the historical raw-httpx error shape of plugin SDK resources."""
    try:
        yield
    except NemoHTTPError as exc:
        try:
            exc.http_response.raise_for_status()
        except httpx.HTTPStatusError as httpx_exc:
            raise httpx_exc from exc
        raise
