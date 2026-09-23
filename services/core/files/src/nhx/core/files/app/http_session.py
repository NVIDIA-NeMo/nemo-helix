# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging

import aiohttp

logger = logging.getLogger(__name__)

# Sessions keyed by event loop - each loop gets its own session
_sessions: dict[asyncio.AbstractEventLoop, aiohttp.ClientSession] = {}


def get_http_session() -> aiohttp.ClientSession:
    """Get the HTTP session for the current event loop.

    Creates a new session with connection pooling if one doesn't exist
    for the current loop. Each event loop gets its own session, which
    handles the test scenario where each TestClient has its own loop.

    Returns:
        The aiohttp.ClientSession for the current event loop.
    """
    loop = asyncio.get_running_loop()
    if loop not in _sessions or _sessions[loop].closed:
        connector = aiohttp.TCPConnector(
            limit=500,  # Total connection pool size
            limit_per_host=100,  # Max connections per host (S3, HF, NGC backends)
            ttl_dns_cache=300,  # Cache DNS lookups for 5 minutes
        )
        _sessions[loop] = aiohttp.ClientSession(connector=connector)
        logger.info("Created HTTP session with connection pooling for event loop")
    return _sessions[loop]


async def close_http_session() -> None:
    """Close the HTTP session for the current event loop.

    Called during app shutdown to properly close connections.
    """
    loop = asyncio.get_running_loop()
    if loop in _sessions and not _sessions[loop].closed:
        await _sessions[loop].close()
        del _sessions[loop]
        logger.info("Closed HTTP session for event loop")
