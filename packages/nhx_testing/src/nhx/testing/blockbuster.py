# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BlockBuster utilities for detecting blocking calls in async code.

This module provides helpers for using BlockBuster to detect blocking operations
(socket, SSL, sleep) in async code during tests.

Usage in conftest.py:

    from nhx.testing.blockbuster import blockbuster_fixture

    blockbuster = blockbuster_fixture(autouse=True)
"""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from blockbuster import BlockBuster


def _get_nhx_namespace_paths() -> list[str]:
    """Get all paths from the nhx namespace package."""
    import nhx

    return list(nhx.__path__)


# Only monitor networking and sleep - the most important to catch in async code.
# File I/O is excluded because many third-party libraries (botocore, OPA, etc.)
# do synchronous file reads internally.
ENABLED_FUNCTIONS = [
    "socket.socket.accept",
    "socket.socket.connect",
    "socket.socket.recv",
    "socket.socket.recv_into",
    "socket.socket.recvfrom",
    "socket.socket.recvfrom_into",
    "socket.socket.recvmsg",
    "socket.socket.send",
    "socket.socket.sendall",
    "socket.socket.sendto",
    "ssl.SSLSocket.read",
    "ssl.SSLSocket.recv",
    "ssl.SSLSocket.send",
    "ssl.SSLSocket.write",
    "time.sleep",
]


@contextmanager
def blockbuster_ctx() -> Iterator[BlockBuster]:
    """Context manager for BlockBuster configured for NeMo Helix.

    Creates a BlockBuster that scans all paths in the nhx namespace package,
    monitoring only networking and sleep operations.

    Yields:
        A configured and activated BlockBuster instance.
    """
    bb = BlockBuster(scanned_modules=["nhx"])

    # Patch _scanned_modules BEFORE activation to include all namespace paths
    all_paths = _get_nhx_namespace_paths()
    for func in bb.functions.values():
        func._scanned_modules = all_paths

    # Only activate the functions we want to monitor
    for func_name in ENABLED_FUNCTIONS:
        if func_name in bb.functions:
            bb.functions[func_name].activate()

    yield bb

    bb.deactivate()


def blockbuster_fixture(autouse: bool = False):
    """Create a pytest fixture for BlockBuster.

    Args:
        autouse: If True, the fixture runs automatically for all tests.

    Returns:
        A pytest fixture function.

    Example:
        # In conftest.py
        from nhx.testing.blockbuster import blockbuster_fixture

        blockbuster = blockbuster_fixture(autouse=True)
    """

    @pytest.fixture(autouse=autouse)
    def blockbuster() -> Iterator[BlockBuster]:
        """Detect blocking calls in async code."""
        with blockbuster_ctx() as bb:
            yield bb

    return blockbuster
