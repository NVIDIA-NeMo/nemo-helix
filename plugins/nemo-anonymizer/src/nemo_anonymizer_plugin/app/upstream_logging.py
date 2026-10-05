# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for isolating upstream library logging side effects."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager

_ROOT_LOGGING_LOCK = threading.RLock()


@contextmanager
def preserve_root_logging() -> Iterator[None]:
    """Restore the platform root logger after upstream library initialization.

    Data Designer's interface runtime configures logging by clearing root
    handlers. Inside the platform API process that removes NeMo Helix's structured
    logging handler and its internal-request filter, so background controller
    traffic starts printing as ordinary request logs and we start seeing flood of
    logs like "Request Completed", which are the requests that jobs and Models
    Controllers send in some interval. This helper keeps the upstream package
    free to initialize its own named loggers, but put the root logger back.
    """

    root = logging.getLogger()
    with _ROOT_LOGGING_LOCK:
        handlers = list(root.handlers)
        filters = list(root.filters)
        level = root.level
        disabled = root.disabled
        propagate = root.propagate
        try:
            yield
        finally:
            root.handlers = handlers
            root.filters = filters
            root.setLevel(level)
            root.disabled = disabled
            root.propagate = propagate
