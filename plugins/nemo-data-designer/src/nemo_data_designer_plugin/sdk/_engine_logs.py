# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary log forwarding for ``data_designer`` engine logs.

``check-models`` needs these logs to be useful rather than merely chatty. The
engine's health check re-raises a failing model's error without naming the
alias it was probing — the alias appears only in the log line immediately
above the failure::

    👀 Checking 'some-model' in provider named 'p' for model alias 'text'...
    ❌ Failed!

Forwarding those records is what lets a fail-fast probe still tell the user
*which* model broke, and show progress across the ones that passed.

This is the short-lived counterpart to
:mod:`nemo_data_designer_plugin.functions._preview_logs`, which solves the same
problem for the long-running service: that module attaches a permanent handler
and routes records through a task-local ``ContextVar`` so concurrent requests
don't cross-talk. A one-shot CLI or SDK call has no such contention, so this
attaches a handler for the duration of one call and removes it afterwards.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager

LogCallback = Callable[[str], None]

_LIBRARY_LOGGER = "data_designer"


class _CallbackHandler(logging.Handler):
    def __init__(self, on_log: LogCallback) -> None:
        super().__init__()
        self._on_log = on_log

    def emit(self, record: logging.LogRecord) -> None:
        self._on_log(record.getMessage())


@contextmanager
def forward_engine_logs(on_log: LogCallback | None) -> Iterator[None]:
    """Forward ``data_designer`` log records to ``on_log`` for the duration of the block.

    A ``None`` callback makes this a no-op, so callers that don't want engine
    logs (SDK users with their own logging setup, ``--output json``) can pass
    one through unconditionally.

    The callback may be invoked from a worker thread, since the engine's sync
    APIs run under ``asyncio.to_thread``.

    Args:
        on_log: Receives each record's formatted message, or ``None`` to disable.
    """
    if on_log is None:
        yield
        return

    lib_logger = logging.getLogger(_LIBRARY_LOGGER)
    handler = _CallbackHandler(on_log)

    # The library's own ``configure_logging`` would raise this logger to INFO,
    # but ``create_data_designer`` deliberately no-ops that to keep the engine
    # from reconfiguring process-wide logging. Without an explicit level the
    # logger inherits root's default (``WARNING``) and the INFO records we care
    # about drop before reaching the handler.
    previous_level = lib_logger.level
    if lib_logger.getEffectiveLevel() > logging.INFO:
        lib_logger.setLevel(logging.INFO)

    lib_logger.addHandler(handler)
    try:
        yield
    finally:
        lib_logger.removeHandler(handler)
        lib_logger.setLevel(previous_level)
