# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-request log forwarding for ``data_designer`` library logs.

The :class:`PreviewHandler` is attached idempotently to the ``data_designer``
logger on first use and stays attached. It reads the per-request callback
from a :class:`contextvars.ContextVar`, which is task-local — so concurrent
preview requests in the FastAPI service don't cross-talk: each task's handler
emit reads its own callback and pushes frames into its own stream.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar

from nemo_data_designer_plugin.functions._types import LogFrame, LogLevel

logger = logging.getLogger(__name__)

MessageCallback = Callable[[LogFrame], None]

_request_callback_cvar: ContextVar[MessageCallback | None] = ContextVar("_dd_preview_request_callback", default=None)
_handler_attached = False


class PreviewHandler(logging.Handler):
    """Logging handler that forwards each ``data_designer`` log record to the task-local callback."""

    def emit(self, record: logging.LogRecord) -> None:
        callback = _request_callback_cvar.get()
        if callback is None:
            return
        callback(LogFrame(message=record.getMessage(), level=_log_level(record)))


def _ensure_handler_attached() -> None:
    """Attach the singleton :class:`PreviewHandler` once per process.

    Idempotent. Called lazily from :func:`forward_data_designer_logs`.
    The handler is never detached because doing so would race with
    concurrent requests; the per-task cvar callback is what makes the
    handler safe to keep around.
    """
    global _handler_attached
    if _handler_attached:
        return
    lib_logger = logging.getLogger("data_designer")
    lib_logger.addHandler(PreviewHandler())
    # The library's own ``configure_logging`` raises this logger to INFO,
    # but that init only runs from the upstream ``data-designer`` CLI or the
    # ``DataDesigner`` interface class — not our ``engine``-direct flow.
    # Without an explicit level the logger inherits root's default
    # (``WARNING``) and INFO records drop before reaching the handler.
    if lib_logger.getEffectiveLevel() > logging.INFO:
        lib_logger.setLevel(logging.INFO)
    _handler_attached = True


@contextmanager
def forward_data_designer_logs(callback: MessageCallback) -> Generator[None, None, None]:
    """Forward ``data_designer`` library logs through *callback* for the ``with`` block.

    Sets a per-task contextvar so concurrent requests don't cross-talk
    (each task's :class:`PreviewHandler` ``emit`` reads its own callback).
    On first use, idempotently attaches a singleton handler to the
    ``data_designer`` logger.
    """
    _ensure_handler_attached()
    token = _request_callback_cvar.set(callback)
    try:
        yield
    finally:
        _request_callback_cvar.reset(token)


def _log_level(record: logging.LogRecord) -> LogLevel:
    if record.levelno >= logging.ERROR:
        return "error"
    if record.levelno >= logging.WARNING:
        return "warning"
    if record.levelno >= logging.INFO:
        return "info"
    return "debug"
