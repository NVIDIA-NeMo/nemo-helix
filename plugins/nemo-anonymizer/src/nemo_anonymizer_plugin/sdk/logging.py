# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging helpers for the Anonymizer plugin SDK resources."""

from __future__ import annotations

import inspect
import logging
import threading
from contextlib import contextmanager
from functools import wraps
from typing import Generator, TypeVar

_LOGGER_NAME = "nemo_helix.anonymizer"
_HANDLER_MARKER = "_nemo_anonymizer_sdk_handler"
_handler_lock = threading.RLock()
_active_handler_users = 0
_managed_handler: logging.Handler | None = None
_saved_level: int | None = None
_saved_propagate: bool | None = None


def _find_managed_handler(logger: logging.Logger) -> logging.Handler | None:
    for handler in logger.handlers:
        if getattr(handler, _HANDLER_MARKER, False):
            return handler
    return None


def _make_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    setattr(handler, _HANDLER_MARKER, True)
    return handler


@contextmanager
def _ensure_logging_handler() -> Generator[None, None, None]:
    global _active_handler_users, _managed_handler, _saved_level, _saved_propagate

    plugin_logger = logging.getLogger(_LOGGER_NAME)
    with _handler_lock:
        if _active_handler_users == 0:
            _saved_level = plugin_logger.level
            _saved_propagate = plugin_logger.propagate
            _managed_handler = _find_managed_handler(plugin_logger)
            if _managed_handler is None:
                _managed_handler = _make_handler()
                plugin_logger.addHandler(_managed_handler)
            plugin_logger.setLevel(logging.INFO)
            plugin_logger.propagate = False
        _active_handler_users += 1
    try:
        yield
    finally:
        with _handler_lock:
            _active_handler_users -= 1
            if _active_handler_users == 0:
                if _managed_handler is not None and _managed_handler in plugin_logger.handlers:
                    plugin_logger.removeHandler(_managed_handler)
                if _saved_level is not None:
                    plugin_logger.setLevel(_saved_level)
                if _saved_propagate is not None:
                    plugin_logger.propagate = _saved_propagate
                _managed_handler = None
                _saved_level = None
                _saved_propagate = None


_ClsT = TypeVar("_ClsT", bound=type)


def with_logging(cls: _ClsT) -> _ClsT:
    """Wrap public methods so SDK logging is configured on demand."""
    for name, method in vars(cls).items():
        if name.startswith("_") or isinstance(method, (staticmethod, classmethod)):
            continue
        if inspect.iscoroutinefunction(method):

            @wraps(method)
            async def async_wrapper(self, *args, _m=method, **kwargs):
                with _ensure_logging_handler():
                    return await _m(self, *args, **kwargs)

            setattr(cls, name, async_wrapper)
        elif inspect.isfunction(method):

            @wraps(method)
            def sync_wrapper(self, *args, _m=method, **kwargs):
                with _ensure_logging_handler():
                    return _m(self, *args, **kwargs)

            setattr(cls, name, sync_wrapper)
    return cls
