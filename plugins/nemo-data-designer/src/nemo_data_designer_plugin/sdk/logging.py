# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import logging
from contextlib import contextmanager
from functools import wraps
from typing import Generator, TypeVar

from data_designer.logging import _make_stream_formatter

_PLUGIN_LOGGER_NAME = "nemo_data_designer_plugin"
_LIBRARY_LOGGER_NAME = "data_designer"


@contextmanager
def _attach_stream_handler(logger_name: str) -> Generator[None, None, None]:
    """Attach a stream handler to ``logger_name`` if nothing is configured for it.

    If the logger (or any of its ancestors) already has handlers, this is a no-op,
    preventing duplicate log output when the caller has configured logging themselves.
    The temporarily added handler is removed on exit.
    """
    logger = logging.getLogger(logger_name)
    handler: logging.Handler | None = None
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(_make_stream_formatter())
        logger.addHandler(handler)
        logger.setLevel("INFO")
    try:
        yield
    finally:
        if handler is not None:
            logger.removeHandler(handler)


@contextmanager
def _ensure_logging_handler() -> Generator[None, None, None]:
    """Attach a logging handler to the plugin logger if none is configured."""
    with _attach_stream_handler(_PLUGIN_LOGGER_NAME):
        yield


@contextmanager
def ensure_library_logging_handler() -> Generator[None, None, None]:
    """Attach a logging handler to the upstream ``data_designer`` logger.

    Deliberately *not* part of :func:`with_logging`. Most SDK calls run the
    engine somewhere else — the platform service for ``preview``, a job
    subprocess for ``create`` — and route its logs back as data, which the
    caller already renders. Attaching a handler here for those calls would print
    the same records a second time whenever the service happens to share the
    caller's process, as it does under the in-process test harness.

    So this is for the paths that drive the engine in the caller's own process,
    where its logs would otherwise go nowhere: today, the model health check.
    """
    with _attach_stream_handler(_LIBRARY_LOGGER_NAME):
        yield


_ClsT = TypeVar("_ClsT", bound=type)


def with_logging(cls: _ClsT) -> _ClsT:
    """Wrap public methods so SDK logging is configured on demand.

    This ensures logging is configured for every public method call without
    requiring manual wrapping. Nesting is safe: if the handler is already active
    (e.g. one public method calls another), the inner entry is a no-op.
    """
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
