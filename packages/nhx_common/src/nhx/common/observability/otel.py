# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observability facade: settings, header context, and the ``initialize_obs`` coordinator.

Sub-modules house the heavy lifting:

* ``structured_logging``  -- structured logging, log filters, OTLP log export
* ``tracing``  -- OTel resource, tracing/metrics init, instrumentations
* ``middleware`` -- ``RequestLoggingMiddleware``

This module re-exports their public symbols via ``__getattr__`` so
existing ``from nhx.common.observability.otel import …`` paths keep
working.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from contextvars import ContextVar, Token
from typing import Dict, Literal

from pydantic_settings import BaseSettings


def _get_env_bool(var_name: str, default: bool = False) -> bool:
    val = os.getenv(var_name)
    if val is None:
        return default
    return val.lower() in ("1", "true")


class OTELSettings(BaseSettings):
    """
    OpenTelemetry configuration settings represented as a Pydantic model.

    See: https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/
    """

    otel_sdk_disabled: bool = False
    otel_exporter_otlp_endpoint: str | None = None
    otel_exporter_otlp_insecure: bool = False
    otel_metrics_exporter: str = "none"
    otel_traces_exporter: str = "none"
    otel_logs_exporter: str = "none"
    otel_exporter_otlp_traces_endpoint: str | None = None
    otel_exporter_otlp_metrics_endpoint: str | None = None
    otel_exporter_otlp_logs_endpoint: str | None = None

    otel_nhx_include_auth_context: bool = True
    log_level: str = "INFO"
    extra_log_config: str = ""

    log_format: Literal["json", "plain"] = "plain"
    log_internal_requests: bool = False

    @property
    def otel_exporter_otlp_traces_insecure(self) -> bool:
        return self.otel_exporter_otlp_insecure or _get_env_bool("OTEL_EXPORTER_OTLP_TRACES_INSECURE")

    @property
    def otel_exporter_otlp_metrics_insecure(self) -> bool:
        return self.otel_exporter_otlp_insecure or _get_env_bool("OTEL_EXPORTER_OTLP_METRICS_INSECURE")

    @property
    def otel_exporter_otlp_logs_insecure(self) -> bool:
        return self.otel_exporter_otlp_insecure or _get_env_bool("OTEL_EXPORTER_OTLP_LOGS_INSECURE")


settings = OTELSettings()

INTERNAL_REQUEST_HEADER = "X-NHX-Internal"
MARK_INTERNAL_REQUEST_HEADERS = {INTERNAL_REQUEST_HEADER: "true"}

otel_headers_context: ContextVar[Dict[str, str] | None] = ContextVar("otel_headers_context", default=None)


def set_otel_headers(headers: Dict[str, str]) -> Token[Dict[str, str] | None]:
    """Set headers to propagate through the request chain (e.g., X-NHX-Internal).

    This is called by middleware to store headers that should be forwarded
    to downstream service calls via EntityClient.  Returns a token that
    can be passed to ``otel_headers_context.reset()`` to restore the
    previous value.
    """
    return otel_headers_context.set(headers.copy())


def get_otel_headers() -> Dict[str, str]:
    """Get headers that should be propagated to downstream service calls.

    Returns:
        A shallow copy of the propagation headers, or empty dict if none set.
    """
    headers = otel_headers_context.get()
    return headers.copy() if headers else {}


@contextlib.contextmanager
def scoped_otel_headers(headers: Dict[str, str]) -> Iterator[None]:
    """Context manager that sets propagation headers and resets them on exit.

    Use this instead of bare ``set_otel_headers`` when the caller is not
    request-scoped middleware (which already resets via the token).  This
    prevents the headers from leaking to unrelated downstream calls in the
    same async task.
    """
    token = set_otel_headers(headers)
    try:
        yield
    finally:
        otel_headers_context.reset(token)


_obs_initialized: bool = False


def initialize_obs(resource_attributes: Dict[str, str] | None = None):
    """
    Entrypoint for initializing OpenTelemetry observability for this application.

    For FastAPI applications, this should be called during application lifespan initialization.
    Safe to call multiple times -- subsequent calls are no-ops.

    Args:
        resource_attributes: Optional attributes to attach to the OTEL resource so they
            appear on every span and metric (e.g. {"nhx.platform.platform_version": "26.2.0"}).
    """
    global _obs_initialized
    if _obs_initialized:
        return
    _obs_initialized = True

    from .tracing import create_otel_resource, initialize_metrics, initialize_tracing

    from .structured_logging import initialize_logging  # isort: skip

    resource = create_otel_resource(attributes=resource_attributes)
    if not settings.otel_sdk_disabled:
        initialize_tracing(resource)
        initialize_metrics(resource)
    initialize_logging(resource)


# ---------------------------------------------------------------------------
# Backward-compatible re-exports
#
# Symbols that moved to sub-modules are re-exported here via __getattr__
# so that existing ``from nhx.common.observability.otel import X`` paths
# keep working without triggering circular-import issues at module-init
# time.
# ---------------------------------------------------------------------------

_REEXPORT_MAP: dict[str, tuple[str, str]] = {
    "DiscardInternalRequests": (".structured_logging", "DiscardInternalRequests"),
    "DiscardSensitiveMessages": (".structured_logging", "DiscardSensitiveMessages"),
    "apply_extra_log_config": (".structured_logging", "apply_extra_log_config"),
    "clear_loggers": (".structured_logging", "clear_loggers"),
    "create_otel_log_processor": (".structured_logging", "create_otel_log_processor"),
    "initialize_logging": (".structured_logging", "initialize_logging"),
    "quiet_loggers": (".structured_logging", "quiet_loggers"),
    "RequestLoggingMiddleware": (".middleware", "RequestLoggingMiddleware"),
    "create_otel_resource": (".tracing", "create_otel_resource"),
    "initialize_metrics": (".tracing", "initialize_metrics"),
    "initialize_tracing": (".tracing", "initialize_tracing"),
    "setup_fastapi_instrumentations": (".tracing", "setup_fastapi_instrumentations"),
    "setup_global_instrumentations": (".tracing", "setup_global_instrumentations"),
}


def __getattr__(name: str):
    if name in _REEXPORT_MAP:
        module_path, attr = _REEXPORT_MAP[name]
        import importlib

        mod = importlib.import_module(module_path, __package__)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
