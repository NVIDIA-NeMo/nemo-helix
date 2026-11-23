# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured logging initialization and log filters.

NOTE: Imports in this module are intentionally deferred inside functions
for startup performance (e.g. OTel log exporters). Do not hoist them to
module level.
"""

from __future__ import annotations

import logging
import logging.config
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Literal

import structlog
import yaml
from nhx.common.observability.context import AppContextLogProcessor
from opentelemetry import trace
from structlog.typing import EventDict

from .otel import settings

if TYPE_CHECKING:
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import LogRecordExporter
    from opentelemetry.sdk.resources import Resource


def _otlp_log_exporter_class(transport: Literal["http", "grpc"]) -> Callable[..., LogRecordExporter]:
    """Import the OTLP log exporter for a transport; deferred so only enabled export pays for it."""
    if transport == "http":
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter as HTTPOTLPLogExporter

        return HTTPOTLPLogExporter
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter as GRPCOTLPLogExporter

    return GRPCOTLPLogExporter


def _new_logger_provider(resource: Resource) -> LoggerProvider:
    from opentelemetry.sdk._logs import LoggerProvider

    return LoggerProvider(resource=resource)


_logger_provider: LoggerProvider | None = None


def _SpanLogProcessor(logger: logging.Logger, method_name: str, event_dict: dict):
    """
    Adds trace context to each log line

    We also add a custom attribute `trace_id` and `span_id`. When traces are corrupted/incomplete, trace_id correlation
    can break. By adding this separate `trace_id` field, we can use it as a barebones
    correlation mechanism for log filtering.
    """
    span = trace.get_current_span()
    span_context = span.get_span_context()
    if span_context.is_valid:
        event_dict["trace_id"] = format(span_context.trace_id, "032x")
        event_dict["span_id"] = format(span_context.span_id, "016x")
    return event_dict


def _drop_color_message(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Remove uvicorn's color_message field from log events.

    Uvicorn adds a 'color_message' extra field containing ANSI color codes for terminal output.
    This is redundant when using structlog's ConsoleRenderer which handles coloring itself.
    """
    event_dict.pop("color_message", None)
    return event_dict


_CONSOLE_HIDDEN_FIELDS = {"client", "port", "http_version", "filename", "func_name", "lineno", "span_id"}


def _drop_console_hidden_fields(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Remove fields that are verbose for console output but useful for OTEL.

    Fields like 'client' are useful for debugging in traces/OTEL but add noise
    to console logs during local development.
    """
    for field in _CONSOLE_HIDDEN_FIELDS:
        event_dict.pop(field, None)
    return event_dict


# CR, LF, NEL, line/paragraph separators. Anything that a multi-line log
# rendering pass could turn into a fake new log entry.
_LOG_NEWLINE_CHARS = "\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"
_LOG_NEWLINE_TABLE = {ord(c): " " for c in _LOG_NEWLINE_CHARS}


def _sanitize_log_strings(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Strip newline-like characters from string values in the event dict.

    Defends against log-forgery via user-controlled fields landing in
    request/response logs. JSON renderers escape these characters already,
    but plain-text renderers do not.
    """
    for key, value in list(event_dict.items()):
        if isinstance(value, str) and any(ch in value for ch in _LOG_NEWLINE_CHARS):
            event_dict[key] = value.translate(_LOG_NEWLINE_TABLE)
    return event_dict


def clear_loggers():
    global _logger_provider
    logging.getLogger().handlers.clear()
    if _logger_provider is not None:
        _logger_provider.shutdown()
        _logger_provider = None


class DiscardSensitiveMessages(logging.Filter):
    """
    Discards messages marked as sensitive via the `sensitive` flag.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "sensitive", False)


class DiscardInternalRequests(logging.Filter):
    """
    Discards request logs marked as internal via the `internal` flag.

    Internal requests are those triggered by controller-to-service calls
    rather than direct user requests. This filter allows filtering them
    from stdout while still allowing them to be captured by log aggregators.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "internal", False)


class EnrichOTelLogRecord(logging.Filter):
    """Enrich the standard handler's records without exporting from a formatter."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Keep exc_info intact: LoggingHandler translates it to standard OTel
        # exception attributes, alongside native severity and span correlation.
        fields = {"message": record.getMessage()}
        _SpanLogProcessor(logging.getLogger(record.name), "", fields)
        AppContextLogProcessor(logging.getLogger(record.name), "", fields)
        record.__dict__.update(fields)
        _sanitize_log_strings(logging.getLogger(record.name), "", record.__dict__)
        record.msg = record.message
        record.args = ()
        return True


def initialize_logging(resource: Resource | None = None):
    """
    Initialize structured logging with optional OTLP export.

    The standard OTel handler is installed unless Python log auto-instrumentation
    or the SDK is disabled. OTEL_LOGS_EXPORTER=otlp enables direct export alongside
    console output. The resource should match what's used for traces/metrics.

    Based on: https://www.structlog.org/en/stable/standard-library.html#rendering-within-structlog
    """
    global _logger_provider

    if settings.log_format == "json":
        renderer = structlog.processors.JSONRenderer()
        timestamp_processor = structlog.processors.TimeStamper(fmt="iso")
    else:
        renderer = structlog.dev.ConsoleRenderer(event_key="message", pad_event=0, pad_level=False)

        def _stamper(event_dict: EventDict) -> EventDict:
            event_dict["timestamp"] = datetime.now().isoformat(timespec="milliseconds")
            return event_dict

        timestamp_processor = structlog.processors.TimeStamper()
        timestamp_processor._stamper = _stamper

    processors: list = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.ExtraAdder(),
        _drop_color_message,
        timestamp_processor,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        _sanitize_log_strings,
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
        _SpanLogProcessor,
        AppContextLogProcessor,
        structlog.processors.EventRenamer(to="message"),
    ]

    final_processors: list = [structlog.stdlib.ProcessorFormatter.remove_processors_meta]

    otel_handler = None
    if not settings.otel_sdk_disabled and settings.otel_python_log_auto_instrumentation:
        from opentelemetry.sdk._logs import LoggingHandler

        from .tracing import create_otel_resource

        if resource is None:
            resource = create_otel_resource()
        logger_provider = _new_logger_provider(resource)
        otel_handler = LoggingHandler(logger_provider=logger_provider)
        otel_handler.addFilter(DiscardSensitiveMessages())
        if not settings.log_internal_requests:
            otel_handler.addFilter(DiscardInternalRequests())
        otel_handler.addFilter(EnrichOTelLogRecord())

        if settings.otel_logs_exporter == "otlp":
            # NOTE: deferred imports -- only reached when OTEL log export is enabled
            from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

            protocol = settings.otel_exporter_otlp_logs_protocol or settings.otel_exporter_otlp_protocol
            endpoint = settings.otel_exporter_otlp_logs_endpoint or settings.otel_exporter_otlp_endpoint
            if protocol == "http/protobuf":
                exporter_class = _otlp_log_exporter_class("http")

                # HTTP general endpoints are base URLs; signal-specific endpoints are used as-is.
                if not settings.otel_exporter_otlp_logs_endpoint and endpoint:
                    endpoint = endpoint.rstrip("/") + "/v1/logs"
            elif protocol == "grpc":
                exporter_class = _otlp_log_exporter_class("grpc")
            else:
                raise ValueError(f"Unsupported OTLP logs protocol: {protocol}")

            logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter_class(endpoint=endpoint)))

    if settings.log_format == "plain":
        final_processors.append(_drop_console_hidden_fields)

    final_processors.append(renderer)

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=processors,
        processors=final_processors,
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(DiscardSensitiveMessages())
    if not settings.log_internal_requests:
        handler.addFilter(DiscardInternalRequests())

    clear_loggers()
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    if otel_handler is not None:
        _logger_provider = logger_provider
        root_logger.addHandler(otel_handler)
    root_logger.setLevel(settings.log_level)
    quiet_loggers()
    apply_extra_log_config()


def quiet_loggers() -> None:
    """
    Downgrade the level on certain verbose loggers.
    """

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def apply_extra_log_config() -> None:
    """
    Apply extra logging configuration from EXTRA_LOG_CONFIG env var.

    The value is parsed as YAML (a superset of JSON) and applied
    on top of the existing logging configuration from the application.
    This is helpful in development to change settings on a per-invocation
    basis, rather than having to modify the code in this file.

    Example:
        EXTRA_LOG_CONFIG='{loggers: {nhx.core.files: {level: DEBUG}}}'
    """
    if not settings.extra_log_config:
        return

    try:
        config = yaml.safe_load(settings.extra_log_config)
    except Exception as e:
        logging.getLogger().warning(f"Failed to parse EXTRA_LOG_CONFIG: {e}")
        return

    if not isinstance(config, dict):
        logging.getLogger().warning("Failed to apply EXTRA_LOG_CONFIG: expected a YAML mapping")
        return

    config.setdefault("version", 1)
    config.setdefault("incremental", True)

    try:
        logging.config.dictConfig(config)
    except Exception as e:
        logging.getLogger().warning(f"Failed to apply EXTRA_LOG_CONFIG: {e}")
