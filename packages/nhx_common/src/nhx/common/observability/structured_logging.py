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
from typing import TYPE_CHECKING, Callable

import structlog
import yaml
from nhx.common.observability.context import AppContextLogProcessor
from opentelemetry import trace
from structlog.typing import EventDict

from .otel import settings

if TYPE_CHECKING:
    from opentelemetry._logs import Logger as OTELLogger
    from opentelemetry.sdk.resources import Resource


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
    logging.getLogger().handlers.clear()


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


def create_otel_log_processor(otel_logger: OTELLogger) -> Callable[[logging.Logger, str, dict], dict]:
    """Create a structlog processor that emits logs to OTEL.

    This processor should be added to the structlog chain BEFORE the final renderer.
    It emits the fully-enriched event dict to OTEL, then passes through unchanged
    so stdout rendering still works.

    Args:
        otel_logger: An OTEL Logger instance from LoggerProvider.get_logger()

    Returns:
        A structlog processor function
    """

    def otel_log_processor(logger: logging.Logger, method_name: str, event_dict: dict) -> dict:
        # NOTE: deferred import -- only reached when OTEL log export is enabled
        from opentelemetry._logs.severity import SeverityNumber

        body = event_dict.get("message") or event_dict.get("event", "")
        level = event_dict.get("level", "info").lower()
        severity = {
            "debug": SeverityNumber.DEBUG,
            "info": SeverityNumber.INFO,
            "warning": SeverityNumber.WARN,
            "warn": SeverityNumber.WARN,
            "error": SeverityNumber.ERROR,
            "critical": SeverityNumber.FATAL,
            "fatal": SeverityNumber.FATAL,
        }.get(level, SeverityNumber.INFO)

        attributes = {k: v for k, v in event_dict.items() if k not in ("message", "event", "level") and v is not None}

        otel_logger.emit(
            body=body,
            severity_number=severity,
            severity_text=level.upper(),
            attributes=attributes,
        )

        return event_dict

    return otel_log_processor


def initialize_logging(resource: Resource | None = None):
    """
    Initialize structured logging with optional OTLP export.

    When OTEL_LOGS_EXPORTER=otlp, logs are sent to the OTLP endpoint in addition
    to stdout. The resource parameter should match what's used for traces/metrics.

    Based on: https://www.structlog.org/en/stable/standard-library.html#rendering-within-structlog
    """
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

    processors = [
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

    # TODO: Reenable with more granular controls
    if False and settings.otel_logs_exporter == "otlp":
        # NOTE: deferred imports -- only reached when OTEL log export is enabled
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

        from .tracing import create_otel_resource

        if resource is None:
            resource = create_otel_resource()
        logger_provider = LoggerProvider(resource=resource)
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(
                OTLPLogExporter(
                    endpoint=settings.otel_exporter_otlp_logs_endpoint or settings.otel_exporter_otlp_endpoint
                )
            )
        )
        set_logger_provider(logger_provider)
        otel_logger = logger_provider.get_logger(__name__)
        final_processors.append(create_otel_log_processor(otel_logger))

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
