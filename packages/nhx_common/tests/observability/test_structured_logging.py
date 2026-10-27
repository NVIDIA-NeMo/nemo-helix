# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the structured-logging processors."""

from __future__ import annotations

import json
import logging
from io import StringIO
from unittest.mock import Mock

import pytest
from nhx.common.observability.otel import OTELSettings, settings
from nhx.common.observability.structured_logging import _sanitize_log_strings, initialize_logging
from opentelemetry._logs import SeverityNumber
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("admin\n[ERROR] forged log line", "admin [ERROR] forged log line"),
        ("with\r\ncrlf", "with  crlf"),
        ("tab\tis fine", "tab\tis fine"),
        ("plain string", "plain string"),
        ("nel\x85next", "nel next"),
        ("ls\u2028lsep", "ls lsep"),
        ("ps\u2029psep", "ps psep"),
    ],
)
def test_sanitize_log_strings_replaces_newline_variants(raw: str, expected: str) -> None:
    event = {"event": "test", "user": raw}
    result = _sanitize_log_strings(logging.getLogger(), "info", event)
    assert result["user"] == expected


def test_sanitize_log_strings_leaves_non_string_values_alone() -> None:
    event = {"event": "test", "count": 5, "ok": True, "items": [1, 2, 3]}
    result = _sanitize_log_strings(logging.getLogger(), "info", event)
    assert result == event


def test_sanitize_log_strings_skips_clean_strings() -> None:
    event = {"event": "test", "name": "default/workspace"}
    result = _sanitize_log_strings(logging.getLogger(), "info", event)
    assert result["name"] == "default/workspace"


def test_sanitize_log_strings_sanitizes_event_key() -> None:
    """The 'event' key carries the primary log message — must be sanitized too."""
    event = {"event": "user input was: bad\ninjected line"}
    result = _sanitize_log_strings(logging.getLogger(), "info", event)
    assert result["event"] == "user input was: bad injected line"


def test_sanitize_log_strings_sanitizes_exception_field() -> None:
    """structlog.format_exc_info writes a multi-line traceback into 'exception'.

    The sanitizer must run after format_exc_info so attacker-controlled
    exception messages can't forge log entries.
    """
    event = {"event": "boom", "exception": "Traceback (most recent call last):\n  ...\nValueError: pwn"}
    result = _sanitize_log_strings(logging.getLogger(), "info", event)
    assert "\n" not in result["exception"]


@pytest.mark.parametrize("log_format", ["plain", "json"])
@pytest.mark.parametrize("logs_endpoint", [None, "http://logs:4318/custom"])
@pytest.mark.parametrize(
    ("logs_protocol", "general_protocol", "transport"),
    [
        (None, "grpc", "grpc"),
        (None, "http/protobuf", "http"),
        ("grpc", "http/protobuf", "grpc"),
        ("http/protobuf", "grpc", "http"),
    ],
)
def test_otlp_logging_exports_trace_context_and_structured_fields(
    monkeypatch, log_format, logs_endpoint, logs_protocol, general_protocol, transport
):
    resource = Resource.create({"service.name": "logging-test"})
    provider = LoggerProvider(resource=resource)
    exporter = InMemoryLogRecordExporter()
    tracer_provider = TracerProvider()
    monkeypatch.setattr("opentelemetry.sdk._logs.LoggerProvider", Mock(return_value=provider))
    monkeypatch.setattr("opentelemetry._logs.set_logger_provider", Mock())
    exporter_factory = Mock(return_value=exporter)
    other_exporter_factory = Mock()
    other_transport = "http" if transport == "grpc" else "grpc"
    monkeypatch.setattr(
        f"opentelemetry.exporter.otlp.proto.{transport}._log_exporter.OTLPLogExporter", exporter_factory
    )
    monkeypatch.setattr(
        f"opentelemetry.exporter.otlp.proto.{other_transport}._log_exporter.OTLPLogExporter", other_exporter_factory
    )
    monkeypatch.setattr(settings, "otel_sdk_disabled", False)
    monkeypatch.setattr(settings, "otel_logs_exporter", "otlp")
    monkeypatch.setattr(settings, "otel_exporter_otlp_logs_protocol", logs_protocol)
    monkeypatch.setattr(settings, "otel_exporter_otlp_protocol", general_protocol)
    monkeypatch.setattr(settings, "otel_exporter_otlp_logs_endpoint", logs_endpoint)
    general_endpoint = "http://collector:4318/base/"
    monkeypatch.setattr(settings, "otel_exporter_otlp_endpoint", general_endpoint)
    monkeypatch.setattr(settings, "log_format", log_format)
    monkeypatch.setattr(settings, "log_level", "INFO")
    monkeypatch.setattr(settings, "extra_log_config", "")
    monkeypatch.setattr(settings, "log_internal_requests", False)
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    output = StringIO()
    try:
        initialize_logging(resource)
        handler = root.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        handler.setStream(output)
        with tracer_provider.get_tracer(__name__).start_as_current_span("test") as span:
            root.info("correlated log", extra={"job": "test-job"})
            root.info("sensitive log", extra={"sensitive": True})
            root.info("internal log", extra={"internal": True})
        provider.force_flush()
        logs = exporter.get_finished_logs()
        assert len(logs) == 1
        record = logs[0].log_record
        assert record.body == "correlated log"
        assert record.severity_number == SeverityNumber.INFO
        assert record.trace_id == span.get_span_context().trace_id
        assert record.span_id == span.get_span_context().span_id
        assert record.attributes is not None
        assert record.attributes["trace_id"] == f"{record.trace_id:032x}"
        assert record.attributes["span_id"] == f"{record.span_id:016x}"
        assert record.attributes["job"] == "test-job"
        assert logs[0].resource.attributes["service.name"] == "logging-test"
        assert "correlated log" in output.getvalue()
        assert "sensitive log" not in output.getvalue()
        assert "internal log" not in output.getvalue()
        if log_format == "json":
            assert json.loads(output.getvalue())["trace_id"] == f"{record.trace_id:032x}"
        expected_endpoint = logs_endpoint or (
            general_endpoint.rstrip("/") + "/v1/logs" if transport == "http" else general_endpoint
        )
        exporter_factory.assert_called_once_with(endpoint=expected_endpoint)
        other_exporter_factory.assert_not_called()
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        provider.shutdown()
        tracer_provider.shutdown()


def test_log_protocol_settings_load_environment_and_default_to_grpc(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_PROTOCOL", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", raising=False)
    defaults = OTELSettings()
    assert defaults.otel_exporter_otlp_protocol == "grpc"
    assert defaults.otel_exporter_otlp_logs_protocol is None

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", "grpc")
    configured = OTELSettings()
    assert configured.otel_exporter_otlp_protocol == "http/protobuf"
    assert configured.otel_exporter_otlp_logs_protocol == "grpc"


@pytest.mark.parametrize(("exporter", "sdk_disabled"), [("none", False), ("otlp", True)])
def test_otlp_logging_respects_disabled_settings(monkeypatch, exporter, sdk_disabled):
    exporter_factory = Mock()
    monkeypatch.setattr("opentelemetry.exporter.otlp.proto.grpc._log_exporter.OTLPLogExporter", exporter_factory)
    monkeypatch.setattr(settings, "otel_logs_exporter", exporter)
    monkeypatch.setattr(settings, "otel_sdk_disabled", sdk_disabled)
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    try:
        initialize_logging()
        exporter_factory.assert_not_called()
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_initialize_logging_wires_sanitizer_into_chain() -> None:
    """Regression guard: 177 dismissed py/log-injection alerts depend on the sanitizer being in the chain."""
    import logging as stdlib_logging

    import structlog
    from nhx.common.observability.structured_logging import _sanitize_log_strings, initialize_logging

    root = stdlib_logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        root.handlers.clear()
        initialize_logging()
        # Locate the ProcessorFormatter by type, not handler index, so a future
        # refactor that reorders or adds handlers doesn't silently false-pass.
        formatters = [
            h.formatter for h in root.handlers if isinstance(h.formatter, structlog.stdlib.ProcessorFormatter)
        ]
        assert formatters, "initialize_logging() did not attach a structlog ProcessorFormatter"
        chain = getattr(formatters[0], "foreign_pre_chain", None) or []
        assert _sanitize_log_strings in chain, (
            "_sanitize_log_strings missing from structlog chain — log-injection defense is disabled"
        )
    finally:
        root.handlers.clear()
        root.handlers.extend(saved_handlers)
        root.setLevel(saved_level)
