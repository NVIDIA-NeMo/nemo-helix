# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the structured-logging processors."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from io import StringIO
from unittest.mock import Mock

import pytest
from nhx.common.observability import context, otel, structured_logging, tracing
from nhx.common.observability.otel import OTELSettings, settings
from nhx.common.observability.structured_logging import _sanitize_log_strings, initialize_logging
from nhx.common.observability.task_logging import HelixTaskLoggingProvider
from opentelemetry._logs import SeverityNumber
from opentelemetry.exporter.otlp.proto.common._log_encoder import encode_logs
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider


@pytest.fixture(autouse=True)
def logging_state(monkeypatch):
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    monkeypatch.setattr(settings, "otel_logs_exporter", "none")
    monkeypatch.setattr(settings, "otel_sdk_disabled", False)
    monkeypatch.setattr(settings, "otel_python_log_auto_instrumentation", True)
    monkeypatch.setattr(settings, "extra_log_config", "")
    monkeypatch.setattr(settings, "log_internal_requests", False)
    monkeypatch.setattr(settings, "log_level", "INFO")
    monkeypatch.setattr(settings, "log_format", "plain")
    monkeypatch.setattr(otel, "_obs_initialized", False)
    yield
    structured_logging.clear_loggers()
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def _capture_console(output: StringIO) -> None:
    handler = logging.getLogger().handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    handler.setStream(output)


def _flush_logs() -> None:
    provider = structured_logging._logger_provider
    assert provider is not None
    provider.force_flush()


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
    monkeypatch.setattr(structured_logging, "_new_logger_provider", Mock(return_value=provider))
    exporter_factory = Mock(return_value=exporter)
    other_exporter_factory = Mock()
    other_transport = "http" if transport == "grpc" else "grpc"
    exporter_factories = {transport: exporter_factory, other_transport: other_exporter_factory}
    monkeypatch.setattr(structured_logging, "_otlp_log_exporter_class", exporter_factories.__getitem__)
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
    monkeypatch.setattr(structured_logging, "_otlp_log_exporter_class", lambda transport: exporter_factory)
    monkeypatch.setattr(settings, "otel_logs_exporter", exporter)
    monkeypatch.setattr(settings, "otel_sdk_disabled", sdk_disabled)
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    try:
        initialize_logging()
        exporter_factory.assert_not_called()
        assert len([h for h in root.handlers if isinstance(h, LoggingHandler)]) == int(not sdk_disabled)
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


@pytest.mark.parametrize(("value", "expected"), [(None, True), ("true", True), ("false", False)])
def test_auto_instrumentation_environment(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("OTEL_PYTHON_LOG_AUTO_INSTRUMENTATION", raising=False)
    else:
        monkeypatch.setenv("OTEL_PYTHON_LOG_AUTO_INSTRUMENTATION", value)
    assert OTELSettings().otel_python_log_auto_instrumentation is expected


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("exporter_name", ["none", "otlp"])
@pytest.mark.parametrize("log_format", ["plain", "json"])
def test_handler_flag_exporter_and_console(monkeypatch, enabled, exporter_name, log_format):
    exporter_factory = Mock(return_value=InMemoryLogRecordExporter())
    monkeypatch.setattr(structured_logging, "_otlp_log_exporter_class", lambda transport: exporter_factory)
    monkeypatch.setattr(settings, "otel_python_log_auto_instrumentation", enabled)
    monkeypatch.setattr(settings, "otel_logs_exporter", exporter_name)
    monkeypatch.setattr(settings, "log_format", log_format)
    initialize_logging()
    root = logging.getLogger()
    assert len([h for h in root.handlers if isinstance(h, LoggingHandler)]) == int(enabled)
    output = StringIO()
    _capture_console(output)
    root.warning("console is alive")
    root.warning("secret", extra={"sensitive": True})
    assert "console is alive" in output.getvalue()
    assert "secret" not in output.getvalue()
    assert exporter_factory.call_count == int(enabled and exporter_name == "otlp")
    if enabled:
        _flush_logs()
    if enabled and exporter_name == "otlp":
        assert len(exporter_factory.return_value.get_finished_logs()) == 1


@pytest.mark.parametrize("internal_enabled", [True, False])
@pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL])
def test_standard_handler_preserves_exceptions_context_and_filtering(monkeypatch, internal_enabled, level):
    exporter = InMemoryLogRecordExporter()
    monkeypatch.setattr(structured_logging, "_otlp_log_exporter_class", lambda transport: lambda **kwargs: exporter)
    monkeypatch.setattr(settings, "otel_logs_exporter", "otlp")
    monkeypatch.setattr(settings, "log_internal_requests", internal_enabled)
    monkeypatch.setattr(settings, "log_level", "DEBUG")
    initialize_logging(Resource.create({"service.name": "test-service"}))
    output = StringIO()
    root = logging.getLogger()
    _capture_console(output)

    @dataclass
    class JobContext(context.BaseContext):
        otel_prefix: str = "job"
        id: str = "job-123"

    app_ctx = context.AppContext(service_name="models", workspace="test-workspace").with_custom_ctx(JobContext())
    token = context._app_context.set(app_ctx)
    try:
        try:
            raise ValueError("failure details")
        except ValueError:
            root.log(level, "failed %s", "operation", exc_info=True, extra={"client": "test-client"})
            root.exception("secret", extra={"sensitive": True})
        root.info("internal", extra={"internal": True})
        root.info("user input: %s", "bad\nforged", extra={"user": "bad\nforged"})
    finally:
        context._app_context.reset(token)
    _flush_logs()
    logs = exporter.get_finished_logs()
    assert len(logs) == 2 + int(internal_enabled)
    record = logs[0].log_record
    assert record.body == "failed operation"
    assert (
        record.severity_number
        == {
            logging.DEBUG: SeverityNumber.DEBUG,
            logging.INFO: SeverityNumber.INFO,
            logging.WARNING: SeverityNumber.WARN,
            logging.ERROR: SeverityNumber.ERROR,
            logging.CRITICAL: SeverityNumber.FATAL,
        }[level]
    )
    assert record.attributes is not None
    assert record.attributes["exception.type"] == "ValueError"
    assert record.attributes["exception.message"] == "failure details"
    stacktrace = record.attributes["exception.stacktrace"]
    assert isinstance(stacktrace, str)
    assert "ValueError: failure details" in stacktrace
    assert record.attributes["service"] == "models"
    assert record.attributes["workspace"] == "test-workspace"
    assert record.attributes["job.id"] == "job-123"
    assert record.attributes["client"] == "test-client"
    assert logs[-1].log_record.body == "user input: bad forged"
    assert logs[-1].log_record.attributes is not None
    assert logs[-1].log_record.attributes["user"] == "bad forged"
    assert "secret" not in output.getvalue()
    assert ("internal" in output.getvalue()) is internal_enabled


@pytest.mark.parametrize("disable_with", ["flag", "exporter"])
def test_reinitialization_emits_each_message_once(monkeypatch, disable_with):
    exporters = [InMemoryLogRecordExporter(), InMemoryLogRecordExporter()]
    factory = Mock(side_effect=exporters)
    monkeypatch.setattr(structured_logging, "_otlp_log_exporter_class", lambda transport: factory)
    monkeypatch.setattr(settings, "otel_logs_exporter", "otlp")
    initialize_logging()
    initialize_logging()
    root = logging.getLogger()
    output = StringIO()
    _capture_console(output)
    root.info("once")
    _flush_logs()
    assert len(exporters[1].get_finished_logs()) == 1
    assert output.getvalue().count("once") == 1
    if disable_with == "flag":
        monkeypatch.setattr(settings, "otel_python_log_auto_instrumentation", False)
    else:
        monkeypatch.setattr(settings, "otel_logs_exporter", "none")
    initialize_logging()
    _capture_console(output)
    root.info("console after disabling export")
    assert output.getvalue().count("console after disabling export") == 1
    assert factory.call_count == 2


@pytest.mark.parametrize("entrypoint", ["observability", "task"])
@pytest.mark.parametrize("enabled", [True, False])
def test_shared_observability_and_task_initialization(monkeypatch, entrypoint, enabled):
    """Repeated initialization keeps one working console and direct export path."""
    exporters = []

    def exporter_factory(**kwargs):
        exporter = InMemoryLogRecordExporter()
        exporters.append(exporter)
        return exporter

    monkeypatch.setattr(structured_logging, "_otlp_log_exporter_class", lambda transport: exporter_factory)
    monkeypatch.setattr(settings, "otel_logs_exporter", "otlp")
    monkeypatch.setattr(settings, "otel_python_log_auto_instrumentation", enabled)
    # Trace/metric initialization is independent of logging; avoid setting global providers.
    monkeypatch.setattr(tracing, "initialize_tracing", Mock())
    monkeypatch.setattr(tracing, "initialize_metrics", Mock())
    output = StringIO()

    def emit():
        logging.getLogger().info("%s initialized", entrypoint)

    if entrypoint == "task":
        provider = HelixTaskLoggingProvider()
        provider.configure_logging(level="INFO", log_format="json")
        provider.configure_logging(level="INFO", log_format="json")
        _capture_console(output)
        emit()
    else:
        otel.initialize_obs()
        otel.initialize_obs()
        _capture_console(output)
        emit()

    assert output.getvalue().count(f"{entrypoint} initialized") == 1
    assert len(logging.getLogger().handlers) == 1 + int(enabled)
    if enabled:
        _flush_logs()
        assert len(exporters[-1].get_finished_logs()) == 1
    else:
        assert exporters == []


def test_direct_log_without_active_span(monkeypatch):
    exporter = InMemoryLogRecordExporter()
    monkeypatch.setattr(structured_logging, "_otlp_log_exporter_class", lambda transport: lambda **kwargs: exporter)
    monkeypatch.setattr(settings, "otel_logs_exporter", "otlp")
    initialize_logging()
    logging.getLogger().info("outside a span")
    _flush_logs()
    records = exporter.get_finished_logs()
    assert len(records) == 1
    record = records[0].log_record
    assert record.body == "outside a span"
    assert record.trace_id == 0
    assert record.span_id == 0
    assert record.attributes is not None
    assert "trace_id" not in record.attributes
    assert "span_id" not in record.attributes


@pytest.mark.parametrize(
    ("source_path", "filename"),
    [
        ("/opt/application/src/worker.py", "worker.py"),
        (r"C:\application\src\worker.py", "worker.py"),
        ("worker.py", "worker.py"),
        ('/opt/application/src/quoted"worker.py', 'quoted"worker.py'),
    ],
)
def test_exported_source_metadata_keeps_filename_without_directories(monkeypatch, source_path, filename):
    exporter = InMemoryLogRecordExporter()
    monkeypatch.setattr(structured_logging, "_otlp_log_exporter_class", lambda transport: lambda **kwargs: exporter)
    monkeypatch.setattr(settings, "otel_logs_exporter", "otlp")
    initialize_logging()
    output = StringIO()
    _capture_console(output)
    record = logging.LogRecord("source-test", logging.INFO, source_path, 37, "message %s", ("preserved",), None, "run")
    logging.getLogger().handle(record)
    _flush_logs()

    logs = exporter.get_finished_logs()
    assert len(logs) == 1
    exported = logs[0].log_record
    assert exported.body == "message preserved"
    assert exported.attributes is not None
    assert exported.attributes["code.file.path"] == filename
    assert exported.attributes["code.function.name"] == "run"
    assert exported.attributes["code.line.number"] == 37
    assert "message preserved" in output.getvalue()
    # The privacy policy affects outgoing OTel metadata, not other handlers.
    assert record.pathname == source_path


@pytest.mark.parametrize(
    ("source_path", "filename"),
    [
        ("/opt/application/src/worker.py", "worker.py"),
        (r"C:\application\src\worker.py", "worker.py"),
        ('/opt/application/src/quoted"worker.py', 'quoted"worker.py'),
    ],
)
def test_exported_exceptions_keep_details_without_source_directories(monkeypatch, source_path, filename):
    exporter = InMemoryLogRecordExporter()
    monkeypatch.setattr(structured_logging, "_otlp_log_exporter_class", lambda transport: lambda **kwargs: exporter)
    monkeypatch.setattr(settings, "otel_logs_exporter", "otlp")
    initialize_logging()
    source = "def fail():\n    raise ValueError('failure details')\nfail()\n"
    try:
        exec(compile(source, source_path, "exec"), {})
    except ValueError:
        logging.getLogger().exception("operation failed")
    _flush_logs()

    logs = exporter.get_finished_logs()
    assert len(logs) == 1
    exported = logs[0].log_record
    assert exported.body == "operation failed"
    assert exported.severity_number == SeverityNumber.ERROR
    assert exported.attributes is not None
    assert exported.attributes["exception.type"] == "ValueError"
    assert exported.attributes["exception.message"] == "failure details"
    stacktrace = exported.attributes["exception.stacktrace"]
    assert isinstance(stacktrace, str)
    assert f'File "{filename}", line 2, in fail' in stacktrace
    assert "ValueError: failure details" in stacktrace
    assert source_path not in stacktrace
    assert __file__ not in stacktrace


def test_only_marked_sensitive_records_are_dropped(monkeypatch):
    exporter = InMemoryLogRecordExporter()
    monkeypatch.setattr(structured_logging, "_otlp_log_exporter_class", lambda transport: lambda **kwargs: exporter)
    monkeypatch.setattr(settings, "otel_logs_exporter", "otlp")
    initialize_logging()
    output = StringIO()
    _capture_console(output)
    root = logging.getLogger()
    root.info("private content", extra={"sensitive": True})
    root.info("public content", extra={"sensitive": False, "sample_value": "example-value"})
    message = 'File "user/input.txt", line 7'
    try:
        raise ValueError(message)
    except ValueError:
        root.exception("private exception", extra={"sensitive": True})
        root.exception("public exception")
    _flush_logs()

    logs = exporter.get_finished_logs()
    assert [item.log_record.body for item in logs] == ["public content", "public exception"]
    attributes = logs[0].log_record.attributes
    assert attributes is not None
    assert attributes["sample_value"] == "example-value"
    attributes = logs[1].log_record.attributes
    assert attributes is not None
    assert attributes["exception.message"] == message
    stacktrace = attributes["exception.stacktrace"]
    assert isinstance(stacktrace, str)
    assert message in stacktrace
    console = output.getvalue()
    assert "private content" not in console
    assert "private exception" not in console
    assert "public content" in console
    assert "public exception" in console
    assert message in console


def test_source_privacy_preserves_otlp_encoding_and_dropped_attribute_counts(monkeypatch):
    exporter = InMemoryLogRecordExporter()
    monkeypatch.setattr(structured_logging, "_otlp_log_exporter_class", lambda transport: lambda **kwargs: exporter)
    monkeypatch.setattr(settings, "otel_logs_exporter", "otlp")
    initialize_logging()
    logging.getLogger().info("bounded record", extra={f"field_{index}": index for index in range(200)})
    _flush_logs()

    logs = exporter.get_finished_logs()
    assert len(logs) == 1
    request = encode_logs(logs)
    # Exercise the actual OTLP wire encoding without opening a connection.
    assert request.SerializeToString()
    record = request.resource_logs[0].scope_logs[0].log_records[0]
    assert record.body.string_value == "bounded record"
    assert record.dropped_attributes_count > 0
    attributes = {attribute.key: attribute.value for attribute in record.attributes}
    assert attributes["code.file.path"].string_value == "test_structured_logging.py"
