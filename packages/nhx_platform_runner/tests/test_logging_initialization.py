# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral unit tests for logging through production runner entrypoints."""

import logging
from types import SimpleNamespace

import pytest
from nhx.common.observability import otel, structured_logging, tracing
from nhx.platform_runner import run as runner
from nhx.platform_runner import server
from nhx.platform_runner.config import ResolvedRunConfiguration
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter


@pytest.fixture
def exported_logs(monkeypatch):
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    exporter = InMemoryLogRecordExporter()
    monkeypatch.setattr(otel, "_obs_initialized", False)
    monkeypatch.setattr(server, "_obs_initialized", False)
    monkeypatch.setattr(otel.settings, "otel_logs_exporter", "otlp")
    monkeypatch.setattr(otel.settings, "otel_sdk_disabled", False)
    monkeypatch.setattr(otel.settings, "otel_exporter_otlp_protocol", "grpc")
    monkeypatch.setattr(otel.settings, "otel_exporter_otlp_logs_protocol", None)
    monkeypatch.setattr(otel.settings, "log_level", "INFO")
    monkeypatch.setattr(otel.settings, "log_format", "json")
    monkeypatch.setattr(otel.settings, "extra_log_config", "")
    monkeypatch.setattr(structured_logging, "_otlp_log_exporter_class", lambda transport: lambda **kwargs: exporter)
    monkeypatch.setattr(tracing, "initialize_tracing", lambda resource: None)
    monkeypatch.setattr(tracing, "initialize_metrics", lambda resource: None)
    monkeypatch.setattr(runner, "setup_global_instrumentations", lambda: None)
    monkeypatch.setattr(server, "setup_global_instrumentations", lambda: None)
    monkeypatch.setattr(runner, "get_platform_resource_attributes", lambda: {"service.name": "runner-test"})
    monkeypatch.setattr(server, "get_platform_resource_attributes", lambda: {"service.name": "runner-test"})
    try:
        yield exporter
    finally:
        structured_logging.clear_loggers()
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


@pytest.mark.parametrize("enabled", [True, False])
def test_api_app_factory_preserves_console_and_controls_direct_export(monkeypatch, capsys, exported_logs, enabled):
    monkeypatch.setattr(otel.settings, "otel_python_log_auto_instrumentation", enabled)
    monkeypatch.setattr(server, "get_available_services", lambda: {})
    monkeypatch.setattr(server, "get_available_controllers", lambda: {})
    monkeypatch.delenv("NHX_SERVICES", raising=False)
    monkeypatch.delenv("NHX_CONTROLLERS", raising=False)

    def create_app(services, *, controller_run_funcs):
        logging.getLogger(__name__).info("API app created")
        return object()

    monkeypatch.setattr(server, "create_app", create_app)
    server.create_default_app()
    server.create_default_app()

    assert capsys.readouterr().err.count("API app created") == 2
    if enabled:
        provider = structured_logging._logger_provider
        assert provider is not None
        provider.force_flush()
    messages = [record.log_record.body for record in exported_logs.get_finished_logs()]
    assert messages.count("API app created") == (2 if enabled else 0)


@pytest.mark.parametrize("enabled", [True, False])
def test_platform_startup_preserves_controller_and_api_logging(monkeypatch, capsys, exported_logs, enabled):
    monkeypatch.setattr(otel.settings, "otel_python_log_auto_instrumentation", enabled)
    monkeypatch.delenv("UVICORN_RELOAD", raising=False)
    resolved = ResolvedRunConfiguration(
        services=set(),
        controllers={"test-controller"},
        sidecars=set(),
        host="127.0.0.1",
        port=8080,
        config_path="",
    )
    monkeypatch.setattr(runner, "resolve_run_configuration", lambda *_args, **_kwargs: resolved)
    monkeypatch.setattr(runner, "apply_run_environment", lambda config: None)
    monkeypatch.setattr(
        runner, "get_common_service_config", lambda: SimpleNamespace(log_format="json", log_level="INFO")
    )
    monkeypatch.setattr(runner, "get_platform_config", lambda: SimpleNamespace(services=""))
    monkeypatch.setattr(runner, "_load_service_instances", lambda *args: [])

    def controller(stop_signal):
        logging.getLogger(__name__).info("controller started")

    monkeypatch.setattr(
        runner,
        "_load_run_functions",
        lambda names, registry, kind: {"test-controller": controller} if kind == "controller" else {},
    )

    def start_controllers(run_funcs, stop_signal):
        for run_func in run_funcs.values():
            run_func(stop_signal)
        return []

    def run_server(*args, **kwargs):
        logging.getLogger(__name__).info("API server started")

    monkeypatch.setattr(runner, "run_controllers_in_threads", start_controllers)
    monkeypatch.setattr(runner, "run_server", run_server)
    monkeypatch.setattr(runner, "signal", lambda *args: None)
    monkeypatch.setattr(runner, "_display_banner", lambda **kwargs: None)
    monkeypatch.setattr(runner, "join_and_untrack_runner_threads", lambda *args: None)
    runner.run_platform()
    runner.run_platform()

    output = capsys.readouterr().err
    assert output.count("controller started") == 2
    assert output.count("API server started") == 2
    if enabled:
        provider = structured_logging._logger_provider
        assert provider is not None
        provider.force_flush()
    messages = [record.log_record.body for record in exported_logs.get_finished_logs()]
    assert messages.count("controller started") == (2 if enabled else 0)
    assert messages.count("API server started") == (2 if enabled else 0)
