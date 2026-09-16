# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AppContext and context management functionality."""

import logging
from dataclasses import dataclass
from typing import Any

import pytest
import structlog
from nhx.common.observability.context import (
    AppContext,
    AppContextLogProcessor,
    AppContextSpanProcessor,
    BaseContext,
    get_app_ctx,
    initialize_app_ctx,
    scoped_app_ctx,
    start_span_with_ctx,
    update_app_ctx,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer
from structlog.typing import EventDict


def create_capture_processor(captured_logs: dict[str, EventDict]):
    def capture_processor(logger, method_name, event_dict):
        captured_logs[event_dict["event"]] = event_dict.copy()
        return event_dict

    return capture_processor


@pytest.fixture
def tracer_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(AppContextSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer(__name__), exporter


@pytest.fixture
def tracer(tracer_and_exporter):
    return tracer_and_exporter[0]


@pytest.fixture
def exporter(tracer_and_exporter):
    return tracer_and_exporter[1]


@pytest.fixture
def logger_and_captured_logs():
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    log.handlers.clear()  # Clear any existing handlers

    captured_logs: dict[str, EventDict] = {}
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[AppContextLogProcessor, create_capture_processor(captured_logs)],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(),
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    log.addHandler(handler)

    return log, captured_logs


@pytest.fixture
def logger(logger_and_captured_logs):
    return logger_and_captured_logs[0]


@pytest.fixture
def captured_logs(logger_and_captured_logs):
    return logger_and_captured_logs[1]


@pytest.fixture(autouse=True)
def reset_app_context():
    """Reset app context before each test to ensure isolation."""
    from nhx.common.observability.context import _app_context

    _app_context.set(None)
    yield
    _app_context.set(None)


@dataclass
class OuterContext(BaseContext):
    otel_prefix: str = "outer"

    test_str: str = "outer_str"
    test_int: int = 42


@dataclass
class InnerContext(BaseContext):
    otel_prefix: str = "inner"

    other_str: str = "inner_str"


def _log_and_span(msg: str, logger: logging.Logger, tracer: Tracer):
    with tracer.start_as_current_span(msg):
        logger.info(msg)


def _assert_attribs(
    expected_attribs: dict[str, dict[str, Any]], captured_logs: dict[str, EventDict], exporter: InMemorySpanExporter
):
    """Helper that ensures both the logger and span properly attached attributes."""
    span_to_attribs = {s.name: s.attributes for s in exporter.get_finished_spans()}

    for msg, attrib_assertions in expected_attribs.items():
        log_attribs = captured_logs[msg]
        span_attribs = span_to_attribs[msg]
        for key, value in attrib_assertions.items():
            assert log_attribs[key] == value
            assert span_attribs[key] == value


def test_add_to_ctx_full(
    logger: logging.Logger, captured_logs: dict[str, EventDict], tracer: Tracer, exporter: InMemorySpanExporter
):
    assert get_app_ctx() is None
    _log_and_span("no_context", logger, tracer)
    with scoped_app_ctx(OuterContext()):
        app_ctx = get_app_ctx()
        assert app_ctx

        outer_ctx = app_ctx.get_custom_ctx(OuterContext)
        assert outer_ctx is not None
        assert outer_ctx.test_str == "outer_str"

        _log_and_span("outer", logger, tracer)

        with scoped_app_ctx(OuterContext(test_str="override")):
            with scoped_app_ctx(InnerContext()):
                _log_and_span("inner", logger, tracer)
                app_ctx = get_app_ctx()
                assert app_ctx

                # the newer OuterContext should override the original one on the outside
                outer_ctx = app_ctx.get_custom_ctx(OuterContext)
                assert outer_ctx is not None
                assert outer_ctx.test_str == "override"

                inner_ctx = app_ctx.get_custom_ctx(InnerContext)
                assert inner_ctx is not None
                assert inner_ctx.other_str == "inner_str"

            # outside of the InnerContext we should get back None
            app_ctx = get_app_ctx()
            assert app_ctx
            inner_ctx = app_ctx.get_custom_ctx(InnerContext)
            assert inner_ctx is None

        # Ensure that the second OuterContext pops off the stack
        _log_and_span("outer_again", logger, tracer)
        app_ctx = get_app_ctx()
        assert app_ctx is not None
        outer_ctx = app_ctx.get_custom_ctx(OuterContext)
        assert outer_ctx is not None
        assert outer_ctx.test_str == "outer_str"

    assertions: dict[str, dict[str, str]] = {
        "no_context": {},
        "outer": {"outer.test_str": "outer_str"},
        "inner": {"outer.test_str": "override", "inner.other_str": "inner_str"},
        "outer_again": {"outer.test_str": "outer_str"},
    }

    _assert_attribs(assertions, captured_logs, exporter)


def test_start_span_with_ctx(
    logger: logging.Logger, captured_logs: dict[str, EventDict], tracer: Tracer, exporter: InMemorySpanExporter
):
    with start_span_with_ctx(tracer, "test", OuterContext(test_str="span_test")):
        logger.info("test")

    _assert_attribs(
        expected_attribs={"test": {"outer.test_str": "span_test"}}, captured_logs=captured_logs, exporter=exporter
    )


# Tests for new initialize_app_ctx and update_app_ctx functions


def test_initialize_app_ctx():
    """initialize_app_ctx sets the entire AppContext."""
    ctx = AppContext()
    initialize_app_ctx(ctx)
    assert get_app_ctx() is ctx


def test_update_app_ctx_creates_context_if_none():
    """update_app_ctx creates AppContext if none exists."""
    assert get_app_ctx() is None
    update_app_ctx(OuterContext())
    ctx = get_app_ctx()
    assert ctx is not None
    assert ctx.get_custom_ctx(OuterContext) is not None


def test_update_app_ctx_does_not_restore_on_scope_exit():
    """update_app_ctx does NOT restore context (unlike scoped_app_ctx)."""
    update_app_ctx(OuterContext(test_str="first"))
    update_app_ctx(InnerContext(other_str="inner"))

    # Both contexts should be present and persist
    ctx = get_app_ctx()
    assert ctx is not None
    assert ctx.get_custom_ctx(OuterContext) is not None
    assert ctx.get_custom_ctx(InnerContext) is not None
