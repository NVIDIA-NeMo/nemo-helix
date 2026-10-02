# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for ClickHouse-backed Intake span tests."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from nhx.intake.config import ClickHouseConfig, IntakeConfig
from nhx.intake.repository.clickhouse.tables import ClickHouseTable, qualified_table
from nhx.intake.service import IntakeService
from nhx.intake.spans.clickhouse_client import (
    ClickHouseSettings,
    ClickHouseSpanClient,
    bootstrap_schema,
)
from nhx.testing import create_test_client

_CLICKHOUSE_VERSION_FILE = Path(__file__).resolve().parents[3] / ".clickhouse-version"
CLICKHOUSE_VERSION = _CLICKHOUSE_VERSION_FILE.read_text(encoding="utf-8").strip()


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _docker_available() -> bool:
    if find_spec("docker") is None:
        return False

    from docker.errors import DockerException

    import docker

    try:
        client = docker.from_env()
        try:
            client.ping()
        finally:
            client.close()
        return True
    except (DockerException, OSError):
        return False


@pytest.fixture(scope="session")
def clickhouse_container():
    if not _docker_available():
        pytest.skip("Docker is not available; skipping ClickHouse span tests")

    from testcontainers.clickhouse import ClickHouseContainer

    with ClickHouseContainer(
        f"clickhouse/clickhouse-server:{CLICKHOUSE_VERSION}",
        username="test",
        password="test",
        dbname="default",
    ) as container:
        yield container


@pytest.fixture(scope="session")
def clickhouse_version() -> str:
    return CLICKHOUSE_VERSION


@pytest.fixture(scope="session")
def clickhouse_settings(clickhouse_container) -> ClickHouseSettings:
    return ClickHouseSettings(
        url=f"http://{clickhouse_container.get_container_host_ip()}:{clickhouse_container.get_exposed_port(8123)}",
        user=clickhouse_container.username,
        password=clickhouse_container.password,
        database=f"intake_test_{uuid4().hex}",
    )


@pytest.fixture
def intake_config(clickhouse_settings: ClickHouseSettings) -> IntakeConfig:
    return IntakeConfig(clickhouse_config=_clickhouse_config(clickhouse_settings))


@pytest.fixture(scope="session")
def clickhouse_client(clickhouse_settings: ClickHouseSettings):
    client = ClickHouseSpanClient(clickhouse_settings)
    _run(bootstrap_schema(client))
    yield client
    _run(client.close())


@pytest.fixture(autouse=True)
def clean_clickhouse(clickhouse_client: ClickHouseSpanClient):
    tables = (
        ClickHouseTable.SPANS,
        ClickHouseTable.EVALUATOR_RESULTS,
        ClickHouseTable.ANNOTATIONS,
        ClickHouseTable.TRACE_INDEX,
    )
    for table in tables:
        _run(clickhouse_client.command(f"TRUNCATE TABLE {qualified_table(clickhouse_client.database, table)}"))
    yield
    for table in tables:
        _run(clickhouse_client.command(f"TRUNCATE TABLE {qualified_table(clickhouse_client.database, table)}"))


@pytest.fixture
def client(intake_config: IntakeConfig):
    with create_test_client(
        IntakeService,
        client_type=TestClient,
        service_configs={IntakeService: intake_config},
    ) as test_client:
        yield test_client


@pytest.fixture
def run_async() -> Callable[[Any], Any]:
    return _run


def _clickhouse_config(settings: ClickHouseSettings) -> ClickHouseConfig:
    return ClickHouseConfig(
        url=settings.url,
        user=settings.user,
        password=settings.password,
        database=settings.database,
    )


@pytest.fixture
def make_otlp_request() -> Callable[..., bytes]:
    def _make(spans: list[dict[str, Any]], trace_id: str = "0" * 31 + "1") -> bytes:
        from opentelemetry.proto.collector.trace.v1 import trace_service_pb2

        base_time_unix_nano = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
        request = trace_service_pb2.ExportTraceServiceRequest()
        resource_spans = request.resource_spans.add()
        _add_attributes(resource_spans.resource.attributes, {"service.name": "intake-span-test"})
        scope_spans = resource_spans.scope_spans.add()
        scope_spans.scope.name = "test-tracer"
        scope_spans.scope.version = "1.0.0"
        for index, spec in enumerate(spans, start=1):
            span = scope_spans.spans.add()
            span.trace_id = bytes.fromhex(str(spec.get("trace_id", trace_id)))
            span.span_id = bytes.fromhex(str(spec.get("span_id", f"{index:016x}")))
            parent_span_id = spec.get("parent_span_id")
            if parent_span_id:
                span.parent_span_id = bytes.fromhex(str(parent_span_id))
            span.name = str(spec.get("name", f"span-{index}"))
            span.start_time_unix_nano = int(spec.get("start_time_unix_nano", base_time_unix_nano + index))
            span.end_time_unix_nano = int(spec.get("end_time_unix_nano", span.start_time_unix_nano + 1_000_000))
            if spec.get("error"):
                from opentelemetry.proto.trace.v1 import trace_pb2

                span.status.code = trace_pb2.Status.STATUS_CODE_ERROR
                span.status.message = str(spec.get("error_message", "boom"))
            _add_attributes(span.attributes, spec.get("attributes", {}))
        return request.SerializeToString()

    return _make


def _add_attributes(attributes: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        item = attributes.add()
        item.key = key
        _set_any_value(item.value, value)


def _set_any_value(any_value: Any, value: Any) -> None:
    if isinstance(value, bool):
        any_value.bool_value = value
    elif isinstance(value, int):
        any_value.int_value = value
    elif isinstance(value, float):
        any_value.double_value = value
    elif isinstance(value, list):
        for item in value:
            child = any_value.array_value.values.add()
            _set_any_value(child, item)
    elif isinstance(value, dict):
        _add_attributes(any_value.kvlist_value.values, value)
    else:
        any_value.string_value = str(value)
