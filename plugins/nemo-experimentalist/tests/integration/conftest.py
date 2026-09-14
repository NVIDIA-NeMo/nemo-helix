# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A real Intake service, ClickHouse-backed, reachable through the platform SDK."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from importlib.util import find_spec
from pathlib import Path
from uuid import uuid4

import pytest
from nemo_platform_plugin.client.client import AsyncNemoClient
from nmp.intake.config import ClickHouseConfig, IntakeConfig
from nmp.intake.service import IntakeService
from nmp.intake.spans.clickhouse_client import ClickHouseSettings, ClickHouseSpanClient, bootstrap_schema
from nmp.testing import create_test_client

CLICKHOUSE_VERSION = (
    (Path(__file__).resolve().parents[4] / "services" / "intake" / ".clickhouse-version")
    .read_text(encoding="utf-8")
    .strip()
)


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
def clickhouse_settings() -> Iterator[ClickHouseSettings]:
    if not _docker_available():
        pytest.skip("Docker is not available; Intake is ClickHouse-backed")

    from testcontainers.clickhouse import ClickHouseContainer

    with ClickHouseContainer(
        f"clickhouse/clickhouse-server:{CLICKHOUSE_VERSION}", username="test", password="test", dbname="default"
    ) as container:
        settings = ClickHouseSettings(
            url=f"http://{container.get_container_host_ip()}:{container.get_exposed_port(8123)}",
            user=container.username,
            password=container.password,
            database=f"experimentalist_it_{uuid4().hex}",
        )
        client = ClickHouseSpanClient(settings)
        asyncio.run(bootstrap_schema(client))
        try:
            yield settings
        finally:
            asyncio.run(client.close())


@pytest.fixture
def platform(clickhouse_settings: ClickHouseSettings) -> Iterator[AsyncNemoClient]:
    config = IntakeConfig(
        clickhouse_config=ClickHouseConfig(
            url=clickhouse_settings.url,
            user=clickhouse_settings.user,
            password=clickhouse_settings.password,
            database=clickhouse_settings.database,
        )
    )
    with create_test_client(
        IntakeService, client_type=AsyncNemoClient, service_configs={IntakeService: config}
    ) as client:
        yield client
