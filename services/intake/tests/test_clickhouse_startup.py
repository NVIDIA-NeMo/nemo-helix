# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Intake startup and ClickHouse readiness."""

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest
from nhx.intake.config import ClickHouseConfig, IntakeConfig
from nhx.intake.local_clickhouse import DockerUnavailableError, LocalClickHouseProvisioningError
from nhx.intake.readiness import CLICKHOUSE_UNAVAILABLE_MESSAGE
from nhx.intake.service import IntakeService


def _external_config() -> IntakeConfig:
    return IntakeConfig(
        clickhouse_config=ClickHouseConfig(
            url="http://127.0.0.1:1",
            user="default",
            password="",
            database="intake_unavailable",
        )
    )


def test_intake_is_not_ready_when_external_clickhouse_is_inaccessible(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caplog.set_level(logging.WARNING, logger="nhx.intake.service")
    stop = AsyncMock(return_value=True)
    monkeypatch.setattr("nhx.intake.service.stop_local_clickhouse", stop)
    service = IntakeService().with_config(_external_config())

    async def check_readiness() -> bool:
        await service.on_startup()
        assert service.clickhouse_client is not None
        monkeypatch.setattr(
            service.clickhouse_client,
            "query",
            AsyncMock(side_effect=ConnectionError("connection refused")),
        )
        try:
            return await service.is_ready()
        finally:
            await service.on_shutdown()

    assert asyncio.run(check_readiness()) is False
    assert service.readiness_message == ""
    stop.assert_not_awaited()
    assert any("readiness probe failed" in record.message for record in caplog.records)


def test_intake_readiness_surfaces_clickhouse_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    service = IntakeService().with_config(_external_config())

    async def check_readiness() -> bool:
        await service.on_startup()
        assert service.clickhouse_client is not None
        monkeypatch.setattr(service.clickhouse_client, "query", AsyncMock(side_effect=PermissionError("denied")))
        return await service.is_ready()

    assert asyncio.run(check_readiness()) is False
    assert service.readiness_message == CLICKHOUSE_UNAVAILABLE_MESSAGE


def test_intake_uses_reconciled_clickhouse_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NHX_INTAKE_CLICKHOUSE_URL", raising=False)
    reconcile = AsyncMock(return_value="http://127.0.0.1:55123")
    stop = AsyncMock(return_value=True)
    monkeypatch.setattr("nhx.intake.service.reconcile_local_clickhouse", reconcile)
    monkeypatch.setattr("nhx.intake.service.stop_local_clickhouse", stop)
    intake_config = IntakeConfig(clickhouse_config=ClickHouseConfig())
    service = IntakeService().with_config(intake_config)

    async def start_and_stop() -> None:
        await service.on_startup()
        try:
            assert service.clickhouse_client is not None
            assert service.clickhouse_client.settings.url == "http://127.0.0.1:55123"
        finally:
            await service.on_shutdown()

    asyncio.run(start_and_stop())
    reconcile.assert_awaited_once()
    assert reconcile.await_args is not None
    assert reconcile.await_args.kwargs == {
        "image": intake_config.clickhouse_config.image,
        "data_dir": intake_config.clickhouse_config.data_dir,
    }
    stop.assert_awaited_once_with(data_dir=intake_config.clickhouse_config.data_dir)


@pytest.mark.parametrize(
    ("provisioning_error", "log_level", "expected_log"),
    [
        (
            DockerUnavailableError("Docker daemon is unavailable"),
            logging.WARNING,
            "Docker daemon is unavailable",
        ),
        (
            LocalClickHouseProvisioningError("container name collision"),
            logging.ERROR,
            "container name collision",
        ),
    ],
)
def test_intake_is_not_ready_after_local_clickhouse_provisioning_failure(
    provisioning_error: Exception,
    log_level: int,
    expected_log: str,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NHX_INTAKE_CLICKHOUSE_URL", raising=False)
    reconcile = AsyncMock(side_effect=provisioning_error)
    monkeypatch.setattr("nhx.intake.service.reconcile_local_clickhouse", reconcile)
    caplog.set_level(log_level, logger="nhx.intake.service")
    service = IntakeService().with_config(IntakeConfig(clickhouse_config=ClickHouseConfig()))

    async def check_readiness() -> bool:
        await service.on_startup()
        assert service.clickhouse_client is not None
        monkeypatch.setattr(
            service.clickhouse_client,
            "query",
            AsyncMock(side_effect=ConnectionError("connection refused")),
        )
        try:
            return await service.is_ready()
        finally:
            await service.on_shutdown()

    assert asyncio.run(check_readiness()) is False
    assert any(expected_log in record.message for record in caplog.records)


def test_intake_readiness_probes_spans_table_without_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    service = IntakeService().with_config(_external_config())
    reconcile = AsyncMock()
    stop = AsyncMock()
    check_data_directory = AsyncMock()
    monkeypatch.setattr("nhx.intake.service.reconcile_local_clickhouse", reconcile)
    monkeypatch.setattr("nhx.intake.service.stop_local_clickhouse", stop)
    monkeypatch.setattr("nhx.intake.service.check_local_clickhouse_data_directory", check_data_directory)

    async def check_readiness() -> bool:
        await service.on_startup()
        assert service.clickhouse_client is not None
        fetch_scalar = AsyncMock()
        monkeypatch.setattr("nhx.intake.service.ClickHouseExecutor.fetch_scalar", fetch_scalar)
        ready = await service.is_ready()
        fetch_scalar.assert_awaited_once()
        readiness_query = fetch_scalar.await_args.args[0]
        assert readiness_query.name == "intake_readiness"
        assert readiness_query.statement == "SELECT 1 AS ready FROM `intake_unavailable`.`spans` LIMIT 1"
        return ready

    assert asyncio.run(check_readiness()) is True
    assert service.readiness_message == ""
    reconcile.assert_not_awaited()
    stop.assert_not_awaited()
    check_data_directory.assert_not_awaited()


def test_managed_clickhouse_readiness_checks_data_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    reconcile = AsyncMock(return_value="http://127.0.0.1:55123")
    check_data_directory = AsyncMock(side_effect=PermissionError("read-only volume"))
    monkeypatch.setattr("nhx.intake.service.reconcile_local_clickhouse", reconcile)
    monkeypatch.setattr("nhx.intake.service.check_local_clickhouse_data_directory", check_data_directory)
    service = IntakeService().with_config(IntakeConfig(clickhouse_config=ClickHouseConfig()))

    async def check_readiness() -> bool:
        await service.on_startup()
        assert await service.is_ready() is False
        return await service.is_ready()

    assert asyncio.run(check_readiness()) is False
    reconcile.assert_awaited_once()
    assert check_data_directory.await_count == 2
    check_data_directory.assert_awaited_with(data_dir=service.service_config.clickhouse_config.data_dir)
    assert service.readiness_message == CLICKHOUSE_UNAVAILABLE_MESSAGE


def test_successful_probe_does_not_report_ready_after_shutdown_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    service = IntakeService().with_config(_external_config())

    async def overlap_shutdown() -> bool:
        await service.on_startup()
        assert service.clickhouse_client is not None
        probe_started = asyncio.Event()
        release_probe = asyncio.Event()

        async def delayed_fetch(_executor, _query):
            probe_started.set()
            await release_probe.wait()

        monkeypatch.setattr("nhx.intake.service.ClickHouseExecutor.fetch_scalar", delayed_fetch)
        readiness = asyncio.create_task(service.is_ready())
        await probe_started.wait()
        service._ready = False
        release_probe.set()
        return await readiness

    assert asyncio.run(overlap_shutdown()) is False
