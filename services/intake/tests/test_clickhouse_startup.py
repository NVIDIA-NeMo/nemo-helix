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

_RECONCILED_URL = "http://127.0.0.1:55123"


def _disable_reconcile_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    import nhx.intake.service as intake_service

    if hasattr(intake_service, "LOCAL_CLICKHOUSE_RECONCILE_RETRY_SECONDS"):
        monkeypatch.setattr(intake_service, "LOCAL_CLICKHOUSE_RECONCILE_RETRY_SECONDS", 0)


_OBSERVATION_TIMEOUT_SECONDS = 5.0


async def _wait_until(ready) -> None:
    async def _poll() -> None:
        while not ready():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout=_OBSERVATION_TIMEOUT_SECONDS)


async def _wait_for_clickhouse_url(service: IntakeService, url: str) -> None:
    def _ready() -> bool:
        client = service.clickhouse_client
        return client is not None and client.settings.url == url

    await _wait_until(_ready)
    client = service.clickhouse_client
    assert client is not None
    assert client.settings.url == url


async def _wait_for_readiness_message(service: IntakeService, message: str) -> None:
    await _wait_until(lambda: service.readiness_message == message)
    assert service.readiness_message == message


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
    _disable_reconcile_backoff(monkeypatch)
    reconcile = AsyncMock(return_value=_RECONCILED_URL)
    stop = AsyncMock(return_value=True)
    monkeypatch.setattr("nhx.intake.service.reconcile_local_clickhouse", reconcile)
    monkeypatch.setattr("nhx.intake.service.stop_local_clickhouse", stop)
    intake_config = IntakeConfig(clickhouse_config=ClickHouseConfig())
    service = IntakeService().with_config(intake_config)

    async def start_and_stop() -> None:
        await service.on_startup()
        try:
            await _wait_for_clickhouse_url(service, _RECONCILED_URL)
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
    _disable_reconcile_backoff(monkeypatch)
    reconcile = AsyncMock(side_effect=provisioning_error)
    monkeypatch.setattr("nhx.intake.service.reconcile_local_clickhouse", reconcile)
    fetch_scalar = AsyncMock()
    monkeypatch.setattr("nhx.intake.service.ClickHouseExecutor.fetch_scalar", fetch_scalar)
    caplog.set_level(log_level, logger="nhx.intake.service")
    service = IntakeService().with_config(IntakeConfig(clickhouse_config=ClickHouseConfig()))

    async def check_readiness() -> bool:
        await service.on_startup()
        await _wait_for_readiness_message(service, str(provisioning_error))
        try:
            ready = await service.is_ready()
            assert service.clickhouse_client is None
            assert service.readiness_message == str(provisioning_error)
            fetch_scalar.assert_not_awaited()
            return ready
        finally:
            await service.on_shutdown()

    assert asyncio.run(check_readiness()) is False
    assert any(expected_log in record.message for record in caplog.records)


def test_local_clickhouse_retry_adopts_ephemeral_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NHX_INTAKE_CLICKHOUSE_URL", raising=False)
    _disable_reconcile_backoff(monkeypatch)
    reconcile = AsyncMock(
        side_effect=[
            DockerUnavailableError("Docker daemon is unavailable"),
            _RECONCILED_URL,
        ]
    )
    stop = AsyncMock(return_value=True)
    fetch_scalar = AsyncMock()
    check_data_directory = AsyncMock()
    monkeypatch.setattr("nhx.intake.service.reconcile_local_clickhouse", reconcile)
    monkeypatch.setattr("nhx.intake.service.stop_local_clickhouse", stop)
    monkeypatch.setattr("nhx.intake.service.check_local_clickhouse_data_directory", check_data_directory)
    monkeypatch.setattr("nhx.intake.service.ClickHouseExecutor.fetch_scalar", fetch_scalar)
    intake_config = IntakeConfig(clickhouse_config=ClickHouseConfig())
    service = IntakeService().with_config(intake_config)

    async def start_probe_and_stop() -> None:
        await service.on_startup()
        try:
            await _wait_for_clickhouse_url(service, _RECONCILED_URL)
            assert await service.is_ready() is True
            assert service.readiness_message == ""
        finally:
            await service.on_shutdown()

    asyncio.run(start_probe_and_stop())
    assert reconcile.await_count == 2
    stop.assert_awaited_once_with(data_dir=intake_config.clickhouse_config.data_dir)


def test_failed_adoption_stops_the_reconciled_container(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NHX_INTAKE_CLICKHOUSE_URL", raising=False)
    reconcile = AsyncMock(return_value=_RECONCILED_URL)
    stop = AsyncMock(return_value=True)
    monkeypatch.setattr("nhx.intake.service.reconcile_local_clickhouse", reconcile)
    monkeypatch.setattr("nhx.intake.service.stop_local_clickhouse", stop)

    def fail_adopt(_self: IntakeService, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("client failed")

    monkeypatch.setattr(IntakeService, "_adopt_reconciled_clickhouse", fail_adopt)
    intake_config = IntakeConfig(clickhouse_config=ClickHouseConfig())
    service = IntakeService().with_config(intake_config)

    async def start_and_stop() -> None:
        await service.on_startup()
        try:
            await _wait_until(lambda: stop.await_count > 0)
        finally:
            await service.on_shutdown()

    asyncio.run(start_and_stop())
    stop.assert_awaited_once_with(data_dir=intake_config.clickhouse_config.data_dir)
    assert service.clickhouse_client is None


def test_unexpected_reconcile_error_retries_and_shutdown_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NHX_INTAKE_CLICKHOUSE_URL", raising=False)
    _disable_reconcile_backoff(monkeypatch)
    failed = asyncio.Event()
    release = asyncio.Event()
    calls = {"n": 0}

    async def reconcile(*_args: object, **_kwargs: object) -> str:
        calls["n"] += 1
        if not failed.is_set():
            failed.set()
            raise RuntimeError("disk full")
        await release.wait()
        return _RECONCILED_URL

    stop = AsyncMock(return_value=True)
    fetch_scalar = AsyncMock()
    check_data_directory = AsyncMock()
    parent_shutdown = AsyncMock()
    monkeypatch.setattr("nhx.intake.service.reconcile_local_clickhouse", reconcile)
    monkeypatch.setattr("nhx.intake.service.stop_local_clickhouse", stop)
    monkeypatch.setattr("nhx.intake.service.check_local_clickhouse_data_directory", check_data_directory)
    monkeypatch.setattr("nhx.intake.service.ClickHouseExecutor.fetch_scalar", fetch_scalar)
    monkeypatch.setattr("nhx.intake.service.Service.on_shutdown", parent_shutdown)
    service = IntakeService().with_config(IntakeConfig(clickhouse_config=ClickHouseConfig()))

    async def failing_stop() -> None:
        raise RuntimeError("worker failed")

    async def start_probe_and_stop() -> None:
        await service.on_startup()
        try:
            await failed.wait()
            await _wait_for_readiness_message(service, "disk full")
            assert service.readiness_message == "disk full"
            release.set()
            await _wait_for_clickhouse_url(service, _RECONCILED_URL)
            assert await service.is_ready() is True
        finally:
            reconciler = service._reconciler
            assert reconciler is not None
            monkeypatch.setattr(reconciler, "stop", failing_stop)
            await service.on_shutdown()

    asyncio.run(start_probe_and_stop())
    assert calls["n"] == 2
    parent_shutdown.assert_awaited_once()


def test_shutdown_during_inflight_reconcile_does_not_adopt_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NHX_INTAKE_CLICKHOUSE_URL", raising=False)
    _disable_reconcile_backoff(monkeypatch)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def reconcile(*_args: object, **_kwargs: object) -> str:
        entered.set()
        await release.wait()
        return _RECONCILED_URL

    stop = AsyncMock(return_value=True)
    monkeypatch.setattr("nhx.intake.service.reconcile_local_clickhouse", reconcile)
    monkeypatch.setattr("nhx.intake.service.stop_local_clickhouse", stop)
    service = IntakeService().with_config(IntakeConfig(clickhouse_config=ClickHouseConfig()))

    async def overlap_shutdown() -> None:
        startup = asyncio.create_task(service.on_startup())
        await entered.wait()
        shutdown = asyncio.create_task(service.on_shutdown())
        await asyncio.sleep(0)
        release.set()
        await startup
        await shutdown

    asyncio.run(overlap_shutdown())
    assert service.clickhouse_client is None
    assert asyncio.run(service.is_ready()) is False
    stop.assert_awaited()


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
    monkeypatch.delenv("NHX_INTAKE_CLICKHOUSE_URL", raising=False)
    _disable_reconcile_backoff(monkeypatch)
    reconcile = AsyncMock(return_value=_RECONCILED_URL)
    check_data_directory = AsyncMock(side_effect=PermissionError("read-only volume"))
    monkeypatch.setattr("nhx.intake.service.reconcile_local_clickhouse", reconcile)
    monkeypatch.setattr("nhx.intake.service.check_local_clickhouse_data_directory", check_data_directory)
    service = IntakeService().with_config(IntakeConfig(clickhouse_config=ClickHouseConfig()))

    async def check_readiness() -> bool:
        await service.on_startup()
        await _wait_for_clickhouse_url(service, _RECONCILED_URL)
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
