# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for platform runner startup display helpers."""

import logging
import threading

import pytest
from nhx.common.config import Configuration, get_platform_config
from nhx.platform_runner import run as runner
from nhx.platform_runner.config import ResolvedRunConfiguration
from nhx.platform_runner.run import _database_display


class _StubService:
    def __init__(self, name: str) -> None:
        self.name = name


@pytest.mark.parametrize(
    ("db_url", "expected"),
    [
        (
            "sqlite:////var/data/nhx-platform.db",
            "sqlite (/var/data/nhx-platform.db)",
        ),
        (
            "sqlite+aiosqlite:////var/data/nhx-platform.db",
            "sqlite (/var/data/nhx-platform.db)",
        ),
        ("sqlite:///relative/nhx-platform.db", "sqlite (relative/nhx-platform.db)"),
        ("sqlite+aiosqlite:///relative/nhx-platform.db", "sqlite (relative/nhx-platform.db)"),
    ],
)
def test_database_display_formats_sqlite_paths(db_url: str, expected: str) -> None:
    assert _database_display(db_url) == expected


def test_database_display_formats_non_sqlite_driver() -> None:
    assert _database_display("postgresql+asyncpg://user:pass@localhost:5432/nemo") == "postgresql"


def test_database_display_logs_parse_failures(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="nhx.platform_runner.run"):
        assert _database_display("postgresql://localhost:not-a-port/nemo") == "postgresql"

    records = [record for record in caplog.records if record.name == "nhx.platform_runner.run"]
    assert len(records) == 1
    assert records[0].message == "Failed to parse database URL for startup banner"
    assert records[0].exc_info is not None


def test_run_platform_marks_loaded_services_local_before_starting_controllers(monkeypatch):
    Configuration.clear_cache()
    captured: dict[str, str] = {}

    resolved = ResolvedRunConfiguration(
        services={"jobs", "entities"},
        controllers={"jobs"},
        sidecars=set(),
        host="127.0.0.1",
        port=8080,
        config_path="",
        keep_alive_timeout_seconds=12,
    )
    services = [_StubService("jobs"), _StubService("entities")]

    monkeypatch.setattr(runner, "resolve_run_configuration", lambda *_args, **_kwargs: resolved)
    monkeypatch.setattr(runner, "apply_run_environment", lambda config: None)
    monkeypatch.setattr(runner, "initialize_obs", lambda *, resource_attributes: None)
    monkeypatch.setattr(runner, "setup_global_instrumentations", lambda: None)
    monkeypatch.setattr(runner, "_load_service_instances", lambda service_names, available_services: services)
    monkeypatch.setattr(
        runner,
        "_load_run_functions",
        lambda names, registry, kind: {"jobs": lambda stop_signal: None} if kind == "controller" else {},
    )
    monkeypatch.setattr(runner, "_display_banner", lambda **_: None)
    monkeypatch.setattr(
        runner,
        "run_server",
        lambda services, host, port, socket_path=None, keep_alive_timeout_seconds=None: captured.update(
            {"keep_alive_timeout_seconds": str(keep_alive_timeout_seconds)}
        ),
    )
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)

    def capture_controller_start(
        controller_run_funcs,
        stop_signal: threading.Event,
    ) -> list[threading.Thread]:
        captured["services"] = get_platform_config().services
        return []

    monkeypatch.setattr(runner, "run_controllers_in_threads", capture_controller_start)

    try:
        runner.run_platform()
    finally:
        Configuration.clear_cache()

    assert captured["services"] == "entities,jobs"
    assert captured["keep_alive_timeout_seconds"] == "12"
