# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Root-logging bootstrap for platform-spawned task processes."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from nemo_helix_plugin.config import NHX_CONFIG_FILE_PATH_ENV_VAR, Configuration
from nemo_helix_plugin.tasks import logging_setup
from nemo_helix_plugin.tasks.logging_setup import (
    FALLBACK_LOG_FORMAT,
    DefaultTaskLoggingProvider,
    configure_task_logging,
    set_task_logging_provider,
)


@pytest.fixture(autouse=True)
def _restore_global_state() -> Iterator[None]:
    """Undo the global logging and provider state these tests mutate by design."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    try:
        yield
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)
        set_task_logging_provider(None)


def _strip_root_handlers() -> None:
    """Reproduce a task container's bare root logger.

    Has to run inside the test body rather than a fixture: pytest's own log
    capture attaches a handler to the root logger for each test phase, so a
    fixture that cleared it would be undone before the test ran.
    """
    logging.getLogger().handlers = []


class _RecordingProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def configure_logging(self, *, level: str, log_format: str) -> None:
        self.calls.append({"level": level, "log_format": log_format})
        logging.getLogger().addHandler(logging.NullHandler())


class _ExplodingProvider:
    def configure_logging(self, *, level: str, log_format: str) -> None:
        raise RuntimeError("provider is broken")


class _PartiallyApplyingProvider:
    """Attaches a handler, then fails before setting a usable level.

    The shape that makes a naive "did it leave any handler?" check wrong: the
    root logger looks configured but drops every record the task emits.
    """

    def configure_logging(self, *, level: str, log_format: str) -> None:
        logging.getLogger().addHandler(logging.NullHandler())
        raise RuntimeError("failed after attaching a handler")


def test_configure_task_logging_makes_info_records_reachable() -> None:
    """The regression this exists for: INFO must survive in a task container.

    With no root handler Python falls back to ``logging.lastResort``, which
    drops everything below WARNING - which is how a failing job produced
    completely empty logs.
    """
    _strip_root_handlers()

    configure_task_logging()

    root = logging.getLogger()
    assert root.handlers, "no handler installed; INFO records would still be dropped"
    assert logging.getLogger("some.task.module").isEnabledFor(logging.INFO)


def test_configure_task_logging_leaves_an_existing_configuration_alone() -> None:
    """A caller that configured logging keeps ownership of it."""
    _strip_root_handlers()
    existing = logging.StreamHandler()
    logging.getLogger().addHandler(existing)

    configure_task_logging()

    assert logging.getLogger().handlers == [existing]


def test_configure_task_logging_is_idempotent() -> None:
    """Calling twice must not double every log line."""
    _strip_root_handlers()
    configure_task_logging()
    handlers_after_first = logging.getLogger().handlers[:]

    configure_task_logging()

    assert logging.getLogger().handlers == handlers_after_first


def test_a_registered_provider_is_used_and_given_the_configured_settings() -> None:
    """The seam exists so the platform can supply structured logging."""
    provider = _RecordingProvider()
    set_task_logging_provider(provider)
    _strip_root_handlers()

    configure_task_logging()

    assert provider.calls == [{"level": "INFO", "log_format": "plain"}]


def test_a_failing_provider_falls_back_to_the_default(caplog: pytest.LogCaptureFixture) -> None:
    """Logging is diagnostic scaffolding: a broken provider must not take the
    task down with it, or it destroys the evidence needed to debug it."""
    set_task_logging_provider(_ExplodingProvider())
    _strip_root_handlers()

    configure_task_logging()

    handlers = logging.getLogger().handlers
    assert handlers, "the task was left silent"
    assert any(getattr(h.formatter, "_fmt", None) == FALLBACK_LOG_FORMAT for h in handlers)


def test_default_provider_is_used_when_nothing_is_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    """``nhx_common`` is optional, so its absence must not leave a silent task."""
    monkeypatch.setattr(logging_setup, "discover_entry_points", lambda group: {})
    _strip_root_handlers()

    configure_task_logging()

    handlers = logging.getLogger().handlers
    assert len(handlers) == 1
    assert handlers[0].formatter is not None
    assert handlers[0].formatter._fmt == FALLBACK_LOG_FORMAT


def test_competing_providers_fall_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Two packages registering their own provider is not something to guess at."""
    monkeypatch.setattr(
        logging_setup,
        "discover_entry_points",
        lambda group: {"platform": object(), "someone-else": object()},
    )
    _strip_root_handlers()

    with caplog.at_level(logging.WARNING, logger="nemo_helix_plugin.tasks.logging_setup"):
        configure_task_logging()

    handlers = logging.getLogger().handlers
    assert any(getattr(h.formatter, "_fmt", None) == FALLBACK_LOG_FORMAT for h in handlers)


def test_a_provider_that_does_not_implement_the_protocol_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong target must degrade, not raise mid-bootstrap."""

    class _Entry:
        name = "platform"
        value = "some.module:NotAProvider"

        def load(self) -> Any:
            return object()

    monkeypatch.setattr(logging_setup, "discover_entry_points", lambda group: {"platform": _Entry()})
    _strip_root_handlers()

    configure_task_logging()

    handlers = logging.getLogger().handlers
    assert any(getattr(h.formatter, "_fmt", None) == FALLBACK_LOG_FORMAT for h in handlers)


def test_default_provider_emits_the_fields_last_resort_drops(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Level and logger name are what make a task log readable at all."""
    _strip_root_handlers()

    DefaultTaskLoggingProvider().configure_logging(level="INFO", log_format="plain")
    logging.getLogger("nemo_agents_plugin.jobs.execute").info("agent finished")

    stderr = capsys.readouterr().err
    assert "INFO" in stderr
    assert "nemo_agents_plugin.jobs.execute" in stderr
    assert "agent finished" in stderr


def test_default_provider_tolerates_an_unparseable_level() -> None:
    """A bad LOG_LEVEL should degrade to INFO, not crash the task before it runs."""
    _strip_root_handlers()

    # Deliberately off-contract: the guard exists for a misconfigured deployment.
    DefaultTaskLoggingProvider().configure_logging(level=cast(Any, "NOT_A_LEVEL"), log_format="plain")

    assert logging.getLogger().level == logging.INFO


def test_resolve_log_config_falls_back_when_settings_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Misconfigured settings must not be the reason a task cannot log."""
    import nemo_helix_plugin.config as plugin_config

    def _explode() -> None:
        raise RuntimeError("bad settings")

    monkeypatch.setattr(plugin_config, "CommonServiceConfig", _explode)

    assert logging_setup._resolve_log_config() == ("INFO", "plain")


def test_a_partially_applied_provider_is_discarded_not_trusted(caplog: pytest.LogCaptureFixture) -> None:
    """A handler left behind by a failed call is not a working configuration."""
    set_task_logging_provider(_PartiallyApplyingProvider())
    _strip_root_handlers()

    configure_task_logging()

    root = logging.getLogger()
    assert not any(isinstance(handler, logging.NullHandler) for handler in root.handlers), (
        "the failed provider's handler survived"
    )
    assert any(getattr(h.formatter, "_fmt", None) == FALLBACK_LOG_FORMAT for h in root.handlers)
    # The point of the cleanup: task INFO records must reach the fallback.
    assert logging.getLogger("some.task.module").isEnabledFor(logging.INFO)


def test_provider_failure_is_reported_through_the_fallback(capsys: pytest.CaptureFixture[str]) -> None:
    """Reporting the failure before the fallback exists would emit it into the
    broken configuration being replaced."""
    set_task_logging_provider(_PartiallyApplyingProvider())
    _strip_root_handlers()

    configure_task_logging()

    stderr = capsys.readouterr().err
    assert "Task logging provider failed" in stderr
    assert "failed after attaching a handler" in stderr


def test_resolve_log_config_reads_the_yaml_backed_platform_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployment configures logging in the platform config file, not the
    environment. Instantiating the settings class directly would read only the
    environment and silently leave such a deployment on the defaults."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("service:\n  LOG_FORMAT: json\n  LOG_LEVEL: DEBUG\n")
    monkeypatch.setenv(NHX_CONFIG_FILE_PATH_ENV_VAR, str(config_file))
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    Configuration.clear_cache()

    try:
        assert logging_setup._resolve_log_config() == ("DEBUG", "json")
    finally:
        Configuration.clear_cache()


def test_environment_still_wins_over_the_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Containers with no config file get their settings from the environment
    the jobs backends inject, so that source has to take precedence."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("service:\n  LOG_FORMAT: plain\n  LOG_LEVEL: ERROR\n")
    monkeypatch.setenv(NHX_CONFIG_FILE_PATH_ENV_VAR, str(config_file))
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    Configuration.clear_cache()

    try:
        assert logging_setup._resolve_log_config() == ("DEBUG", "json")
    finally:
        Configuration.clear_cache()


def test_resolve_log_config_falls_back_when_the_config_file_is_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed config file must not be the reason a task cannot log."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("this: is: not: valid: yaml:\n")
    monkeypatch.setenv(NHX_CONFIG_FILE_PATH_ENV_VAR, str(config_file))
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    Configuration.clear_cache()

    try:
        assert logging_setup._resolve_log_config() == ("INFO", "plain")
    finally:
        Configuration.clear_cache()
