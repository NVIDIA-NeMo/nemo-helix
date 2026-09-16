# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from nemo_anonymizer_plugin.tasks.anonymizer import run as task_run
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.job_usage import PlatformJobUsageReporter
from nemo_platform_plugin.jobs.constants import (
    EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
    NEMO_JOB_ID_ENVVAR,
    NEMO_JOB_WORKSPACE_ENVVAR,
    PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
)


def _task_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [handler for handler in logger.handlers if getattr(handler, task_run._TASK_LOG_HANDLER_MARKER, False)]


def test_configure_logging_is_idempotent_and_disables_propagation() -> None:
    loggers = [logging.getLogger(name) for name in ("anonymizer", "data_designer", "nemo_anonymizer_plugin")]
    original_state = {logger.name: (list(logger.handlers), logger.level, logger.propagate) for logger in loggers}

    try:
        for logger in loggers:
            logger.handlers = []
            logger.propagate = True

        task_run._configure_logging()
        task_run._configure_logging()

        for logger in loggers:
            assert len(_task_handlers(logger)) == 1
            assert logger.level == logging.INFO
            assert logger.propagate is False
    finally:
        for logger in loggers:
            handlers, level, propagate = original_state[logger.name]
            logger.handlers = handlers
            logger.setLevel(level)
            logger.propagate = propagate


def test_get_job_name_fails_when_job_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(NEMO_JOB_ID_ENVVAR, raising=False)

    with pytest.raises(RuntimeError, match=NEMO_JOB_ID_ENVVAR):
        task_run._get_job_name()


def test_get_workspace_fails_when_workspace_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(NEMO_JOB_WORKSPACE_ENVVAR, raising=False)

    with pytest.raises(RuntimeError, match=NEMO_JOB_WORKSPACE_ENVVAR):
        task_run._get_workspace()


def test_get_ctx_uses_platform_usage_reporter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NEMO_JOB_ID_ENVVAR, "anonymizer-job")
    monkeypatch.setenv(NEMO_JOB_WORKSPACE_ENVVAR, "team-a")
    monkeypatch.setenv(EPHEMERAL_TASK_STORAGE_PATH_ENVVAR, str(tmp_path / "ephemeral"))
    monkeypatch.setenv(PERSISTENT_JOB_STORAGE_PATH_ENVVAR, str(tmp_path / "persistent"))
    sdk = NeMoPlatform(base_url="http://platform.test", workspace="team-a")

    ctx = task_run._get_ctx(sdk)

    assert isinstance(ctx.usage, PlatformJobUsageReporter)
