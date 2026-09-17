# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NMPJobContext."""

from pathlib import Path

import pytest
from nmp.automodel.app.constants import DEFAULT_JOB_STORAGE_PATH, NMP_FILES_URL_ENVVAR, NMP_JOBS_URL_ENVVAR
from nmp.common.entities.constants import DEFAULT_WORKSPACE
from nmp.common.jobs.constants import (
    DEFAULT_NEMO_JOB_STEP_CONFIG_FILE_PATH,
    NEMO_JOB_ATTEMPT_ID_ENVVAR,
    NEMO_JOB_ID_ENVVAR,
    NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
    NEMO_JOB_STEP_ENVVAR,
    NEMO_JOB_TASK_ENVVAR,
    NEMO_JOB_WORKSPACE_ENVVAR,
    PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
)
from nmp.customization_common.service.context import (
    DEFAULT_ATTEMPT_ID,
    DEFAULT_JOB_ID,
    DEFAULT_STEP,
    DEFAULT_TASK,
    NMPJobContext,
)
from nmp.customization_common.service.path_utils import (
    CURRENT_NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
    CURRENT_PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
    LEGACY_NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVARS,
    LEGACY_PERSISTENT_JOB_STORAGE_PATH_ENVVARS,
)

PATH_ENVVARS = tuple(
    dict.fromkeys(
        (
            CURRENT_PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
            PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
            *LEGACY_PERSISTENT_JOB_STORAGE_PATH_ENVVARS,
            CURRENT_NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
            NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR,
            *LEGACY_NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVARS,
        )
    )
)


def _clear_path_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in PATH_ENVVARS:
        monkeypatch.delenv(var, raising=False)


class TestNMPJobContextFromEnv:
    def test_uses_defaults_when_env_vars_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_path_env(monkeypatch)
        for var in (
            NEMO_JOB_WORKSPACE_ENVVAR,
            NEMO_JOB_ID_ENVVAR,
            NEMO_JOB_ATTEMPT_ID_ENVVAR,
            NEMO_JOB_STEP_ENVVAR,
            NEMO_JOB_TASK_ENVVAR,
            NMP_JOBS_URL_ENVVAR,
            NMP_FILES_URL_ENVVAR,
        ):
            monkeypatch.delenv(var, raising=False)

        ctx = NMPJobContext.from_env()

        assert ctx.workspace == DEFAULT_WORKSPACE
        assert ctx.job_id == DEFAULT_JOB_ID
        assert ctx.attempt_id == DEFAULT_ATTEMPT_ID
        assert ctx.step == DEFAULT_STEP
        assert ctx.task == DEFAULT_TASK
        assert ctx.jobs_url is None
        assert ctx.files_url is None
        assert ctx.storage_path == Path(DEFAULT_JOB_STORAGE_PATH)
        assert ctx.config_path == Path(DEFAULT_NEMO_JOB_STEP_CONFIG_FILE_PATH)

    def test_uses_env_vars_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_path_env(monkeypatch)
        monkeypatch.setenv(NEMO_JOB_WORKSPACE_ENVVAR, "test-workspace")
        monkeypatch.setenv(NEMO_JOB_ID_ENVVAR, "job-123")
        monkeypatch.setenv(NEMO_JOB_ATTEMPT_ID_ENVVAR, "attempt-5")
        monkeypatch.setenv(NEMO_JOB_STEP_ENVVAR, "training")
        monkeypatch.setenv(NEMO_JOB_TASK_ENVVAR, "train-model")
        monkeypatch.setenv(NMP_JOBS_URL_ENVVAR, "http://jobs.example.com")
        monkeypatch.setenv(NMP_FILES_URL_ENVVAR, "http://files.example.com")
        monkeypatch.setenv(PERSISTENT_JOB_STORAGE_PATH_ENVVAR, "/custom/storage")
        monkeypatch.setenv(NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVAR, "/custom/config.json")

        ctx = NMPJobContext.from_env()

        assert ctx.workspace == "test-workspace"
        assert ctx.job_id == "job-123"
        assert ctx.normalized_task == "task-train-model"
        assert ctx.jobs_url == "http://jobs.example.com"
        assert ctx.files_url == "http://files.example.com"
        assert ctx.storage_path == Path("/custom/storage")
        assert ctx.config_path == Path("/custom/config.json")

    def test_current_storage_env_wins_over_legacy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_path_env(monkeypatch)
        monkeypatch.setenv(CURRENT_PERSISTENT_JOB_STORAGE_PATH_ENVVAR, "/var/run/scratch/job")
        monkeypatch.setenv(LEGACY_PERSISTENT_JOB_STORAGE_PATH_ENVVARS[0], "/run/scratch/job")

        ctx = NMPJobContext.from_env()

        assert ctx.storage_path == Path("/var/run/scratch/job")

    def test_uses_legacy_storage_env_when_current_env_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_path_env(monkeypatch)
        monkeypatch.setenv(LEGACY_PERSISTENT_JOB_STORAGE_PATH_ENVVARS[0], "/run/scratch/job")

        ctx = NMPJobContext.from_env()

        assert ctx.storage_path == Path("/run/scratch/job")

    def test_uses_legacy_config_env_when_current_env_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_path_env(monkeypatch)
        monkeypatch.setenv(LEGACY_NEMO_JOB_STEP_CONFIG_FILE_PATH_ENVVARS[0], "/run/scratch/config/job.json")

        ctx = NMPJobContext.from_env()

        assert ctx.config_path == Path("/run/scratch/config/job.json")
