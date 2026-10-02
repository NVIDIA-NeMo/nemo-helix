# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the auditor plugin seed job."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from nemo_auditor.entities import AuditConfig
from nemo_auditor.seed_job import DEFAULT_CONFIG_NAME, SYSTEM_WORKSPACE, AuditorSeedJob
from nemo_helix_plugin.entity_client import NemoEntityConflictError


@pytest.mark.asyncio
async def test_seed_job_creates_default_config() -> None:
    job = AuditorSeedJob()
    job.entities_client = AsyncMock()

    await job.run()

    job.entities_client.create.assert_awaited_once()
    config = job.entities_client.create.await_args.args[0]
    assert isinstance(config, AuditConfig)
    assert config.name == DEFAULT_CONFIG_NAME
    assert config.workspace == SYSTEM_WORKSPACE
    assert config.system.lite is False
    assert config.system.parallel_attempts == 32
    assert config.run.generations == 3


@pytest.mark.asyncio
async def test_seed_job_is_idempotent_on_conflict() -> None:
    job = AuditorSeedJob()
    job.entities_client = AsyncMock()
    job.entities_client.create.side_effect = NemoEntityConflictError("exists")

    await job.run()

    job.entities_client.create.assert_awaited_once()
