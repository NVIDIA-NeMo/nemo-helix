# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the example plugin seed job."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from nemo_example_plugin.entities import ExampleItem
from nemo_example_plugin.seed_job import ExampleSeedJob
from nemo_helix_plugin.entity_client import NemoEntityConflictError


@pytest.mark.asyncio
async def test_seed_job_creates_default_item() -> None:
    job = ExampleSeedJob()
    job.entities_client = AsyncMock()

    await job.run()

    job.entities_client.create.assert_awaited_once()
    item = job.entities_client.create.await_args.args[0]
    assert isinstance(item, ExampleItem)
    assert item.name == "seeded-example-item"
    assert item.workspace == "system"


@pytest.mark.asyncio
async def test_seed_job_is_idempotent_on_conflict() -> None:
    job = ExampleSeedJob()
    job.entities_client = AsyncMock()
    job.entities_client.create.side_effect = NemoEntityConflictError("exists")

    await job.run()

    job.entities_client.create.assert_awaited_once()
