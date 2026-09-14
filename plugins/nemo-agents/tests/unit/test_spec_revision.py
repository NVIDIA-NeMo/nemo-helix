# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recording the revision a deployment actually staged."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

from nemo_agents_plugin.spec_revision import SpecRevision, read_spec_revision, stage_with_spec_revision
from nemo_platform_plugin.files.storage_config import GithubStorageConfig

FIRST_SHA = "1" * 40
SECOND_SHA = "2" * 40
THIRD_SHA = "3" * 40


@contextmanager
def _fileset_reporting(*revisions: str) -> Iterator[AsyncMock]:
    """Serve each revision in turn for the agent's fileset, then the last one forever."""
    responses = [
        SimpleNamespace(
            data=lambda revision=revision: SimpleNamespace(
                storage=GithubStorageConfig(owner="acme", repo="agents", revision=revision, original_revision="main")
            )
        )
        for revision in revisions
    ]
    client = AsyncMock()
    client.get_fileset = AsyncMock(side_effect=lambda **_: responses.pop(0) if len(responses) > 1 else responses[0])

    yield client


class TestStageWithSpecRevision:
    async def test_records_the_revision_staging_read(self) -> None:
        stage = AsyncMock(return_value="staged")

        with _fileset_reporting(FIRST_SHA) as files_client:
            staged, spec = await stage_with_spec_revision(
                files_client, workspace="default", agent_name="calc", stage=stage
            )

        assert (staged, spec) == ("staged", SpecRevision(revision=FIRST_SHA, tracked_revision="main"))
        assert stage.await_count == 1

    async def test_restages_when_a_refresh_lands_mid_stage(self) -> None:
        stage = AsyncMock(return_value="staged")

        with _fileset_reporting(FIRST_SHA, SECOND_SHA, SECOND_SHA) as files_client:
            _, spec = await stage_with_spec_revision(files_client, workspace="default", agent_name="calc", stage=stage)

        assert spec.revision == SECOND_SHA
        assert stage.await_count == 2

    async def test_records_nothing_when_it_keeps_moving(self) -> None:
        stage = AsyncMock(return_value="staged")

        with _fileset_reporting(FIRST_SHA, SECOND_SHA, THIRD_SHA) as files_client:
            _, spec = await stage_with_spec_revision(files_client, workspace="default", agent_name="calc", stage=stage)

        assert spec == SpecRevision()
        assert stage.await_count == 2

    async def test_reads_nothing_without_a_files_client(self) -> None:
        assert await read_spec_revision(None, workspace="default", agent_name="calc") == SpecRevision()

    async def test_stages_anyway_without_a_files_client(self) -> None:
        stage = AsyncMock(return_value="staged")

        staged, spec = await stage_with_spec_revision(None, workspace="default", agent_name="calc", stage=stage)

        assert (staged, spec) == ("staged", SpecRevision())
        assert stage.await_count == 1
