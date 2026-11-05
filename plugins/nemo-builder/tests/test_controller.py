# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``builder-settle``: each pass fails the abandoned ``pending`` images of every workspace."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_builder_plugin.controller import SettleController
from nemo_builder_plugin.entities import ContainerImage, Provenance
from nemo_builder_plugin.settle import job_status_from
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.jobs.schemas import HelixJobStatus


def _row(name: str, job: str) -> ContainerImage:
    row = ContainerImage(
        name=name,
        workspace="ws-a",
        registry="reg.example.com",
        repository="ws-a/app",
        provenance=Provenance(build_set="demo", revision=1, job=f"ws-a/{job}", request_digest="sha256:" + "a" * 64),
    )
    row._created_at = datetime.now(UTC) - timedelta(minutes=1)
    return row


class FakeEntities:
    """``pending`` rows over pages of ``page_size``; records each list and update in order."""

    def __init__(self, rows: Sequence[ContainerImage], page_size: int = 2) -> None:
        self.rows = [*rows]
        self.page_size = page_size
        self.events: list[str] = []
        self.filters: list[tuple[str, object]] = []

    async def list(self, entity_type: type, *, workspace: str, filter_obj: dict, sort: str, page: int):
        self.events.append(f"list:{page}")
        self.filters.append((workspace, filter_obj))
        pending = [row for row in self.rows if row.status == "pending"]
        pages = max(1, -(-len(pending) // self.page_size))
        data = pending[(page - 1) * self.page_size : page * self.page_size]
        return SimpleNamespace(data=data, pagination=SimpleNamespace(total_pages=pages))

    async def get(self, entity_type: type, *, name: str, workspace: str) -> ContainerImage:
        return next(row for row in self.rows if row.name == name)

    async def update(self, entity: ContainerImage) -> ContainerImage:
        self.events.append(f"update:{entity.name}:{entity.status}")
        self.rows = [entity if row.name == entity.name else row for row in self.rows]
        return entity


def _controller(entities: FakeEntities, job_status: AsyncMock) -> SettleController:
    controller = SettleController()
    controller._entities = entities  # ty: ignore[invalid-assignment]
    controller._job_status = job_status
    return controller


class TestAPass:
    @pytest.mark.asyncio
    async def test_it_reads_every_page_before_failing_the_abandoned_ones(self) -> None:
        """A write before the last page is read would move a row out of the pages still to come."""
        entities = FakeEntities([_row("a", "ended"), _row("b", "ended"), _row("c", "running")])
        statuses = {"ended": HelixJobStatus.ERROR, "running": HelixJobStatus.ACTIVE}
        job_status = AsyncMock(side_effect=lambda workspace, name: statuses[name])

        await _controller(entities, job_status).reconcile()

        assert entities.events == ["list:1", "list:2", "update:a:failed", "update:b:failed"]
        assert entities.filters == [("-", {"status": "pending"})] * 2
        # Once per job, however many of its images are pending.
        assert [call.args for call in job_status.await_args_list] == [("ws-a", "ended"), ("ws-a", "running")]

    @pytest.mark.asyncio
    async def test_nothing_pending_asks_jobs_nothing(self) -> None:
        entities = FakeEntities([])
        job_status = AsyncMock()
        await _controller(entities, job_status).reconcile()
        assert entities.events == ["list:1"]
        job_status.assert_not_awaited()


class TestReadingAJobAsTheBuilder:
    @pytest.mark.asyncio
    async def test_a_job_that_doesnt_exist_is_none(self) -> None:
        jobs = MagicMock()
        jobs.get_job_status = AsyncMock(
            side_effect=NotFoundError(
                httpx.Response(404, json={"detail": "no"}, request=httpx.Request("GET", "http://j"))
            )
        )
        assert await job_status_from(jobs)("ws-a", "builder-demo-1") is None

    @pytest.mark.asyncio
    async def test_a_job_that_exists_is_its_status(self) -> None:
        response = MagicMock()
        response.data.return_value.status = HelixJobStatus.QUEUED
        jobs = MagicMock()
        jobs.get_job_status = AsyncMock(return_value=response)
        assert await job_status_from(jobs)("ws-a", "builder-demo-1") == HelixJobStatus.QUEUED
        jobs.get_job_status.assert_awaited_once_with(workspace="ws-a", name="builder-demo-1")
