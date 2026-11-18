# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``builder-settle``: each pass fails the abandoned ``pending`` images of every workspace."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_builder_plugin.controller import BuildQueueController, SettleController
from nemo_builder_plugin.entities import ContainerImage, Provenance, QueuedBuild
from nemo_builder_plugin.settle import QUEUE_GONE, job_status_from
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.entities import EntityNotFoundError
from nemo_helix_plugin.jobs.schemas import HelixJobStatus


def _row(
    name: str, job: str, *, status: Literal["queued", "pending"] = "pending", age: timedelta = timedelta(minutes=1)
) -> ContainerImage:
    row = ContainerImage(
        name=name,
        workspace="ws-a",
        status=status,
        registry="reg.example.com",
        repository="ws-a/app",
        provenance=Provenance(build_set="demo", revision=1, job=f"ws-a/{job}", request_digest="sha256:" + "a" * 64),
    )
    row._created_at = datetime.now(UTC) - age
    return row


class FakeEntities:
    """Rows and queued builds over pages of ``page_size``; records each list, update and delete in order."""

    def __init__(
        self, rows: Sequence[ContainerImage], page_size: int = 2, queued_builds: Sequence[QueuedBuild] = ()
    ) -> None:
        self.rows = [*rows]
        self.queued_builds = [*queued_builds]
        self.page_size = page_size
        self.events: list[str] = []
        self.filters: list[tuple[str, object]] = []
        #: Names whose delete fails, as an entity store outage would.
        self.undeletable: set[str] = set()

    async def list(self, entity_type: type, *, workspace: str, filter_obj: dict | None, sort: str, page: int):
        self.filters.append((workspace, filter_obj))
        if entity_type is QueuedBuild:
            self.events.append(f"list-queued-builds:{page}")
            matching: list = self.queued_builds
        else:
            status = (filter_obj or {})["status"]
            self.events.append(f"list-{status}:{page}")
            matching = [row for row in self.rows if row.status == status]
        pages = max(1, -(-len(matching) // self.page_size))
        data = matching[(page - 1) * self.page_size : page * self.page_size]
        return SimpleNamespace(data=data, pagination=SimpleNamespace(total_pages=pages))

    async def delete_by_id(self, entity_type: type, entity_id: str, *, expected_db_version: int | None) -> None:
        assert entity_type is QueuedBuild
        name = entity_id.removeprefix("id-")
        self.events.append(f"delete:{name}")
        if name in self.undeletable:
            raise RuntimeError("entity store down")
        if name not in [build.name for build in self.queued_builds]:
            raise EntityNotFoundError(name)
        self.queued_builds = [build for build in self.queued_builds if build.name != name]

    async def get(self, entity_type: type, *, name: str, workspace: str) -> ContainerImage:
        return next(row for row in self.rows if row.name == name)

    async def update(self, entity: ContainerImage) -> ContainerImage:
        self.events.append(f"update:{entity.name}:{entity.status}")
        self.rows = [entity if row.name == entity.name else row for row in self.rows]
        return entity


def _controller(
    entities: FakeEntities, job_status: AsyncMock, controller_type: type[SettleController] = SettleController
) -> SettleController:
    controller = controller_type()
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

        assert entities.events == [
            "list-pending:1",
            "list-pending:2",
            "list-queued:1",
            "update:a:failed",
            "update:b:failed",
            "list-queued-builds:1",
        ]
        assert entities.filters[:3] == [("-", {"status": "pending"})] * 2 + [("-", {"status": "queued"})]
        # Once per job, however many of its images are pending.
        assert [call.args for call in job_status.await_args_list] == [("ws-a", "ended"), ("ws-a", "running")]

    @pytest.mark.asyncio
    async def test_nothing_pending_asks_jobs_nothing(self) -> None:
        entities = FakeEntities([])
        job_status = AsyncMock()
        await _controller(entities, job_status).reconcile()
        assert entities.events == ["list-pending:1", "list-queued:1", "list-queued-builds:1"]
        job_status.assert_not_awaited()


def _queued_build(name: str) -> QueuedBuild:
    build = QueuedBuild(name=name, workspace="ws-a")
    build._id = f"id-{name}"
    return build


class TestWhat010Left:
    """Deprecated: 0.10's builder queued builds itself, and wrote their images ``queued``."""

    @pytest.mark.asyncio
    async def test_a_queued_image_whose_job_was_never_created_fails(self) -> None:
        entities = FakeEntities([_row("a", "never", status="queued", age=timedelta(hours=1))])
        await _controller(entities, AsyncMock(return_value=None)).reconcile()
        [row] = entities.rows
        assert (row.status, row.status_detail) == ("failed", QUEUE_GONE)

    @pytest.mark.asyncio
    async def test_a_queued_image_whose_job_runs_is_left_to_it(self) -> None:
        """0.10's queue created its job, then died before moving the image to ``pending``: the job completes it."""
        entities = FakeEntities([_row("a", "running", status="queued", age=timedelta(hours=1))])
        await _controller(entities, AsyncMock(return_value=HelixJobStatus.ACTIVE)).reconcile()
        assert [row.status for row in entities.rows] == ["queued"]

    @pytest.mark.asyncio
    async def test_its_queued_builds_are_deleted_after_every_page_is_read(self) -> None:
        """Each holds its submitter's principal, and nothing reads it any more."""
        entities = FakeEntities([], queued_builds=[_queued_build(name) for name in "abc"])
        await _controller(entities, AsyncMock()).reconcile()
        assert entities.events[2:] == [
            "list-queued-builds:1",
            "list-queued-builds:2",
            "delete:a",
            "delete:b",
            "delete:c",
        ]
        assert entities.queued_builds == []

    @pytest.mark.asyncio
    async def test_a_queued_build_that_cant_be_deleted_is_left_for_the_next_pass(self) -> None:
        entities = FakeEntities([], queued_builds=[_queued_build("a"), _queued_build("b")])
        entities.undeletable = {"a"}
        await _controller(entities, AsyncMock()).reconcile()
        assert [build.name for build in entities.queued_builds] == ["a"]

    @pytest.mark.asyncio
    async def test_one_already_deleted_is_not_an_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Another pass, under the deprecated name, deleted it first."""
        entities = FakeEntities([])
        controller = _controller(entities, AsyncMock())
        # Listed before the other pass deleted it.
        controller._every = AsyncMock(return_value=[_queued_build("a")])
        await controller._forget_queued_builds()
        assert entities.events == ["delete:a"]
        assert "could not delete" not in caplog.text


class TestTheDeprecatedName:
    def test_builder_queue_runs_builder_settle(self) -> None:
        """So a 0.10 selection that names builder-queue keeps settling."""
        assert BuildQueueController.name == "builder-queue"
        assert issubclass(BuildQueueController, SettleController)

    @pytest.mark.asyncio
    async def test_it_warns_as_it_starts(self, caplog: pytest.LogCaptureFixture) -> None:
        client = MagicMock()
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=client)
        controller = BuildQueueController(client_factory=lambda: context)
        await controller.on_startup()
        assert "builder-queue controller is deprecated" in caplog.text

    @pytest.mark.asyncio
    async def test_it_settles_as_builder_settle_does(self) -> None:
        entities = FakeEntities([_row("a", "ended")])
        await _controller(entities, AsyncMock(return_value=HelixJobStatus.ERROR), BuildQueueController).reconcile()
        assert [row.status for row in entities.rows] == ["failed"]


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
