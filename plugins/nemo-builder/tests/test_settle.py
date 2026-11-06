# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Settling on read: which ``pending`` images a read fails, and how it writes them."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Literal

import pytest
from nemo_builder_plugin.entities import ContainerImage, Provenance
from nemo_builder_plugin.settle import JOB_GRACE, JOB_READS, NO_JOB, QUEUE_GONE, settle
from nemo_helix_plugin.entities import EntityConflictError
from nemo_helix_plugin.jobs.schemas import HelixJobStatus

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
DIGEST = "sha256:" + "d" * 64


def _row(
    name: str = "demo-1.img0",
    *,
    status: Literal["queued", "pending", "ready", "failed"] = "pending",
    job: str = "ws-a/builder-demo-1",
    age: timedelta | None = timedelta(minutes=1),
) -> ContainerImage:
    row = ContainerImage(
        name=name,
        workspace="ws-a",
        status=status,
        registry="reg.example.com",
        repository="ws-a/app",
        provenance=Provenance(build_set="demo", revision=1, job=job, request_digest="sha256:" + "a" * 64),
    )
    row._created_at = None if age is None else NOW - age
    return row


class FakeRows:
    def __init__(self, *rows: ContainerImage) -> None:
        self.rows = {row.name: row for row in rows}
        self.updates: list[ContainerImage] = []
        #: Rows whose update conflicts, as one changed since it was read would.
        self.changed: dict[str, ContainerImage] = {}
        self.broken = False

    async def get(self, entity_type: type[ContainerImage], *, name: str, workspace: str) -> ContainerImage:
        return self.rows[name]

    async def update(self, entity: ContainerImage) -> ContainerImage:
        if self.broken:
            raise RuntimeError("entity store down")
        if entity.name in self.changed:
            self.rows[entity.name] = self.changed[entity.name]
            raise EntityConflictError(f"{entity.name} changed")
        self.updates.append(entity)
        self.rows[entity.name] = entity
        return entity


class FakeJobs:
    def __init__(self, statuses: dict[str, HelixJobStatus | Exception]) -> None:
        self.statuses = statuses
        self.asked: list[str] = []

    async def __call__(self, workspace: str, name: str) -> HelixJobStatus | None:
        self.asked.append(f"{workspace}/{name}")
        found = self.statuses.get(name)
        if isinstance(found, Exception):
            raise found
        return found


async def _settle(rows: FakeRows, jobs: FakeJobs, *images: ContainerImage) -> list[ContainerImage]:
    return await settle(rows, list(images or rows.rows.values()), jobs, now=NOW)


class TestWhatAReadFails:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", HelixJobStatus.terminals())
    async def test_a_pending_image_whose_job_ended_fails(self, status: HelixJobStatus) -> None:
        rows = FakeRows(_row())
        [image] = await _settle(rows, FakeJobs({"builder-demo-1": status}))
        assert image.status == "failed"
        assert image.status_detail == f"not built: its build job ended ({status.value}) without publishing it"
        assert rows.rows["demo-1.img0"].status == "failed"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", HelixJobStatus.non_terminals())
    async def test_a_pending_image_whose_job_may_still_complete_it_is_left(self, status: HelixJobStatus) -> None:
        rows = FakeRows(_row())
        [image] = await _settle(rows, FakeJobs({"builder-demo-1": status}))
        assert image.status == "pending"
        assert rows.updates == []

    @pytest.mark.asyncio
    async def test_a_pending_image_with_no_job_fails_once_the_grace_has_passed(self) -> None:
        rows = FakeRows(_row(age=JOB_GRACE))
        [image] = await _settle(rows, FakeJobs({}))
        assert (image.status, image.status_detail) == ("failed", NO_JOB)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("age", [JOB_GRACE - timedelta(seconds=1), None], ids=["within-the-grace", "undated"])
    async def test_a_pending_image_with_no_job_is_left_until_then(self, age: timedelta | None) -> None:
        """Its submit writes the row, then creates the job; an undated row is never taken for old."""
        rows = FakeRows(_row(age=age))
        [image] = await _settle(rows, FakeJobs({}))
        assert image.status == "pending"
        assert rows.updates == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["ready", "failed"])
    async def test_a_settled_image_is_returned_without_asking_jobs(self, status: Literal["ready", "failed"]) -> None:
        jobs = FakeJobs({"builder-demo-1": HelixJobStatus.ERROR})
        rows = FakeRows(_row(status=status))
        [image] = await _settle(rows, jobs)
        assert image.status == status
        assert jobs.asked == []


class TestHowAReadWrites:
    @pytest.mark.asyncio
    async def test_jobs_are_read_several_at_a_time_but_no_more_than_job_reads(self) -> None:
        """One at a time, a pass over about 100 pending images outran the controller's 90-second health window."""
        reading = most = 0
        asked: list[str] = []

        async def slow_status(workspace: str, name: str) -> HelixJobStatus | None:
            nonlocal reading, most
            reading += 1
            most = max(most, reading)
            asked.append(name)
            await asyncio.sleep(0.01)
            reading -= 1
            return HelixJobStatus.ACTIVE

        names = [f"builder-demo-{i}" for i in range(3 * JOB_READS)]
        rows = FakeRows(*(_row(f"img-{i}", job=f"ws-a/{name}") for i, name in enumerate(names)))
        images = await settle(rows, list(rows.rows.values()), slow_status, now=NOW)
        assert most == JOB_READS
        assert sorted(asked) == sorted(names)
        assert [image.status for image in images] == ["pending"] * len(names)

    @pytest.mark.asyncio
    async def test_every_job_is_read_before_any_image_is_written(self) -> None:
        """A write before the last read would only make the pass longer; the reads are what run at once."""
        events: list[str] = []
        jobs = FakeJobs({"builder-demo-1": HelixJobStatus.ERROR, "builder-demo-2": HelixJobStatus.ERROR})

        async def recorded(workspace: str, name: str) -> HelixJobStatus | None:
            events.append(f"read:{name}")
            return await jobs(workspace, name)

        rows = FakeRows(_row("a"), _row("b", job="ws-a/builder-demo-2"))
        update = rows.update

        async def recorded_update(entity: ContainerImage) -> ContainerImage:
            events.append(f"write:{entity.name}")
            return await update(entity)

        rows.update = recorded_update  # ty: ignore[invalid-assignment]
        await settle(rows, list(rows.rows.values()), recorded, now=NOW)
        assert events == ["read:builder-demo-1", "read:builder-demo-2", "write:a", "write:b"]

    @pytest.mark.asyncio
    async def test_each_job_is_asked_about_once(self) -> None:
        jobs = FakeJobs({"builder-demo-1": HelixJobStatus.ERROR, "builder-demo-2": HelixJobStatus.ACTIVE})
        rows = FakeRows(_row("a"), _row("b"), _row("c", job="ws-a/builder-demo-2"))
        images = await _settle(rows, jobs)
        assert [image.status for image in images] == ["failed", "failed", "pending"]
        assert jobs.asked == ["ws-a/builder-demo-1", "ws-a/builder-demo-2"]

    @pytest.mark.asyncio
    async def test_the_write_is_conditional_on_the_version_read(self) -> None:
        row = _row()
        row._db_version = 7
        rows = FakeRows(row)
        await _settle(rows, FakeJobs({"builder-demo-1": HelixJobStatus.COMPLETED}))
        assert [update.db_version for update in rows.updates] == [7]
        # The caller's copy is not the one written.
        assert row.status == "pending"

    @pytest.mark.asyncio
    async def test_an_image_completed_meanwhile_is_read_again_not_failed(self) -> None:
        """Read pending, then its push step completes it and the job ends: the write conflicts."""
        ready = _row(status="ready")
        ready.digest = DIGEST
        rows = FakeRows(_row())
        rows.changed["demo-1.img0"] = ready
        [image] = await _settle(rows, FakeJobs({"builder-demo-1": HelixJobStatus.COMPLETED}))
        assert (image.status, image.digest) == ("ready", DIGEST)
        assert rows.updates == []

    @pytest.mark.asyncio
    async def test_a_job_that_cant_be_read_leaves_its_images_and_is_asked_once(self) -> None:
        jobs = FakeJobs({"builder-demo-1": RuntimeError("jobs down")})
        rows = FakeRows(_row("a"), _row("b"))
        images = await _settle(rows, jobs)
        assert [image.status for image in images] == ["pending", "pending"]
        assert jobs.asked == ["ws-a/builder-demo-1"]

    @pytest.mark.asyncio
    async def test_an_image_that_cant_be_written_is_returned_pending(self) -> None:
        rows = FakeRows(_row())
        rows.broken = True
        [image] = await _settle(rows, FakeJobs({"builder-demo-1": HelixJobStatus.ERROR}))
        assert image.status == "pending"


class TestWhat010Queued:
    """Deprecated: 0.10's builder wrote an image ``queued`` while its own queue held the build."""

    @pytest.mark.asyncio
    async def test_one_with_no_job_fails_once_the_grace_has_passed(self) -> None:
        rows = FakeRows(_row(status="queued", age=JOB_GRACE))
        [image] = await _settle(rows, FakeJobs({}))
        assert (image.status, image.status_detail) == ("failed", QUEUE_GONE)

    @pytest.mark.asyncio
    async def test_one_with_no_job_within_the_grace_is_left(self) -> None:
        """A 0.10 queue controller still running mid-upgrade may yet create its job."""
        rows = FakeRows(_row(status="queued", age=JOB_GRACE - timedelta(seconds=1)))
        [image] = await _settle(rows, FakeJobs({}))
        assert image.status == "queued"
        assert rows.updates == []

    @pytest.mark.asyncio
    async def test_one_whose_job_ended_fails_as_a_pending_one_does(self) -> None:
        rows = FakeRows(_row(status="queued"))
        [image] = await _settle(rows, FakeJobs({"builder-demo-1": HelixJobStatus.ERROR}))
        assert image.status == "failed"
        assert "ended (error)" in (image.status_detail or "")

    @pytest.mark.asyncio
    async def test_one_whose_job_runs_is_left_to_it(self) -> None:
        rows = FakeRows(_row(status="queued"))
        [image] = await _settle(rows, FakeJobs({"builder-demo-1": HelixJobStatus.ACTIVE}))
        assert image.status == "queued"
        assert rows.updates == []
