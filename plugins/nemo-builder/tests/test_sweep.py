# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The failure sweep: when a row can no longer complete, and nothing else.

Every decision rests on one fact from Jobs -- has the row's job ended -- and on the store's
conditional write, which is what keeps the sweep from ever undoing a `ready` that a signature wrote.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from nemo_builder_plugin.controller import BuilderController
from nemo_builder_plugin.entities import ContainerImage, Provenance
from nemo_helix_plugin.client.errors import InternalServerError, NotFoundError
from nemo_helix_plugin.client.response import NemoResponse
from nemo_helix_plugin.client.types import PreparedRequest
from nemo_helix_plugin.entities import EntityConflictError, ListResponse, PaginationInfo
from nemo_helix_plugin.jobs.schemas import HelixJobStatus, HelixJobStatusResponse


def _row(
    name: str = "demo-1-0", *, age: timedelta = timedelta(hours=1), job: str = "ws/demo-1", naive: bool = False
) -> ContainerImage:
    row = ContainerImage(
        name=name,
        workspace="ws",
        registry="reg.example.com",
        repository="ws/team/app",
        provenance=Provenance(
            build_set="demo", revision=1, job=job, system_tag="ws--demo-1-0", request_digest="sha256:" + "a" * 64
        ),
    )
    # What the entity store sets on write, and the only way to have it here. The store keeps UTC
    # without a timezone, so rows read back from it are naive.
    created = datetime.now(UTC) - age
    row._created_at = created.replace(tzinfo=None) if naive else created
    return row


def _response(status: HelixJobStatus) -> NemoResponse[HelixJobStatusResponse]:
    now = datetime.now(UTC)
    body = HelixJobStatusResponse(
        id="job-id",
        name="demo-1",
        status=status,
        status_details={},
        error_details=None,
        steps=[],
        created_at=now,
        updated_at=now,
    )
    prepared = PreparedRequest(
        path_template="/apis/jobs/v2/workspaces/{workspace}/jobs/{name}/status",
        path_params={"workspace": "ws", "name": "demo-1"},
        method="GET",
        content=None,
        content_type=None,
        response_type=HelixJobStatusResponse,
    )
    http_response = httpx.Response(200, request=httpx.Request("GET", "http://jobs"))
    return NemoResponse[HelixJobStatusResponse](http_response=http_response, body=body, request=prepared)


class FakeJobs:
    def __init__(self, status: HelixJobStatus | None = None, *, error: Exception | None = None) -> None:
        self.status = status
        self.error = error
        self.asked: list[tuple[str | None, str]] = []

    async def get_job_status(self, *, workspace: str | None = None, name: str) -> NemoResponse[HelixJobStatusResponse]:
        self.asked.append((workspace, name))
        if self.error is not None:
            raise self.error
        if self.status is None:
            raise NotFoundError(httpx.Response(404, json={"detail": "no"}, request=httpx.Request("GET", "http://j")))
        return _response(self.status)


class FakeEntities:
    def __init__(self, rows: Sequence[ContainerImage] = (), *, conflict: bool = False) -> None:
        self.rows = tuple(rows)
        self.conflict = conflict
        self.updated: list[ContainerImage] = []
        self.asked: list[tuple[str | None, int, dict[str, object] | None]] = []

    async def list(
        self,
        entity_type: type[ContainerImage],
        *,
        workspace: str,
        filter_obj: dict[str, object] | None = None,
        sort: str | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> ListResponse[ContainerImage]:
        self.asked.append((sort, page, filter_obj))
        data = self.rows[(page - 1) * page_size : page * page_size]
        return ListResponse(
            data=list(data),
            pagination=PaginationInfo(
                page=page,
                page_size=page_size,
                current_page_size=len(data),
                total_pages=-(-len(self.rows) // page_size),
                total_results=len(self.rows),
            ),
        )

    async def update(self, entity: ContainerImage) -> ContainerImage:
        if self.conflict:
            raise EntityConflictError("version mismatch")
        self.updated.append(entity)
        return entity


def _sweep(jobs: FakeJobs, entities: FakeEntities | None = None) -> tuple[BuilderController, FakeEntities]:
    controller = BuilderController()
    entities = entities or FakeEntities()
    controller._jobs = jobs
    controller._entities = entities
    return controller, entities


class TestWhenARowFails:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [HelixJobStatus.ERROR, HelixJobStatus.CANCELLED, HelixJobStatus.COMPLETED])
    async def test_its_job_ended_without_delivering_a_signature(self, status: HelixJobStatus) -> None:
        """Completed too: its last step delivers its signatures before it exits, so none is coming."""
        controller, entities = _sweep(FakeJobs(status))
        await controller.reconcile_one(_row())
        assert [row.status for row in entities.updated] == ["failed"]
        assert entities.updated[0].status_detail is not None and status.value in entities.updated[0].status_detail

    @pytest.mark.asyncio
    @pytest.mark.parametrize("naive", [False, True])
    async def test_its_job_never_appeared_and_the_grace_period_is_over(self, naive: bool) -> None:
        """Rows are written before the job; a submit that died between the two leaves these."""
        controller, entities = _sweep(FakeJobs(None))
        await controller.reconcile_one(_row(age=timedelta(minutes=10), naive=naive))
        assert entities.updated[0].status == "failed"
        assert "never created" in (entities.updated[0].status_detail or "")


class TestWhenItWaits:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status", [HelixJobStatus.CREATED, HelixJobStatus.PENDING, HelixJobStatus.ACTIVE, HelixJobStatus.CANCELLING]
    )
    async def test_while_the_job_is_running(self, status: HelixJobStatus) -> None:
        controller, entities = _sweep(FakeJobs(status))
        await controller.reconcile_one(_row())
        assert entities.updated == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("naive", [False, True])
    async def test_while_a_missing_job_may_still_be_about_to_exist(self, naive: bool) -> None:
        controller, entities = _sweep(FakeJobs(None))
        await controller.reconcile_one(_row(age=timedelta(seconds=5), naive=naive))
        assert entities.updated == []

    @pytest.mark.asyncio
    async def test_while_jobs_cannot_be_asked(self) -> None:
        """A 503 says nothing about the build. Failing on it would be permanent, and wrong."""
        error = InternalServerError(
            httpx.Response(503, json={"detail": "down"}, request=httpx.Request("GET", "http://j"))
        )
        controller, entities = _sweep(FakeJobs(error=error))
        await controller.reconcile_one(_row())
        assert entities.updated == []


class TestWhatItNeverTouches:
    @pytest.mark.asyncio
    async def test_a_row_its_signature_completed_a_moment_ago(self) -> None:
        """The write is conditional on the version read; a signature moved it, so the sweep stands down."""
        controller, entities = _sweep(FakeJobs(HelixJobStatus.COMPLETED), FakeEntities(conflict=True))
        await controller.reconcile_one(_row())
        assert entities.updated == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["ready", "failed"])
    async def test_a_settled_row(self, status: str) -> None:
        jobs = FakeJobs(HelixJobStatus.ERROR)
        controller, entities = _sweep(jobs)
        row = _row()
        row.status = status  # ty: ignore[invalid-assignment]
        await controller.reconcile_one(row)
        assert entities.updated == [] and jobs.asked == []

    @pytest.mark.asyncio
    async def test_it_asks_jobs_about_the_rows_own_job_in_the_rows_workspace(self) -> None:
        jobs = FakeJobs(HelixJobStatus.ACTIVE)
        controller, _ = _sweep(jobs)
        await controller.reconcile_one(_row(job="ws/demo-7"))
        assert jobs.asked == [("ws", "demo-7")]


class TestWhichRowsItReads:
    @pytest.mark.asyncio
    async def test_every_pending_row_oldest_first_across_every_page(self) -> None:
        rows = [_row(f"demo-1-{i}") for i in range(450)]
        controller, entities = _sweep(FakeJobs(HelixJobStatus.ACTIVE), FakeEntities(rows))
        listed = await controller.list_objects()
        assert len(listed) == 450
        assert [(sort, page) for sort, page, _ in entities.asked] == [
            ("created_at", 1),
            ("created_at", 2),
            ("created_at", 3),
        ]
        assert all(filters == {"status": "pending"} for _, _, filters in entities.asked)
