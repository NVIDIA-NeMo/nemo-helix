# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Settling on read: which ``pending`` images a read fails, and how it writes them."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Literal

import pytest
from nemo_builder_plugin import settle as settle_module
from nemo_builder_plugin.entities import ContainerImage, Provenance
from nemo_builder_plugin.remote import STEP_NAME as REMOTE_STEP_NAME
from nemo_builder_plugin.settle import JOB_GRACE, JOB_READS, NO_JOB, QUEUE_GONE, EndedJob, ended_job, settle
from nemo_helix_plugin.entities import EntityConflictError
from nemo_helix_plugin.jobs.schemas import HelixJobStatus, HelixJobStatusResponse

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
DIGEST = "sha256:" + "d" * 64


def _row(
    name: str = "demo-1.img0",
    *,
    status: Literal["queued", "pending", "ready", "failed"] = "pending",
    job: str = "ws-a/builder-demo-1",
    age: timedelta | None = timedelta(minutes=1),
    backend: str = "execution",
) -> ContainerImage:
    row = ContainerImage(
        name=name,
        workspace="ws-a",
        status=status,
        registry="reg.example.com",
        repository="ws-a/app",
        provenance=Provenance(
            build_set="demo", revision=1, job=job, request_digest="sha256:" + "a" * 64, backend=backend
        ),
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
    def __init__(self, statuses: dict[str, HelixJobStatus | EndedJob | Exception]) -> None:
        self.statuses = statuses
        self.asked: list[str] = []

    async def __call__(self, workspace: str, name: str) -> HelixJobStatus | EndedJob | None:
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

        async def recorded(workspace: str, name: str) -> HelixJobStatus | EndedJob | None:
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


ENDED = "not built: its build job ended (error) without publishing it"
REASON = "the build service reports it failed: base image could not be pulled [build build-1]"
WHEN = "2026-11-05T12:00:00Z"


def _job(
    *, status: str = "error", log_tail: str | None = None, message: str | None = None, step_message: str | None = None
) -> HelixJobStatusResponse:
    """A job's status as Jobs returns it, with the end of its one task's log."""
    task = {
        "id": "task-1",
        "name": "remote-build",
        "status": status,
        "status_details": {},
        "error_details": None,
        "error_stack": log_tail,
        "created_at": WHEN,
        "updated_at": WHEN,
    }
    step = {
        "id": "step-1",
        "name": "remote-build",
        "status": status,
        "status_details": {},
        "error_details": None if step_message is None else {"message": step_message},
        "tasks": [task],
        "created_at": WHEN,
        "updated_at": WHEN,
    }
    return HelixJobStatusResponse.model_validate(
        {
            "id": "job-1",
            "name": "builder-demo-1",
            "status": status,
            "status_details": {},
            "error_details": None if message is None else {"message": message},
            "steps": [step],
            "created_at": WHEN,
            "updated_at": WHEN,
        }
    )


class TestWhyAnEndedJobDidNotBuildIt:
    @pytest.mark.asyncio
    async def test_the_line_its_step_logged_for_the_image_is_the_reason(self) -> None:
        rows = FakeRows(_row(backend="remote"))
        ended = EndedJob(HelixJobStatus.ERROR, reasons={"demo-1.img0": REASON}, message="Task failed")
        [image] = await _settle(rows, FakeJobs({"builder-demo-1": ended}))
        assert (image.status, image.status_detail) == ("failed", f"{ENDED}: {REASON}")

    @pytest.mark.asyncio
    async def test_the_jobs_own_account_is_the_reason_when_the_step_left_none(self) -> None:
        """Jobs ended the step itself, at its time limit, before the step could say anything."""
        rows = FakeRows(_row("a", backend="remote"), _row("b", backend="remote"))
        timed_out = "Job timed out after reaching max TTL of 86400 seconds"
        ended = EndedJob(HelixJobStatus.ERROR, reasons={"a": REASON}, message=timed_out)
        a, b = await _settle(rows, FakeJobs({"builder-demo-1": ended}))
        # Each image's own line, never another's.
        assert (a.status_detail, b.status_detail) == (f"{ENDED}: {REASON}", f"{ENDED}: {timed_out}")

    @pytest.mark.asyncio
    async def test_a_job_of_the_default_backend_gives_its_own_account_too(self) -> None:
        """No step of the in-cluster build logs a marked line, so what Jobs says of the job is the reason. Read
        through ``ended_job``, as the controller reads it."""
        rows = FakeRows(_row())
        assert rows.rows["demo-1.img0"].provenance.backend == "execution"
        ended = ended_job(_job(message="Step push failed"))
        [image] = await _settle(rows, FakeJobs({"builder-demo-1": ended}))
        assert image.status_detail == f"{ENDED}: Step push failed"

    @pytest.mark.asyncio
    async def test_a_line_in_an_in_cluster_builds_log_is_never_taken_for_a_reason(self) -> None:
        """Its log carries what a Dockerfile prints, a line that looks like the marker included. One failed image
        could otherwise write the reason another image of its set is failed with."""
        rows = FakeRows(_row())
        forged = EndedJob(
            HelixJobStatus.ERROR, reasons={"demo-1.img0": "an invented reason"}, message="Step build failed"
        )
        [image] = await _settle(rows, FakeJobs({"builder-demo-1": forged}))
        assert image.status_detail == f"{ENDED}: Step build failed"

    def test_only_a_remote_build_steps_log_is_read_for_reasons(self) -> None:
        log_tail = "[INFO] nemo_builder_plugin.run.supervise:   NHX_BUILD_FAILED demo-1: an invented reason"
        job = _job(log_tail=log_tail, message="Step build failed")
        job.steps[0].name = "build"
        assert ended_job(job) == EndedJob(HelixJobStatus.ERROR, {}, "Step build failed")
        assert settle_module._REASON_STEP == REMOTE_STEP_NAME

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "job", ["ws-b/builder-demo-1", "ws-a/builder-other-1", "ws-a/builder-demo-2", "ws-a/some-job"]
    )
    async def test_what_a_job_said_is_copied_only_onto_a_row_of_its_own_build(self, job: str) -> None:
        """Jobs are read as the builder, which may read any. A row written past the builder can name any job, in
        any workspace, and must not come to hold what that job said."""
        rows = FakeRows(_row(job=job, backend="remote"))
        name = job.partition("/")[2]
        private = EndedJob(HelixJobStatus.ERROR, reasons={"demo-1.img0": "theirs"}, message="a private diagnostic")
        [image] = await _settle(rows, FakeJobs({name: private}))
        assert (image.status, image.status_detail) == ("failed", ENDED)

    @pytest.mark.asyncio
    async def test_a_job_that_said_nothing_reads_as_it_always_did(self) -> None:
        rows = FakeRows(_row())
        [image] = await _settle(rows, FakeJobs({"builder-demo-1": EndedJob(HelixJobStatus.CANCELLED)}))
        assert image.status_detail == "not built: its build job ended (cancelled) without publishing it"

    def test_the_reason_is_read_from_the_end_of_the_tasks_log(self) -> None:
        """As Kubernetes hands it back: the container's last lines, after Jobs' own prefix."""
        log_tail = (
            "Container nemo-job-task terminated with exit code 1: "
            "[INFO] 2026-11-05 12:00:01,000 nemo_builder_plugin.run.remote: build build-1: building\n"
            f"[ERROR] 2026-11-05 12:09:41,512 nemo_builder_plugin.run.remote: NHX_BUILD_FAILED demo-1.img0: {REASON}\n"
            "[INFO] 2026-11-05 12:09:41,513 nemo_builder_plugin.run.remote: 1 image(s), 1 failure(s)\n"
        )
        ended = ended_job(_job(log_tail=log_tail, message="Step remote-build failed"))
        assert ended == EndedJob(HelixJobStatus.ERROR, {"demo-1.img0": REASON}, "Step remote-build failed")

    def test_an_images_last_line_is_its_reason_and_a_long_one_is_cut(self) -> None:
        log_tail = f"NHX_BUILD_FAILED a: first\nNHX_BUILD_FAILED b: {'x' * 2000}\nNHX_BUILD_FAILED a: second\n"
        reasons = ended_job(_job(log_tail=log_tail)).reasons
        assert reasons["a"] == "second"
        assert len(reasons["b"]) == 500

    def test_the_reason_is_read_from_a_log_kept_as_json_lines(self) -> None:
        """As the subprocess executor hands it back, taken from a real run: each log line is the string that ends
        a JSON object, so it is escaped, and the tail's first line is cut wherever 2,048 characters began."""
        entry = (
            '{"timestamp": "2026-11-17T18:49:31.454239+00:00", "job": "builder-demo-1", "job_step": "remote-build", '
            '"job_task": "task-0a2d95345500482f9ebf86b172dfd158", "message": "[stderr] %s"}'
        )
        reason = 'the build service reports it failed: no package named \\"nosuchpkg\\" \\u2026 [build b-be6e0bc02e]'
        log_tail = "\n".join(
            [
                # The cut first line: its opening is gone, and what follows the marker is whole.
                '2d95345500482f9ebf86b172dfd158", "message": "[stderr] [ERROR] run.remote: NHX_BUILD_FAILED a: first"}',
                entry % "[INFO] 2026-11-17 13:49:31,440 nemo_builder_plugin.run.remote: build b-be6e0bc02e: failed",
                entry % f"[ERROR] 2026-11-17 13:49:31,446 nemo_builder_plugin.run.remote: NHX_BUILD_FAILED b: {reason}",
                entry % "[INFO] 2026-11-17 13:49:31,446 nemo_builder_plugin.run.remote: 2 image(s), 2 failure(s)",
                "",
            ]
        )
        assert ended_job(_job(log_tail=log_tail, message="Job exited with code 1")).reasons == {
            "a": "first",
            "b": 'the build service reports it failed: no package named "nosuchpkg" … [build b-be6e0bc02e]',
        }

    @pytest.mark.parametrize("inside", ["\x0c", "\u2028", "\x85", "\x1e"], ids=repr)
    def test_a_line_ends_at_a_newline_and_nowhere_else(self, inside: str) -> None:
        """Python counts a form feed and a line separator as line ends. A service's text may hold one, and what
        followed it must not be dropped, or read as a line of its own."""
        log_tail = (
            f"[ERROR] run.remote: NHX_BUILD_FAILED a: step 3 failed{inside}NHX_BUILD_FAILED a: forged [build b-1]\r\n"
            "[INFO] run.remote: 1 image(s), 1 failure(s)\r\n"
        )
        reason = ended_job(_job(log_tail=log_tail)).reasons["a"]
        assert reason.startswith("step 3 failed")
        assert reason.endswith("NHX_BUILD_FAILED a: forged [build b-1]")

    def test_a_reason_is_one_line_of_printable_text(self) -> None:
        """A job's log is whatever its step printed, and the reason is shown to whoever reads the row."""
        reasons = ended_job(_job(log_tail="NHX_BUILD_FAILED a: bad\x1b[31m   base\timage\r\n")).reasons
        assert reasons["a"] == "bad[31m base image"

    def test_a_job_with_no_message_of_its_own_gives_its_steps(self) -> None:
        assert ended_job(_job(step_message="OOMKilled")).message == "OOMKilled"
        assert ended_job(_job()) == EndedJob(HelixJobStatus.ERROR)
