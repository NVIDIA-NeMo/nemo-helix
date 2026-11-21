# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completion: who may make a row `ready`, the one write that does, and everything that must not."""

from __future__ import annotations

import pytest
from nemo_builder_plugin import completion as completion_module
from nemo_builder_plugin.completion import Caller, CompletionConflict, caller_refusal, complete, current_caller
from nemo_builder_plugin.entities import ContainerImage, ImageRuntimeConfig, Provenance, UpstreamImage
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.entities import EntityConflictError, EntityNotFoundError

DIGEST = "sha256:" + "d" * 64
STEP = Caller(auth=True, actor="job-step", submitter="alice")


def _row(*, submitter: str | None = "alice", entity_id: str | None = None) -> ContainerImage:
    row = ContainerImage(
        name="build-1-0",
        workspace="ws-a",
        registry="reg.example.com",
        repository="ws-a/app",
        provenance=Provenance(build_set="build", revision=1, job="ws-a/build-1", request_digest="sha256:" + "a" * 64),
    )
    # What the entity store records for whoever created the row: the submit's effective principal.
    row._created_by = submitter
    # And the ID it gave the row, which no row created under the name later is given.
    row._id = entity_id
    return row


class FakeRows:
    """The one row under a name. It is read by the name, and written by its ID at the version it was read at."""

    def __init__(self, row: ContainerImage) -> None:
        self.row = row
        self.version = 1
        self.reads = 0
        self.writes = 0
        self.interfere: ContainerImage | None = None
        #: The row that has taken the name by the time it is read after a write was refused: the first was
        #: deleted meanwhile.
        self.replaced_by: ContainerImage | None = None
        self._refused = False

    async def get(self, entity_type: type[ContainerImage], *, name: str, workspace: str) -> ContainerImage:
        assert (name, workspace) == (self.row.name, self.row.workspace)
        if self.replaced_by is not None and self._refused:
            self.row, self.replaced_by, self.version = self.replaced_by, None, 1
        self.reads += 1
        copy = self.row.model_copy(deep=True)
        copy._db_version = self.version
        return copy

    async def update(self, entity: ContainerImage) -> ContainerImage:
        if self.interfere is not None:
            # Someone else writes between this completion's read and its write.
            self.row, self.interfere = self.interfere, None
            self.version += 1
        if entity.id != self.row.id:
            raise EntityNotFoundError("no row has that ID any more")
        if entity.db_version != self.version:
            self._refused = True
            raise EntityConflictError("version mismatch")
        self.writes += 1
        self.version += 1
        self.row = entity
        return entity


async def _complete(rows: FakeRows, digest: str = DIGEST) -> ContainerImage:
    return await complete(rows, workspace="ws-a", name="build-1-0", digest=digest)


class TestTheCaller:
    def test_with_platform_auth_off_no_caller_is_checked(self) -> None:
        assert caller_refusal(_row(), Caller(auth=False)) is None

    def test_a_job_step_acting_for_the_images_submitter_may(self) -> None:
        assert caller_refusal(_row(), STEP) is None

    def test_a_users_own_token_is_refused(self) -> None:
        refusal = caller_refusal(_row(), Caller(auth=True, actor="alice"))
        assert refusal is not None and "job step" in refusal

    def test_a_job_step_acting_for_someone_else_is_refused(self) -> None:
        refusal = caller_refusal(_row(), Caller(auth=True, actor="job-step", submitter="bob"))
        assert refusal is not None and "someone else" in refusal

    def test_a_row_with_no_recorded_submitter_is_refused(self) -> None:
        assert caller_refusal(_row(submitter=None), STEP) is not None

    def test_the_caller_comes_from_the_platforms_auth_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        context = AuthContext(principal_id="job-step", principal_on_behalf_of="alice")
        monkeypatch.setattr(completion_module, "current_auth_context", lambda: context)
        monkeypatch.setattr(completion_module, "platform_auth_enabled", lambda: True)
        assert current_caller() == STEP
        monkeypatch.setattr(completion_module, "platform_auth_enabled", lambda: False)
        assert current_caller() == Caller(auth=False)


class TestComplete:
    @pytest.mark.asyncio
    async def test_a_pending_row_is_made_ready_in_one_write(self) -> None:
        rows = FakeRows(_row())
        ready = await _complete(rows)
        assert (ready.status, ready.digest) == ("ready", DIGEST)
        assert rows.writes == 1

    @pytest.mark.asyncio
    async def test_how_the_image_starts_is_written_with_it(self) -> None:
        rows = FakeRows(_row())
        runtime_config = ImageRuntimeConfig(entrypoint=["/bin/bash", "-l", "-c"], working_dir="/app")
        ready = await complete(rows, workspace="ws-a", name="build-1-0", digest=DIGEST, runtime_config=runtime_config)
        assert ready.runtime_config == runtime_config
        assert rows.row.runtime_config == runtime_config

    @pytest.mark.asyncio
    async def test_the_same_digest_again_is_accepted_and_writes_nothing(self) -> None:
        rows = FakeRows(_row())
        await _complete(rows)
        again = await _complete(rows)
        assert again.status == "ready" and rows.writes == 1

    @pytest.mark.asyncio
    async def test_another_digest_on_a_ready_row_is_a_conflict(self) -> None:
        rows = FakeRows(_row())
        await _complete(rows)
        with pytest.raises(CompletionConflict, match="already ready"):
            await _complete(rows, "sha256:" + "e" * 64)

    @pytest.mark.asyncio
    async def test_a_failed_row_is_never_made_ready(self) -> None:
        row = _row()
        row.status, row.status_detail = "failed", "not built: build revision 1 belongs to a different request"
        rows = FakeRows(row)
        with pytest.raises(CompletionConflict, match="already failed"):
            await _complete(rows)
        assert rows.writes == 0

    @pytest.mark.asyncio
    async def test_a_row_failed_in_between_wins(self) -> None:
        rows = FakeRows(_row())
        failed = _row()
        failed.status, failed.status_detail = "failed", "not built: build revision 1 belongs to a different request"
        rows.interfere = failed
        with pytest.raises(CompletionConflict, match="already failed"):
            await _complete(rows)

    @pytest.mark.asyncio
    async def test_a_twin_completion_that_finishes_first_is_success(self) -> None:
        rows = FakeRows(_row())
        twin = await _complete(FakeRows(_row()))
        rows.interfere = twin
        assert (await _complete(rows)).status == "ready"


class Refused(Exception):
    """What a route's checks raise for a caller who may not complete a row."""


class Checks:
    """The route's checks, as it hands them to ``complete``: which rows they were asked about."""

    def __init__(self, refuse: bool = False) -> None:
        self.asked: list[ContainerImage] = []
        self.refuse = refuse

    async def __call__(self, row: ContainerImage) -> None:
        self.asked.append(row)
        if self.refuse:
            raise Refused(f"{row.name}, as it is now, is not this caller's to complete")


class TestTheRowThatIsWritten:
    """A route decides who may complete a row, of the row it read. The write is of that row."""

    async def _complete(self, rows: FakeRows, row: ContainerImage | None, admit: Checks | None) -> ContainerImage:
        return await complete(rows, workspace="ws-a", name="build-1-0", digest=DIGEST, row=row, admit=admit)

    async def _read(self, rows: FakeRows) -> ContainerImage:
        """The row as a route reads it before it decides. Not counted among the completion's own reads."""
        row = await rows.get(ContainerImage, name="build-1-0", workspace="ws-a")
        rows.reads = 0
        return row

    @pytest.mark.asyncio
    async def test_the_row_handed_in_is_written_and_the_name_is_not_read_again(self) -> None:
        rows, admit = FakeRows(_row(entity_id="image-1")), Checks()
        ready = await self._complete(rows, await self._read(rows), admit)
        assert (ready.status, ready.digest, ready.id) == ("ready", DIGEST, "image-1")
        assert (rows.reads, rows.writes) == (0, 1)
        # It was asked about before it was handed in: nothing is asked again of the same row at the same version.
        assert admit.asked == []

    @pytest.mark.asyncio
    async def test_a_row_that_took_the_name_since_is_not_written(self) -> None:
        """The write is by the ID of the row that was decided of, and the store has no row under it any more.
        Read by its name again, the completion would have made someone else's row ready."""
        rows = FakeRows(_row(entity_id="image-1"))
        decided_of = await self._read(rows)
        rows.interfere = _row(submitter="bob", entity_id="image-2")
        with pytest.raises(EntityNotFoundError):
            await self._complete(rows, decided_of, Checks())
        assert (rows.reads, rows.writes) == (0, 0)
        assert (rows.row.status, rows.row.created_by) == ("pending", "bob")

    @pytest.mark.asyncio
    async def test_a_row_that_changed_is_read_again_and_asked_about_again_before_it_is_written(self) -> None:
        rows, admit = FakeRows(_row(entity_id="image-1")), Checks()
        decided_of = await self._read(rows)
        rows.interfere = _row(entity_id="image-1")
        ready = await self._complete(rows, decided_of, admit)
        assert (ready.status, rows.reads, rows.writes) == ("ready", 1, 1)
        # Of the row as it was read the second time: the same one, at its next version.
        assert [(row.id, row.db_version) for row in admit.asked] == [("image-1", 2)]

    @pytest.mark.asyncio
    async def test_a_row_that_changed_and_is_no_longer_the_callers_to_complete_is_not_written(self) -> None:
        """What the checks raise is what the completion ends with: a route turns it into its own answer."""
        rows = FakeRows(_row(entity_id="image-1"))
        decided_of = await self._read(rows)
        rows.interfere = _row(entity_id="image-1")
        with pytest.raises(Refused, match="as it is now"):
            await self._complete(rows, decided_of, Checks(refuse=True))
        assert (rows.reads, rows.writes, rows.row.status) == (1, 0, "pending")

    @pytest.mark.asyncio
    async def test_a_row_replaced_before_it_is_read_again_ends_the_completion(self) -> None:
        """The first write is refused because the row changed. By the time the name is read again the row is
        gone, and another is under its name: nothing was decided of that one, and it is not asked about."""
        rows, admit = FakeRows(_row(entity_id="image-1")), Checks()
        decided_of = await self._read(rows)
        rows.interfere = _row(entity_id="image-1")
        rows.replaced_by = _row(submitter="bob", entity_id="image-2")
        with pytest.raises(CompletionConflict, match="build-1-0 was replaced while it was being completed"):
            await self._complete(rows, decided_of, admit)
        assert (rows.reads, rows.writes, admit.asked) == (1, 0, [])
        assert (rows.row.id, rows.row.status) == ("image-2", "pending")

    @pytest.mark.asyncio
    async def test_without_a_row_the_name_is_read_and_every_row_read_is_asked_about(self) -> None:
        rows, admit = FakeRows(_row(entity_id="image-1")), Checks()
        rows.interfere = _row(entity_id="image-1")
        ready = await self._complete(rows, None, admit)
        assert (ready.status, rows.reads, rows.writes) == ("ready", 2, 1)
        assert [row.db_version for row in admit.asked] == [1, 2]
        # And the row the first read found is the one it stays with.
        rows = FakeRows(_row(entity_id="image-1"))
        rows.interfere = _row(entity_id="image-1")
        rows.replaced_by = _row(submitter="bob", entity_id="image-2")
        with pytest.raises(CompletionConflict, match="was replaced"):
            await self._complete(rows, None, None)

    @pytest.mark.asyncio
    async def test_a_repeat_returns_the_row_handed_in_as_it_is(self) -> None:
        rows = FakeRows(_row(entity_id="image-1"))
        await self._complete(rows, None, None)
        again = await self._complete(rows, await self._read(rows), Checks())
        assert (again.status, again.digest, rows.reads, rows.writes) == ("ready", DIGEST, 0, 1)


class TestCompletingACopy:
    UPSTREAM = "sha256:" + "e" * 64

    def _copy(self) -> ContainerImage:
        row = _row()
        row.provenance.upstream = UpstreamImage(image="docker.io/library/ubuntu:24.04", manifest_digest=self.UPSTREAM)
        return row

    @pytest.mark.asyncio
    async def test_a_copy_completes_at_the_manifest_it_copies(self) -> None:
        ready = await _complete(FakeRows(self._copy()), digest=self.UPSTREAM)
        assert (ready.status, ready.digest) == ("ready", self.UPSTREAM)

    @pytest.mark.asyncio
    async def test_a_copy_completed_at_any_other_digest_is_a_conflict_and_writes_nothing(self) -> None:
        """Anything in the cluster can call complete, so the server holds a copy to the digest it resolved."""
        rows = FakeRows(self._copy())
        with pytest.raises(CompletionConflict, match="copies"):
            await _complete(rows, digest=DIGEST)
        assert rows.writes == 0
