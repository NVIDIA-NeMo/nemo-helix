# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completion: who may make a row `ready`, the one write that does, and everything that must not."""

from __future__ import annotations

import pytest
from nemo_builder_plugin import completion as completion_module
from nemo_builder_plugin.completion import Caller, CompletionConflict, caller_refusal, complete, current_caller
from nemo_builder_plugin.entities import ContainerImage, Provenance
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.entities import EntityConflictError

DIGEST = "sha256:" + "d" * 64
STEP = Caller(auth=True, actor="job-step", submitter="alice")


def _row(*, submitter: str | None = "alice") -> ContainerImage:
    row = ContainerImage(
        name="build-1-0",
        workspace="ws-a",
        registry="reg.example.com",
        repository="ws-a/app",
        provenance=Provenance(build_set="build", revision=1, job="ws-a/build-1", request_digest="sha256:" + "a" * 64),
    )
    # What the entity store records for whoever created the row: the submit's effective principal.
    row._created_by = submitter
    return row


class FakeRows:
    def __init__(self, row: ContainerImage) -> None:
        self.row = row
        self.version = 1
        self.writes = 0
        self.interfere: ContainerImage | None = None

    async def get(self, entity_type: type[ContainerImage], *, name: str, workspace: str) -> ContainerImage:
        assert (name, workspace) == (self.row.name, self.row.workspace)
        copy = self.row.model_copy(deep=True)
        copy._db_version = self.version
        return copy

    async def update(self, entity: ContainerImage) -> ContainerImage:
        if self.interfere is not None:
            # Someone else writes between this completion's read and its write.
            self.row, self.interfere = self.interfere, None
            self.version += 1
        if entity.db_version != self.version:
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
