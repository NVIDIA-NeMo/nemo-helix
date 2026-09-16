# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A taskset's own files: one Files reference, shared by its members and owned by none of them."""

from __future__ import annotations

import hashlib

import pytest
from nemo_evaluator.api.schemas import TaskRef, TasksetInput
from nemo_evaluator.api.service.taskset_service import TasksetService
from nemo_platform_plugin.entity_client import NemoEntityNotFoundError
from pydantic import ValidationError

DIGEST_A = hashlib.sha256(b"files-v1").hexdigest()
DIGEST_B = hashlib.sha256(b"files-v2").hexdigest()
REF_A = f"default/harbor-packages#packages/nvidia.ds/{DIGEST_A}"
REF_B = f"default/harbor-packages#packages/nvidia.ds/{DIGEST_B}"


class _FakeTaskService:
    def __init__(self, existing: set[tuple[str, str]]) -> None:
        self.existing = existing

    async def get_task(self, workspace: str, name: str) -> object | None:
        return object() if (workspace, name) in self.existing else None

    async def resolve_revision(self, workspace: str, name: str, fragment: str = "latest") -> str:
        if (workspace, name) not in self.existing:
            raise NemoEntityNotFoundError(f"{workspace}/{name} not found")
        return hashlib.sha256(f"{workspace}/{name}#{fragment}".encode()).hexdigest()


@pytest.fixture
def service(entity_store) -> TasksetService:
    return TasksetService(entity_store, _FakeTaskService({("default", "task-a")}))


def _input(files_ref: str | None = None) -> TasksetInput:
    return TasksetInput(tasks=[TaskRef("task-a")], files_ref=files_ref)


async def test_files_ref_round_trips(service: TasksetService) -> None:
    created, _ = await service.create_taskset("ts-1", _input(REF_A), workspace="default")
    assert created.files_ref == REF_A

    got = await service.get_taskset("default", "ts-1")
    assert got is not None and got.files_ref == REF_A


async def test_repointing_the_files_publishes_a_revision(service: TasksetService) -> None:
    """The reference is content, not annotation: changing where a taskset's files come from
    changes what the taskset is."""
    created, _ = await service.create_taskset("ts-1", _input(REF_A), workspace="default")
    assert created.revision == 1

    replaced, published = await service.replace_taskset("ts-1", _input(REF_B), workspace="default")

    assert published is True
    assert replaced.revision == 2
    assert replaced.files_ref == REF_B


async def test_republishing_the_same_ref_publishes_nothing(service: TasksetService) -> None:
    await service.create_taskset("ts-1", _input(REF_A), workspace="default")

    replaced, published = await service.replace_taskset("ts-1", _input(REF_A), workspace="default")

    assert published is False
    assert replaced.revision == 1


async def test_a_pinned_revision_keeps_the_ref_it_was_published_with(service: TasksetService) -> None:
    """Why the digest covers the reference at all: revision 1 must still name the files it was
    published with, after the head has been repointed."""
    await service.create_taskset("ts-1", _input(REF_A), workspace="default")
    await service.replace_taskset("ts-1", _input(REF_B), workspace="default")

    # Selected by content digest, not ordinal: a non-digest fragment is read as a *tag* name, so
    # "1" would be a lookup for a tag called "1".
    revisions = await service.list_revisions("default", "ts-1")
    first_digest = next(r.content_hash for r in revisions.data if r.revision == 1)

    first = await service.get_taskset("default", "ts-1", revision=first_digest)
    assert first is not None and first.files_ref == REF_A

    head = await service.get_taskset("default", "ts-1")
    assert head is not None and head.files_ref == REF_B


async def test_the_ref_defaults_to_none(service: TasksetService) -> None:
    """The field is additive: a taskset that ships no files is exactly as it was before."""
    created, _ = await service.create_taskset("ts-1", _input(), workspace="default")
    assert created.files_ref is None


async def test_clearing_the_ref_publishes_a_revision(service: TasksetService) -> None:
    """Dropping a taskset's files is as much a content change as repointing them."""
    await service.create_taskset("ts-1", _input(REF_A), workspace="default")

    replaced, published = await service.replace_taskset("ts-1", _input(None), workspace="default")

    assert published is True
    assert replaced.files_ref is None


def test_a_plain_prefix_is_accepted() -> None:
    """The simplest useful form: a fileset plus the prefix its files sit under."""
    assert TasksetInput(files_ref="default/my-dataset#files").files_ref == "default/my-dataset#files"


def test_the_ref_must_be_a_fileset_reference() -> None:
    """Including the fragment. Same shape as ``bundle_ref``/Harbor tree references, so a bare
    ``workspace/fileset`` is not a Files reference here and is rejected rather than quietly
    stored as something no reader can resolve."""
    for bad in ("not a ref", "", "default/my-dataset", "default/fs#bad path"):
        with pytest.raises(ValidationError):
            TasksetInput(files_ref=bad)
