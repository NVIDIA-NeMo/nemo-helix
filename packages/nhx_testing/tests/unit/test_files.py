# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the in-memory Files service."""

import pytest
from nemo_helix_plugin.client.errors import ConflictError, InternalServerError, NotFoundError
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nhx.testing.files import InMemoryFilesService


@pytest.mark.asyncio
async def test_round_trips_files_through_the_real_client() -> None:
    service = InMemoryFilesService()
    files = service.client()

    created = (await files.create_fileset(body=CreateFilesetRequest(name="bundle"))).data()
    uploaded = (await files.upload_file(name="bundle", path="dir/bundle.json", content=b"{}")).data()
    downloaded = await (await files.download_file(name="bundle", path="dir/bundle.json")).read()

    assert (created.workspace, created.name) == ("default", "bundle")
    assert uploaded.size == 2
    assert downloaded == b"{}"
    assert service.filesets == {("default", "bundle"): {"dir/bundle.json": b"{}"}}


@pytest.mark.asyncio
async def test_create_conflicts_unless_exist_ok_and_delete_removes() -> None:
    service = InMemoryFilesService()
    files = service.client()
    await files.create_fileset(body=CreateFilesetRequest(name="bundle"))

    with pytest.raises(ConflictError):
        await files.create_fileset(body=CreateFilesetRequest(name="bundle"))
    assert (await files.create_fileset(body=CreateFilesetRequest(name="bundle"), exist_ok=True)).data().name == "bundle"

    await files.delete_fileset(name="bundle")
    assert service.filesets == {}
    with pytest.raises(NotFoundError):
        await files.delete_fileset(name="bundle")


@pytest.mark.asyncio
async def test_failures_answer_with_a_status_or_raise_without_storing() -> None:
    service = InMemoryFilesService()
    files = service.client()
    await files.create_fileset(body=CreateFilesetRequest(name="bundle"))
    service.failures["upload"] = 500
    service.fail_when = lambda operation, workspace, name: RuntimeError("boom") if operation == "download" else None

    with pytest.raises(InternalServerError):
        await files.upload_file(name="bundle", path="bundle.json", content=b"{}")
    with pytest.raises(RuntimeError, match="boom"):
        await (await files.download_file(name="bundle", path="bundle.json")).read()
    assert service.filesets == {("default", "bundle"): {}}
    assert service.calls == [
        ("create", "default", "bundle"),
        ("upload", "default", "bundle"),
        ("download", "default", "bundle"),
    ]


@pytest.mark.asyncio
async def test_after_upload_runs_once_the_content_is_stored() -> None:
    service = InMemoryFilesService()
    files = service.client()
    await files.create_fileset(body=CreateFilesetRequest(name="bundle"))
    seen: list[dict[str, bytes]] = []

    async def after_upload() -> None:
        seen.append(dict(service.filesets[("default", "bundle")]))

    service.after_upload = after_upload
    await files.upload_file(name="bundle", path="bundle.json", content=b"{}")
    assert seen == [{"bundle.json": b"{}"}]


@pytest.mark.asyncio
async def test_percent_escapes_in_file_paths_are_decoded_once() -> None:
    service = InMemoryFilesService()
    files = service.client()
    await files.create_fileset(body=CreateFilesetRequest(name="bundle"))

    await files.upload_file(name="bundle", path="dir/a%2Fb.json", content=b"{}")

    assert service.filesets == {("default", "bundle"): {"dir/a%2Fb.json": b"{}"}}
    assert await (await files.download_file(name="bundle", path="dir/a%2Fb.json")).read() == b"{}"
