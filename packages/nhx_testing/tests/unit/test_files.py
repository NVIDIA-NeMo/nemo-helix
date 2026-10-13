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
async def test_upload_failure_status_fails_uploads_without_storing() -> None:
    service = InMemoryFilesService()
    files = service.client()
    await files.create_fileset(body=CreateFilesetRequest(name="bundle"))
    service.upload_failure_status = 500

    with pytest.raises(InternalServerError):
        await files.upload_file(name="bundle", path="bundle.json", content=b"{}")
    assert service.filesets == {("default", "bundle"): {}}
