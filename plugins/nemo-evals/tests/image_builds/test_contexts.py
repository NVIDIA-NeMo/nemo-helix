# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Uploading a generated build context to Files as one tar archive."""

from __future__ import annotations

import io
import tarfile
import tempfile
from pathlib import Path

import pytest
from nemo_builder_plugin.schema import FileSetSource
from nemo_evals.image_builds.contexts import upload_context
from nemo_helix_plugin.client.errors import InternalServerError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nhx.testing.files import InMemoryFilesService
from pydantic import ValidationError


def _generated(tmp_path: Path) -> Path:
    """A generated context on disk, in a directory named ``generated``: a Dockerfile, and a script in a folder."""
    directory = tmp_path / "generated"
    (directory / "bin").mkdir(parents=True)
    (directory / "Dockerfile").write_text("FROM ubuntu:24.04\n", encoding="utf-8")
    (directory / "bin" / "setup.sh").write_text("echo hi\n", encoding="utf-8")
    return directory


def _members(upload: bytes) -> dict[str, bytes | None]:
    """Every member of an uploaded archive by name: a file's content, or None for a directory."""
    members: dict[str, bytes | None] = {}
    with tarfile.open(fileobj=io.BytesIO(upload), mode="r:gz") as archive:
        for member in archive.getmembers():
            handle = archive.extractfile(member) if member.isfile() else None
            members[member.name] = handle.read() if handle is not None else None
    return members


def _gzip_name(upload: bytes) -> bytes | None:
    """The file name an uploaded archive's gzip header records, or None when it records none.

    The header is ten fixed bytes, the fourth of them its flags, and then the name up to a zero byte.
    """
    if not upload[3] & 0x08:
        return None
    return upload[10 : upload.index(b"\x00", 10)]


async def test_the_archive_holds_the_directory_under_its_root(
    files_service: InMemoryFilesService, files_client: AsyncFilesClient, tmp_path: Path
) -> None:
    await upload_context(
        files_client, "default", "harbor-build", _generated(tmp_path), archive="context.tar.gz", root="context"
    )

    uploaded = files_service.filesets[("default", "harbor-build")]
    assert list(uploaded) == ["context.tar.gz"]
    # Under the root it was given: the directory's own name isn't in the archive.
    assert _members(uploaded["context.tar.gz"]) == {
        "context": None,
        "context/Dockerfile": b"FROM ubuntu:24.04\n",
        "context/bin": None,
        "context/bin/setup.sh": b"echo hi\n",
    }


async def test_an_existing_fileset_is_reused(
    files_service: InMemoryFilesService, files_client: AsyncFilesClient, tmp_path: Path
) -> None:
    """An earlier attempt at the same build left its fileset: the upload goes into it, over the archive it held."""
    files_service.filesets[("default", "harbor-build")] = {"context.tar.gz": b"earlier", "mounts.tar.gz": b"kept"}

    await upload_context(
        files_client, "default", "harbor-build", _generated(tmp_path), archive="context.tar.gz", root="context"
    )

    uploaded = files_service.filesets[("default", "harbor-build")]
    assert "context/Dockerfile" in _members(uploaded["context.tar.gz"])
    assert uploaded["mounts.tar.gz"] == b"kept"


async def test_the_source_names_the_fileset_the_archive_and_the_context_path(
    files_service: InMemoryFilesService, files_client: AsyncFilesClient, tmp_path: Path
) -> None:
    """The fileset by the workspace it was uploaded to, which isn't the client's own here."""
    source = await upload_context(
        files_client, "team", "harbor-build", _generated(tmp_path), archive="context.tar.gz", root="context"
    )

    assert source == FileSetSource(fileset="team/harbor-build", archive="context.tar.gz", context_path="context")
    assert list(files_service.filesets) == [("team", "harbor-build")]


async def test_a_second_archive_goes_into_the_fileset_without_asking_for_it_again(
    files_service: InMemoryFilesService, files_client: AsyncFilesClient, tmp_path: Path
) -> None:
    directory = _generated(tmp_path)
    await upload_context(files_client, "default", "harbor-build", directory, archive="context.tar.gz", root="context")

    source = await upload_context(
        files_client, "default", "harbor-build", directory, archive="mounts.tar.gz", root="mounts", create_fileset=False
    )

    assert files_service.calls == [
        ("create", "default", "harbor-build"),
        ("upload", "default", "harbor-build"),
        ("upload", "default", "harbor-build"),
    ]
    assert source == FileSetSource(fileset="default/harbor-build", archive="mounts.tar.gz", context_path="mounts")
    assert set(_members(files_service.filesets[("default", "harbor-build")]["mounts.tar.gz"])) == {
        "mounts",
        "mounts/Dockerfile",
        "mounts/bin",
        "mounts/bin/setup.sh",
    }


@pytest.mark.parametrize(
    ("archive", "recorded"), [("context.tar.gz", b"context.tar"), ("layers/mounts.tar.gz", b"mounts.tar")]
)
async def test_the_archive_is_packed_under_its_own_file_name(
    files_service: InMemoryFilesService, files_client: AsyncFilesClient, tmp_path: Path, archive: str, recorded: bytes
) -> None:
    """Gzip records the name of the file it wrote, less ``.gz``: the archive's, in whichever folder of the fileset."""
    source = await upload_context(
        files_client, "default", "harbor-build", _generated(tmp_path), archive=archive, root="context"
    )

    uploaded = files_service.filesets[("default", "harbor-build")]
    assert list(uploaded) == [archive]
    assert _gzip_name(uploaded[archive]) == recorded
    assert "context/Dockerfile" in _members(uploaded[archive])
    assert source.archive == archive


async def test_nothing_is_left_on_disk_whether_or_not_the_upload_goes_through(
    files_service: InMemoryFilesService, files_client: AsyncFilesClient, tmp_path: Path
) -> None:
    """The temporary directory is empty by the upload and after it, and nothing is left beside the context."""
    directory = _generated(tmp_path)
    # The suite's conftest points tempfile at a directory of this test's own, so anything left there is this call's.
    scratch = Path(tempfile.gettempdir())
    during_upload: list[list[Path]] = []

    async def look() -> None:
        during_upload.append(list(scratch.iterdir()))

    files_service.after_upload = look

    await upload_context(files_client, "default", "harbor-build", directory, archive="context.tar.gz", root="context")
    files_service.failures["upload"] = 500
    with pytest.raises(InternalServerError):
        await upload_context(
            files_client, "default", "harbor-build", directory, archive="context.tar.gz", root="context"
        )

    assert during_upload == [[]]
    assert list(scratch.iterdir()) == []
    assert list(tmp_path.iterdir()) == [directory]


@pytest.mark.parametrize(
    ("workspace", "archive", "root"),
    [
        ("Team One", "context.tar.gz", "context"),
        ("default", "../context.tar.gz", "context"),
        ("default", "context.tar.gz", "../context"),
    ],
)
async def test_a_source_builder_would_refuse_is_refused_before_anything_is_made(
    files_service: InMemoryFilesService,
    files_client: AsyncFilesClient,
    tmp_path: Path,
    workspace: str,
    archive: str,
    root: str,
) -> None:
    """Not after the upload: Files is asked for nothing, and nothing is left on disk."""
    directory = _generated(tmp_path)

    with pytest.raises(ValidationError):
        await upload_context(files_client, workspace, "harbor-build", directory, archive=archive, root=root)

    assert files_service.calls == []
    assert list(Path(tempfile.gettempdir()).iterdir()) == []
    assert list(tmp_path.iterdir()) == [directory]
