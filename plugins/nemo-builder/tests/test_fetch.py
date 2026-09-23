# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`fetch` against a fake Files service that reports paths the way the real one does.

The detail that matters: a listing narrowed to ``path=tests`` still reports each entry relative
to the fileset ROOT -- ``tests/Dockerfile``, not ``Dockerfile``
(``services/core/files/.../backends/local.py``, ``relative_to(resolved_path)``). A fetch that
missed this nested every subtree inside itself, and the sandbox found no Dockerfile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import pytest
from nemo_builder_plugin.run.fetch import fetch_source
from nemo_builder_plugin.run.supervise import _volume_mounts
from nemo_builder_plugin.steps import ContextSource, SandboxGroup, SandboxImage, WorkLayout
from nemo_helix_plugin.files.client import FilesClient

JOB_SLICE = "jobs/default/abc"


@dataclass
class _Entry:
    path: str


@dataclass
class _Listing:
    data: list[_Entry]


class FakeFiles:
    """Lists and serves one fileset. Entry paths are root-relative, as in the real service."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def list_files(self, *, workspace: str, name: str, query_params: dict | None = None):
        prefix = (query_params or {}).get("path")
        entries = [_Entry(p) for p in self.files if prefix is None or p.startswith(f"{prefix.rstrip('/')}/")]

        class _Response:
            def data(self) -> _Listing:
                return _Listing(entries)

        return _Response()

    def download_file(self, *, workspace: str, name: str, path: str):
        content = self.files[path]

        class _Body:
            def read(self) -> bytes:
                return content

        return _Body()


def _client(files: dict[str, bytes]) -> FilesClient:
    # A structural stand-in for the two calls `fetch` makes; the generated client is not a Protocol.
    return cast(FilesClient, FakeFiles(files))


FILESET = {"tests/Dockerfile": b"FROM tests", "env/Dockerfile": b"FROM env", "README": b"hi"}


def _sandbox_context(tmp_path: Path, source: ContextSource) -> Path:
    """The directory the sandbox would see, found the way `supervise` finds it: from the mount."""
    group = SandboxGroup(
        source=source, images=[SandboxImage(image="demo-1-0", platform="linux/amd64", dockerfile="Dockerfile")]
    )
    context_mount = _volume_mounts(group, JOB_SLICE)[0]
    return tmp_path / PurePosixPath(context_mount.sub_path).relative_to(JOB_SLICE)


class TestWhatTheSandboxSees:
    def test_a_subtree_lands_where_the_sandbox_mounts_it(self, tmp_path: Path) -> None:
        """The regression: this used to produce `context/fs-a/tests/tests/Dockerfile`."""
        source = ContextSource(fileset="fs-a", context_path="tests")
        fetch_source(_client(FILESET), WorkLayout(PurePosixPath(tmp_path)), source, workspace="default")

        context = _sandbox_context(tmp_path, source)
        assert (context / "Dockerfile").read_bytes() == b"FROM tests"
        assert not (context / "tests").exists()
        # Only the subtree was downloaded; a sibling subtree is not on the volume at all.
        assert not (tmp_path / "context/fs-a/env").exists()

    def test_a_whole_fileset_download_serves_its_subtrees(self, tmp_path: Path) -> None:
        """Why the compiler lets a whole-fileset request absorb subtree requests."""
        fetch_source(
            _client(FILESET), WorkLayout(PurePosixPath(tmp_path)), ContextSource(fileset="fs-a"), workspace="default"
        )

        for subtree, content in (("tests", b"FROM tests"), ("env", b"FROM env")):
            context = _sandbox_context(tmp_path, ContextSource(fileset="fs-a", context_path=subtree))
            assert (context / "Dockerfile").read_bytes() == content

    def test_the_context_hash_is_written_beside_the_context(self, tmp_path: Path) -> None:
        source = ContextSource(fileset="fs-a", context_path="tests")
        layout = WorkLayout(PurePosixPath(tmp_path))
        digest = fetch_source(_client(FILESET), layout, source, workspace="default")
        assert Path(layout.context_hash_file(source)).read_text() == digest
        assert not any(p.name.endswith(".nhx-context-hash") for p in _sandbox_context(tmp_path, source).rglob("*"))


class TestHostileListings:
    def test_an_entry_path_that_escapes_the_fileset_is_refused(self, tmp_path: Path) -> None:
        """Entry paths come from a listing a tenant controls."""
        with pytest.raises(ValueError, match="escapes"):
            fetch_source(
                _client({"../other-job/x": b"!"}),
                WorkLayout(PurePosixPath(tmp_path)),
                ContextSource(fileset="fs-a"),
                workspace="default",
            )
