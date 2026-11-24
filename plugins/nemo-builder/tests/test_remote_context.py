# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packing a context for a remote build service: nothing but the context's own files and directories leaves."""

from __future__ import annotations

import io
import os
import tarfile
from collections.abc import Callable
from pathlib import Path

import pytest
from nemo_builder_plugin.run import remote_context
from nemo_builder_plugin.run.remote_context import ContextRefused, pack_context
from nemo_builder_plugin.steps import ContextSource, WorkLayout
from nemo_helix_plugin.files.client import FilesClient
from test_fetch import _client


def _archive(*members: tuple[str, bytes | None, str | None]) -> bytes:
    """A tar of ``(name, content, link)``: a file, a directory when both are None, or a symlink to ``link``."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content, link in members:
            info = tarfile.TarInfo(name)
            if link is not None:
                info.type, info.linkname = tarfile.SYMTYPE, link
                tar.addfile(info)
            elif content is None:
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                info.size, info.mode = len(content), 0o4755
                tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def _pack(tmp_path: Path, source: ContextSource, files: dict[str, bytes]) -> Path:
    tmp_path.mkdir(exist_ok=True)
    return pack_context(_client(files), source, workspace="default", scratch=tmp_path)


def _members(packed: Path) -> dict[str, tarfile.TarInfo]:
    with tarfile.open(packed) as tar:
        return {member.name: member for member in tar}


FILESET = {"app/Dockerfile": b"FROM scratch", "app/src/main.py": b"print()", "other/secret": b"x"}


def test_a_fileset_context_packs_its_own_files_the_same_every_time(tmp_path: Path) -> None:
    source = ContextSource(fileset="default/ctx", context_path="app")
    packed = _pack(tmp_path / "a", source, FILESET)
    members = _members(packed)
    assert sorted(members) == ["Dockerfile", "src", "src/main.py"]
    assert all(m.isreg() or m.isdir() for m in members.values())
    assert all((m.uid, m.gid, m.mtime) == (0, 0, 0) for m in members.values())
    assert packed.read_bytes() == _pack(tmp_path / "b", source, FILESET).read_bytes()


def test_an_archive_context_drops_set_id_bits(tmp_path: Path) -> None:
    archive = _archive(("Dockerfile", b"FROM scratch", None), ("tool", b"#!/bin/sh", None))
    packed = _pack(tmp_path, ContextSource(fileset="default/ctx", archive="c.tar"), {"c.tar": archive})
    assert {name: m.mode for name, m in _members(packed).items()} == {"Dockerfile": 0o755, "tool": 0o755}


@pytest.mark.parametrize(
    "members",
    [
        [("Dockerfile", b"FROM scratch", None), ("inside", None, "Dockerfile")],
        [("Dockerfile", b"FROM scratch", None), ("dangling", None, "missing")],
        [("Dockerfile", b"FROM scratch", None), ("d", None, None), ("linked", None, "d")],
        [("Dockerfile", None, "real"), ("real", b"FROM scratch", None)],
    ],
)
def test_any_link_is_refused(tmp_path: Path, members: list[tuple[str, bytes | None, str | None]]) -> None:
    with pytest.raises(ContextRefused, match="is a link"):
        _pack(tmp_path, ContextSource(fileset="default/ctx", archive="c.tar"), {"c.tar": _archive(*members)})


def test_a_context_path_through_a_link_is_refused(tmp_path: Path) -> None:
    archive = _archive(("real", None, None), ("real/Dockerfile", b"FROM scratch", None), ("app", None, "real"))
    source = ContextSource(fileset="default/ctx", archive="c.tar", context_path="app")
    with pytest.raises(ContextRefused, match="not a directory"):
        _pack(tmp_path, source, {"c.tar": archive})


@pytest.mark.parametrize("files", [{"app/README": b""}, {"app/Dockerfile/x": b""}])
def test_a_context_without_a_root_dockerfile_file_is_refused(tmp_path: Path, files: dict[str, bytes]) -> None:
    with pytest.raises(ContextRefused, match="no regular file Dockerfile"):
        _pack(tmp_path, ContextSource(fileset="default/ctx", context_path="app"), files)


def test_a_special_file_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _with_fetched(monkeypatch, lambda context: os.mkfifo(context / "pipe"))
    with pytest.raises(ContextRefused, match="pipe is not a regular file"):
        _pack(tmp_path, ContextSource(fileset="default/ctx"), {"Dockerfile": b"FROM scratch"})


def test_hard_links_pack_as_two_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _with_fetched(monkeypatch, lambda context: os.link(context / "Dockerfile", context / "copy"))
    members = _members(_pack(tmp_path, ContextSource(fileset="default/ctx"), {"Dockerfile": b"FROM scratch"}))
    assert [(m.type, m.size) for m in members.values()] == [(tarfile.REGTYPE, 12)] * 2


def test_the_caps_are_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = {"Dockerfile": b"FROM scratch", "a": b"12345"}
    monkeypatch.setattr(remote_context, "MAX_PACKED_ENTRIES", 1)
    with pytest.raises(ContextRefused, match="more than 1 entries"):
        _pack(tmp_path / "a", ContextSource(fileset="default/ctx"), files)
    monkeypatch.setattr(remote_context, "MAX_PACKED_ENTRIES", 10)
    monkeypatch.setattr(remote_context, "MAX_PACKED_BYTES", 15)
    with pytest.raises(ContextRefused, match="more than 15 bytes"):
        _pack(tmp_path / "b", ContextSource(fileset="default/ctx"), files)


def _with_fetched(monkeypatch: pytest.MonkeyPatch, change: Callable[[Path], None]) -> None:
    """Change the fetched tree before it is packed, as no fileset can."""
    fetch = remote_context.fetch_source

    def fetched(client: FilesClient, layout: WorkLayout, source: ContextSource, *, workspace: str) -> int:
        count = fetch(client, layout, source, workspace=workspace)
        change(Path(layout.fileset(source.fileset)))
        return count

    monkeypatch.setattr(remote_context, "fetch_source", fetched)
