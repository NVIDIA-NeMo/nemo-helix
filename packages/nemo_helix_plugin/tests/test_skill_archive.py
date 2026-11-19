# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_helix_plugin.skill_archive — the deterministic, bounded skill archive format."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import stat
import tarfile
import unicodedata
from dataclasses import replace
from pathlib import Path
from typing import IO

import pytest
from nemo_helix_plugin import skill_archive
from nemo_helix_plugin.skill_archive import (
    DEFAULT_LIMITS,
    MAX_FRONTMATTER_BYTES,
    SkillArchiveError,
    SkillArchiveHashMismatchError,
    SkillArchiveLimitError,
    SkillArchiveLimits,
    SkillArchivePathError,
    SkillFrontmatterError,
    build_skill_archive,
    extract_skill_archive,
    parse_skill_frontmatter,
    verify_skill_archive,
)

Member = tuple[tarfile.TarInfo, bytes]

SKILL_MD = """---
name: pdf-tools
description: Work with PDF files.
license: Apache-2.0
metadata:
  tags: [pdf, documents]
  author: someone
---
# PDF tools

Body text.
"""


def make_skill(root: Path, skill_md: str = SKILL_MD) -> Path:
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(skill_md)
    (root / "scripts").mkdir()
    (root / "scripts" / "run.sh").write_text("#!/bin/sh\necho hi\n")
    (root / "scripts" / "run.sh").chmod(0o755)
    (root / "reference.md").write_text("reference\n")
    return root


def build(
    source: Path, tmp_path: Path, name: str = "skill.tar", *, limits: SkillArchiveLimits = DEFAULT_LIMITS
) -> tuple[Path, str]:
    archive = tmp_path / name
    info = build_skill_archive(source, archive, limits=limits)
    return archive, info.files_hash


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def member(name: str, *, kind: bytes = tarfile.REGTYPE, mode: int | None = None, data: bytes = b"") -> Member:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.mode = mode if mode is not None else (0o755 if kind == tarfile.DIRTYPE else 0o644)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    if kind == tarfile.REGTYPE:
        info.size = len(data)
    return info, data


def craft(path: Path, members: list[Member]) -> str:
    """Write a hand-made tar (bypassing the builder) and return its sha256."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for info, data in members:
            archive.addfile(info, io.BytesIO(data) if info.type == tarfile.REGTYPE else None)
    path.write_bytes(buffer.getvalue())
    return sha256(path)


def valid_members(root: str = "pdf-tools") -> list[Member]:
    return [
        member(root, kind=tarfile.DIRTYPE),
        member(f"{root}/SKILL.md", data=SKILL_MD.encode()),
    ]


# ---------------------------------------------------------------------------
# Building: determinism and the format contract
# ---------------------------------------------------------------------------


def test_build_is_deterministic_across_copies_and_mtimes(tmp_path: Path) -> None:
    source = make_skill(tmp_path / "src")
    first, first_hash = build(source, tmp_path, "a.tar")
    second, second_hash = build(source, tmp_path, "b.tar")

    copy = tmp_path / "copy"
    shutil.copytree(source, copy)
    for path in [copy, *copy.rglob("*")]:
        os.utime(path, (1_000_000, 1_000_000))
    third, third_hash = build(copy, tmp_path, "c.tar")

    assert first.read_bytes() == second.read_bytes() == third.read_bytes()
    assert first_hash == second_hash == third_hash == sha256(first)


def test_build_normalises_headers_and_orders_members(tmp_path: Path) -> None:
    source = make_skill(tmp_path / "src")
    (source / "a-b").write_text("x")  # '-' sorts before '/', so this lands between dirs and contents
    archive, _ = build(source, tmp_path)

    with tarfile.open(archive) as tar:
        members = tar.getmembers()
    names = [m.name for m in members]
    assert names == sorted(names)
    assert names[0] == "pdf-tools"
    assert set(names) == {
        "pdf-tools",
        "pdf-tools/SKILL.md",
        "pdf-tools/a-b",
        "pdf-tools/reference.md",
        "pdf-tools/scripts",
        "pdf-tools/scripts/run.sh",
    }
    by_name = {m.name: m for m in members}
    assert by_name["pdf-tools/scripts"].mode == 0o755
    assert by_name["pdf-tools/scripts/run.sh"].mode == 0o755
    assert by_name["pdf-tools/reference.md"].mode == 0o644
    for m in members:
        assert (m.uid, m.gid, m.uname, m.gname, m.mtime) == (0, 0, "", "", 0)
        assert m.pax_headers == {}


def test_executable_bit_changes_hash(tmp_path: Path) -> None:
    source = make_skill(tmp_path / "src")
    _, before = build(source, tmp_path, "a.tar")
    (source / "reference.md").chmod(0o744)
    _, after = build(source, tmp_path, "b.tar")
    assert before != after


def test_non_execute_mode_bits_do_not_change_hash(tmp_path: Path) -> None:
    source = make_skill(tmp_path / "src")
    _, before = build(source, tmp_path, "a.tar")
    (source / "reference.md").chmod(0o600)
    _, after = build(source, tmp_path, "b.tar")
    assert before == after


def test_top_level_evals_is_excluded_but_nested_evals_is_kept(tmp_path: Path) -> None:
    source = make_skill(tmp_path / "src")
    _, without = build(source, tmp_path, "a.tar")
    (source / "evals").mkdir()
    (source / "evals" / "evals.json").write_text('{"expected": "secret"}')
    _, with_top_level = build(source, tmp_path, "b.tar")
    assert with_top_level == without

    (source / "scripts" / "evals").mkdir()
    (source / "scripts" / "evals" / "x.txt").write_text("kept")
    archive, _ = build(source, tmp_path, "c.tar")
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert "pdf-tools/scripts/evals/x.txt" in names
    assert not any(name.startswith("pdf-tools/evals") for name in names)


def test_root_is_renamed_to_frontmatter_name(tmp_path: Path) -> None:
    source = make_skill(tmp_path / "some-checkout-dir")
    archive, files_hash = build(source, tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()

    extracted = extract_skill_archive(archive, dest, expected_files_hash=files_hash)

    assert extracted == dest / "pdf-tools"
    assert sorted(p.name for p in dest.iterdir()) == ["pdf-tools"]


def test_build_info_describes_archive(tmp_path: Path) -> None:
    source = make_skill(tmp_path / "src")
    archive = tmp_path / "skill.tar"
    info = build_skill_archive(source, archive)
    assert info.files_hash == sha256(archive)
    assert info.skill_name == "pdf-tools"
    assert info.archive_bytes == archive.stat().st_size
    assert info.entry_count == 5
    assert info.payload_bytes == sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
    assert info.frontmatter.tags == ["pdf", "documents"]


def test_long_and_non_ascii_paths_round_trip(tmp_path: Path) -> None:
    source = make_skill(tmp_path / "src")
    deep = source / ("d" * 60) / ("e" * 60)
    deep.mkdir(parents=True)
    (deep / "résumé.txt").write_text("long")
    archive, files_hash = build(source, tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()

    extracted = extract_skill_archive(archive, dest, expected_files_hash=files_hash)

    assert (extracted / ("d" * 60) / ("e" * 60) / "résumé.txt").read_text() == "long"


def test_build_refuses_existing_archive_path(tmp_path: Path) -> None:
    source = make_skill(tmp_path / "src")
    archive = tmp_path / "skill.tar"
    archive.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        build_skill_archive(source, archive)
    assert archive.read_bytes() == b"keep me"


# ---------------------------------------------------------------------------
# Building: rejected sources
# ---------------------------------------------------------------------------


def test_build_rejects_file_symlink(tmp_path: Path) -> None:
    source = make_skill(tmp_path / "src")
    (source / "link.md").symlink_to(source / "reference.md")
    with pytest.raises(SkillArchivePathError, match="Symlink"):
        build(source, tmp_path)
    assert not (tmp_path / "skill.tar").exists()


def test_build_rejects_directory_symlink(tmp_path: Path) -> None:
    source = make_skill(tmp_path / "src")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("x")
    (source / "scripts" / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(SkillArchivePathError, match="Symlink"):
        build(source, tmp_path)


def test_build_rejects_symlinked_skill_md(tmp_path: Path) -> None:
    source = make_skill(tmp_path / "src")
    real = tmp_path / "real.md"
    real.write_text(SKILL_MD)
    (source / "SKILL.md").unlink()
    (source / "SKILL.md").symlink_to(real)
    with pytest.raises(SkillFrontmatterError, match="regular file"):
        build(source, tmp_path)


def test_build_rejects_fifo(tmp_path: Path) -> None:
    source = make_skill(tmp_path / "src")
    os.mkfifo(source / "pipe")
    with pytest.raises(SkillArchivePathError, match="regular files and directories"):
        build(source, tmp_path)


def test_build_rejects_missing_skill_md(tmp_path: Path) -> None:
    source = make_skill(tmp_path / "src")
    (source / "SKILL.md").unlink()
    with pytest.raises(SkillFrontmatterError, match="missing"):
        build(source, tmp_path)


def test_build_rejects_skill_md_directory(tmp_path: Path) -> None:
    source = make_skill(tmp_path / "src")
    (source / "SKILL.md").unlink()
    (source / "SKILL.md").mkdir()
    with pytest.raises(SkillFrontmatterError, match="regular file"):
        build(source, tmp_path)


def test_build_rejects_names_that_collide_after_nfc(tmp_path: Path) -> None:
    source = make_skill(tmp_path / "src")
    nfc = unicodedata.normalize("NFC", "é.txt")
    nfd = unicodedata.normalize("NFD", "é.txt")
    (source / nfc).write_text("a")
    try:
        with open(source / nfd, "x") as handle:
            handle.write("b")
    except FileExistsError:
        pytest.skip("filesystem is normalisation-insensitive; it cannot hold both names")
    with pytest.raises(SkillArchivePathError, match="collides"):
        build(source, tmp_path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_entries": 4},
        {"max_file_bytes": 10},
        {"max_total_payload_bytes": 100},
        {"max_path_length": 20},
        {"max_archive_bytes": 4096},
    ],
    ids=["entries", "file", "payload", "path", "archive"],
)
def test_build_enforces_limits(tmp_path: Path, overrides: dict[str, int]) -> None:
    source = make_skill(tmp_path / "src")
    with pytest.raises(SkillArchiveLimitError):
        build(source, tmp_path, limits=replace(DEFAULT_LIMITS, **overrides))
    assert not (tmp_path / "skill.tar").exists()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def test_verify_accepts_built_archive(tmp_path: Path) -> None:
    archive, files_hash = build(make_skill(tmp_path / "src"), tmp_path)
    info = verify_skill_archive(archive, expected_files_hash=files_hash)
    assert info.files_hash == files_hash
    assert info.frontmatter.name == "pdf-tools"


def test_verify_rejects_hash_mismatch(tmp_path: Path) -> None:
    archive, _ = build(make_skill(tmp_path / "src"), tmp_path)
    with pytest.raises(SkillArchiveHashMismatchError):
        verify_skill_archive(archive, expected_files_hash="0" * 64)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"max_entries": 4}, SkillArchiveLimitError),
        ({"max_file_bytes": 10}, SkillArchiveLimitError),
        ({"max_total_payload_bytes": 100}, SkillArchiveLimitError),
        ({"max_path_length": 20}, SkillArchiveLimitError),
        ({"max_archive_bytes": 4096}, SkillArchiveLimitError),
    ],
    ids=["entries", "file", "payload", "path", "archive"],
)
def test_verify_enforces_limits(tmp_path: Path, overrides: dict[str, int], error: type[Exception]) -> None:
    archive, files_hash = build(make_skill(tmp_path / "src"), tmp_path)
    with pytest.raises(error):
        verify_skill_archive(archive, expected_files_hash=files_hash, limits=replace(DEFAULT_LIMITS, **overrides))


def test_verify_rejects_symlinked_archive_path(tmp_path: Path) -> None:
    archive, files_hash = build(make_skill(tmp_path / "src"), tmp_path)
    link = tmp_path / "link.tar"
    link.symlink_to(archive)
    with pytest.raises(SkillArchivePathError):
        verify_skill_archive(link, expected_files_hash=files_hash)


def _symlink_member() -> Member:
    info, _ = member("pdf-tools/link", kind=tarfile.SYMTYPE, mode=0o777)
    info.linkname = "../../etc/passwd"
    return info, b""


def _hardlink_member() -> Member:
    info, _ = member("pdf-tools/hard", kind=tarfile.LNKTYPE)
    info.linkname = "pdf-tools/SKILL.md"
    return info, b""


@pytest.mark.parametrize(
    ("extra", "error", "match"),
    [
        ([_symlink_member()], SkillArchivePathError, "regular files and directories"),
        ([_hardlink_member()], SkillArchivePathError, "regular files and directories"),
        ([member("pdf-tools/dev", kind=tarfile.CHRTYPE)], SkillArchivePathError, "regular files and directories"),
        ([member("pdf-tools/fifo", kind=tarfile.FIFOTYPE)], SkillArchivePathError, "regular files and directories"),
        ([member("pdf-tools/../escape", data=b"x")], SkillArchivePathError, "Invalid path segment"),
        ([member("pdf-tools/./x", data=b"x")], SkillArchivePathError, "Invalid path segment"),
        ([member("pdf-tools//x", data=b"x")], SkillArchivePathError, "Invalid path segment"),
        ([member("/etc/passwd", data=b"x")], SkillArchivePathError, "absolute"),
        ([member("pdf-tools/SKILL.md", data=b"dup")], SkillArchivePathError, "duplicated or out of order"),
        ([member("pdf-tools/a", data=b"x")], SkillArchivePathError, "duplicated or out of order"),
        ([member("other/x", data=b"x")], SkillArchivePathError, "out of order|outside the single root"),
        ([member("pdf-tools/x/y", data=b"x")], SkillArchivePathError, "no parent directory"),
        ([member("pdf-tools/x", mode=0o777, data=b"x")], SkillArchivePathError, "non-normalised mode"),
        ([member("pdf-tools/x", mode=0o4755, data=b"x")], SkillArchivePathError, "non-normalised mode"),
        ([member("pdf-tools/evals", kind=tarfile.DIRTYPE)], SkillArchivePathError, "excluded"),
        (
            [member(unicodedata.normalize("NFD", "pdf-tools/é"), data=b"x")],
            SkillArchivePathError,
            "NFC",
        ),
    ],
    ids=[
        "symlink",
        "hardlink",
        "device",
        "fifo",
        "dotdot",
        "dot",
        "empty-segment",
        "absolute",
        "duplicate",
        "unsorted",
        "second-root",
        "missing-parent",
        "world-writable",
        "setuid",
        "evals",
        "not-nfc",
    ],
)
def test_verify_rejects_malicious_members(
    tmp_path: Path, extra: list[Member], error: type[Exception], match: str
) -> None:
    archive = tmp_path / "evil.tar"
    members = valid_members()
    if extra[0][0].name == "pdf-tools/a":
        # Out of order: 'a' after 'b'.
        members.append(member("pdf-tools/b", data=b"x"))
    files_hash = craft(archive, [*members, *extra])
    with pytest.raises(error, match=match):
        verify_skill_archive(archive, expected_files_hash=files_hash)
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(error):
        extract_skill_archive(archive, dest, expected_files_hash=files_hash)
    assert list(dest.iterdir()) == []


def test_verify_rejects_non_normalised_ownership(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar"
    info, data = member("pdf-tools/x", data=b"x")
    info.uid = 1000
    files_hash = craft(archive, [*valid_members(), (info, data)])
    with pytest.raises(SkillArchivePathError, match="ownership"):
        verify_skill_archive(archive, expected_files_hash=files_hash)


def test_verify_rejects_root_that_does_not_match_frontmatter(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar"
    files_hash = craft(archive, valid_members("renamed"))
    with pytest.raises(SkillFrontmatterError, match="does not match"):
        verify_skill_archive(archive, expected_files_hash=files_hash)


def test_verify_rejects_file_as_root(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar"
    files_hash = craft(archive, [member("pdf-tools", data=b"x")])
    with pytest.raises(SkillArchivePathError, match="single root directory"):
        verify_skill_archive(archive, expected_files_hash=files_hash)


def test_verify_rejects_missing_skill_md(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar"
    files_hash = craft(archive, [member("pdf-tools", kind=tarfile.DIRTYPE), member("pdf-tools/x", data=b"x")])
    with pytest.raises(SkillFrontmatterError, match="no regular SKILL.md"):
        verify_skill_archive(archive, expected_files_hash=files_hash)


def test_verify_rejects_trailing_data(tmp_path: Path) -> None:
    archive, _ = build(make_skill(tmp_path / "src"), tmp_path)
    tampered = tmp_path / "tampered.tar"
    tampered.write_bytes(archive.read_bytes() + b"\x01" + b"\x00" * (tarfile.RECORDSIZE - 1))
    with pytest.raises(SkillArchiveError, match="end-of-archive"):
        verify_skill_archive(tampered, expected_files_hash=sha256(tampered))


def test_verify_rejects_truncated_archive(tmp_path: Path) -> None:
    archive, _ = build(make_skill(tmp_path / "src"), tmp_path)
    truncated = tmp_path / "truncated.tar"
    truncated.write_bytes(archive.read_bytes()[:1536])
    with pytest.raises(SkillArchiveError):
        verify_skill_archive(truncated, expected_files_hash=sha256(truncated))


def test_verify_rejects_gzip_and_garbage(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.tar"
    garbage.write_bytes(b"not a tar" * 100)
    with pytest.raises(SkillArchiveError):
        verify_skill_archive(garbage, expected_files_hash=sha256(garbage))


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_extract_round_trips_content_and_modes(tmp_path: Path) -> None:
    source = make_skill(tmp_path / "src")
    archive, files_hash = build(source, tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()

    extracted = extract_skill_archive(archive, dest, expected_files_hash=files_hash)

    assert (extracted / "SKILL.md").read_text() == SKILL_MD
    assert (extracted / "scripts" / "run.sh").read_text() == "#!/bin/sh\necho hi\n"
    assert stat.S_IMODE((extracted / "scripts" / "run.sh").stat().st_mode) == 0o755
    assert stat.S_IMODE((extracted / "reference.md").stat().st_mode) == 0o644
    assert stat.S_IMODE((extracted / "scripts").stat().st_mode) == 0o755
    assert stat.S_IMODE(extracted.stat().st_mode) == 0o755
    # Rebuilding the extracted tree reproduces the same archive.
    _, rebuilt = build(extracted, tmp_path, "rebuilt.tar")
    assert rebuilt == files_hash


def test_extract_rejects_hash_mismatch_without_writing(tmp_path: Path) -> None:
    archive, _ = build(make_skill(tmp_path / "src"), tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(SkillArchiveHashMismatchError):
        extract_skill_archive(archive, dest, expected_files_hash="f" * 64)
    assert list(dest.iterdir()) == []


def test_extract_refuses_existing_skill_dir(tmp_path: Path) -> None:
    archive, files_hash = build(make_skill(tmp_path / "src"), tmp_path)
    dest = tmp_path / "dest"
    (dest / "pdf-tools").mkdir(parents=True)
    (dest / "pdf-tools" / "keep").write_text("mine")
    with pytest.raises(SkillArchivePathError, match="already exists"):
        extract_skill_archive(archive, dest, expected_files_hash=files_hash)
    assert (dest / "pdf-tools" / "keep").read_text() == "mine"


def test_extract_requires_existing_parent(tmp_path: Path) -> None:
    archive, files_hash = build(make_skill(tmp_path / "src"), tmp_path)
    with pytest.raises(SkillArchivePathError, match="not an existing directory"):
        extract_skill_archive(archive, tmp_path / "missing", expected_files_hash=files_hash)


def test_extract_removes_partial_tree_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive, files_hash = build(make_skill(tmp_path / "src"), tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    real_write = skill_archive._write_member
    calls = 0

    def failing_write(fd: int, source: IO[bytes], size: int) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        return real_write(fd, source, size)

    monkeypatch.setattr(skill_archive, "_write_member", failing_write)
    with pytest.raises(OSError, match="disk full"):
        extract_skill_archive(archive, dest, expected_files_hash=files_hash)
    assert calls == 2
    assert list(dest.iterdir()) == []


def test_extract_does_not_write_when_verification_fails(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar"
    # A valid first file followed by an escape: nothing may be written for the valid part either.
    files_hash = craft(archive, [*valid_members(), member("pdf-tools/z/../../../escape", data=b"x")])
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(SkillArchivePathError):
        extract_skill_archive(archive, dest, expected_files_hash=files_hash)
    assert list(dest.iterdir()) == []
    assert not (tmp_path / "escape").exists()


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------


def test_parse_frontmatter_splits_tags_and_extra() -> None:
    frontmatter = parse_skill_frontmatter(SKILL_MD)
    assert frontmatter.name == "pdf-tools"
    assert frontmatter.description == "Work with PDF files."
    assert frontmatter.tags == ["pdf", "documents"]
    assert frontmatter.extra == {"license": "Apache-2.0", "metadata": {"author": "someone"}}


def test_parse_frontmatter_drops_empty_metadata_and_keeps_dates_as_text() -> None:
    frontmatter = parse_skill_frontmatter(
        "\ufeff---\r\nname: ab\r\ndescription: d\r\nupdated: 2026-01-01\r\nmetadata:\r\n  tags: []\r\n---\r\n"
    )
    assert frontmatter.tags == []
    assert frontmatter.extra == {"updated": "2026-01-01"}


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("# no frontmatter\n", "must start"),
        ("---\nname: ab\ndescription: d\n", "not closed"),
        ("---\nname: [unclosed\n---\n", "not valid YAML"),
        ("---\n- a\n- b\n---\n", "mapping"),
        ("---\ndescription: d\n---\n", "'name'"),
        ("---\nname: Not_Valid\ndescription: d\n---\n", "'name'"),
        ("---\nname: a--b\ndescription: d\n---\n", "'name'"),
        (f"---\nname: {'a' * 65}\ndescription: d\n---\n", "'name'"),
        ("---\nname: 12\ndescription: d\n---\n", "'name'"),
        ("---\nname: ab\n---\n", "'description'"),
        ("---\nname: ab\ndescription: '  '\n---\n", "'description'"),
        ("---\nname: ab\ndescription: d\nmetadata:\n  tags: pdf\n---\n", "metadata.tags"),
        ("---\nname: ab\ndescription: d\nmetadata:\n  tags: [1, 2]\n---\n", "metadata.tags"),
        ("---\nname: ab\ndescription: d\nx: &a [1]\ny: *a\n---\n", "aliases"),
        ("---\nname: ab\ndescription: d\nx: .inf\n---\n", "non-finite"),
        ("---\nname: ab\ndescription: d\n1: x\n---\n", "keys must be strings"),
        ("---\nname: ab\ndescription: d\nx: !!binary aGk=\n---\n", "non-JSON"),
        ("---\nname: ab\ndescription: d\nx: !!python/object:os.system {}\n---\n", "not valid YAML"),
        ("---\n" + "x: " + "y" * MAX_FRONTMATTER_BYTES + "\n---\n", "exceeds"),
    ],
    ids=[
        "no-block",
        "unclosed",
        "bad-yaml",
        "not-mapping",
        "missing-name",
        "bad-name",
        "double-hyphen",
        "long-name",
        "non-string-name",
        "missing-description",
        "blank-description",
        "tags-not-list",
        "tags-not-strings",
        "alias",
        "infinity",
        "int-key",
        "binary",
        "python-tag",
        "too-large",
    ],
)
def test_parse_frontmatter_rejects(text: str, match: str) -> None:
    with pytest.raises(SkillFrontmatterError, match=match):
        parse_skill_frontmatter(text)


def test_build_rejects_invalid_frontmatter_and_leaves_nothing(tmp_path: Path) -> None:
    source = make_skill(tmp_path / "src", skill_md="---\nname: Bad Name\ndescription: d\n---\n")
    with pytest.raises(SkillFrontmatterError):
        build(source, tmp_path)
    assert not (tmp_path / "skill.tar").exists()


def test_build_rejects_non_utf8_skill_md(tmp_path: Path) -> None:
    source = make_skill(tmp_path / "src")
    (source / "SKILL.md").write_bytes(b"---\nname: ab\ndescription: \xff\n---\n")
    with pytest.raises(SkillFrontmatterError, match="UTF-8"):
        build(source, tmp_path)


def test_verify_rejects_extra_pax_headers(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar"
    info, data = member("pdf-tools/x", data=b"x")
    info.pax_headers = {"comment": "hidden"}
    files_hash = craft(archive, [*valid_members(), (info, data)])
    with pytest.raises(SkillArchivePathError, match="PAX"):
        verify_skill_archive(archive, expected_files_hash=files_hash)


def test_verify_rejects_global_pax_header(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT, pax_headers={"comment": "g"}) as tar:
        for info, data in valid_members():
            tar.addfile(info, io.BytesIO(data) if info.type == tarfile.REGTYPE else None)
    archive.write_bytes(buffer.getvalue())
    with pytest.raises(SkillArchivePathError, match="regular files and directories"):
        verify_skill_archive(archive, expected_files_hash=sha256(archive))


def test_frontmatter_parse_ignores_a_huge_body() -> None:
    # Millions of short lines after the block must not be split; only the bounded prefix is scanned.
    skill_md = "---\nname: ab\ndescription: d\n---\n" + "x\n" * 5_000_000
    assert parse_skill_frontmatter(skill_md).name == "ab"
