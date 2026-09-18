# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import os
import stat
import tarfile
from unittest.mock import patch

import pytest
from nemo_evaluator.api.task_definitions.harbor import HarborArchiveSource
from nemo_evaluator.harbor import archive

pytest.importorskip("harbor")


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "task"
    for name, text in {
        "task.toml": "",
        "instruction.md": "Do it",
        "environment/Dockerfile": "FROM ubuntu",
        "tests/test.sh": "exit 0",
        ".gitignore": "*.txt",
        "ignored.txt": "included",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "empty").mkdir()
    (root / "tests/test.sh").chmod(0o751)
    return root


@pytest.mark.parametrize("path", ["", "/a", "a/../b", "a//b", "a/", "a\\b", "%2e%2e", "a?b", "a#b", "a\x00b", "a."])
def test_unsafe_ref(path):
    with pytest.raises(ValueError):
        HarborArchiveSource(fileset_ref=f"default/files#{path}", files_hash="a" * 64)


def test_round_trip_and_repeatability(root, tmp_path):
    root.chmod(0o750)
    (root / "empty").chmod(0o1750)
    first = tmp_path / "first.tar.gz"
    digest = archive.pack_task(root, first)
    for path in root.rglob("*"):
        os.utime(path, (123, 123))
    assert archive.pack_task(root, tmp_path / "second.tar.gz") == digest
    dest = tmp_path / "out"
    dest.mkdir()
    restored, native = archive.extract_task(first, dest)
    assert native.task_id == "task"
    for path in [root, *root.rglob("*")]:
        copy = restored / path.relative_to(root)
        assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(copy.stat().st_mode)
        if path.is_file():
            assert path.read_bytes() == copy.read_bytes()
    (root / "tests/test.sh").chmod(0o755)
    assert archive.pack_task(root, tmp_path / "third.tar.gz") != digest


@pytest.mark.parametrize("name", ["/outside", "../../outside", "a/b", "a\\b"])
def test_step_escape_before_native(root, name):
    (root / "task.toml").write_text(f"[[steps]]\nname = {name!r}\n")
    with patch("harbor.models.task.task.Task") as native:
        with pytest.raises(ValueError):
            archive.validate_native_task_inputs(root)
        native.assert_not_called()


@pytest.mark.parametrize(
    "filename,constant",
    [
        ("task.toml", "MAX_CONFIG_BYTES"),
        ("instruction.md", "MAX_INSTRUCTION_BYTES"),
        (".gitignore", "MAX_IGNORE_BYTES"),
    ],
)
def test_metadata_rejected_before_native(root, monkeypatch, filename, constant):
    monkeypatch.setattr(archive, constant, 4)
    (root / filename).write_text("x" * 5)
    with patch("harbor.models.task.task.Task") as native:
        with pytest.raises(ValueError):
            archive.validate_native_task_inputs(root)
        native.assert_not_called()


def test_symlink_rejected(root, tmp_path):
    (root / "link").symlink_to(root / "instruction.md")
    with pytest.raises(ValueError, match="Unsupported"):
        archive.capture_task(root, tmp_path / "out")


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.XGLTYPE])
def test_bad_member_types(tmp_path, kind):
    header = tarfile.TarInfo("task")
    header.type = kind
    path = tmp_path / "bad.tar.gz"
    path.write_bytes(gzip.compress(header.tobuf() + b"\0" * 1024))
    with pytest.raises(ValueError):
        archive.extract_task(path, tmp_path)


def test_pax_allocation_guard(tmp_path):
    header = tarfile.TarInfo("././@PaxHeader")
    header.type = tarfile.XHDTYPE
    header.size = archive.MAX_EXTENSION_BYTES + 1
    path = tmp_path / "bad.tar.gz"
    path.write_bytes(gzip.compress(header.tobuf()))
    with pytest.raises(ValueError, match="metadata"):
        archive.extract_task(path, tmp_path)


def test_truncated_and_trailing(root, tmp_path):
    path = tmp_path / "task.tar.gz"
    archive.pack_task(root, path)
    data = path.read_bytes()
    path.write_bytes(data[:-5])
    (tmp_path / "out").mkdir()
    with pytest.raises((EOFError, ValueError, tarfile.ReadError)):
        archive.extract_task(path, tmp_path / "out")


async def test_validation_cancellation_drains_thread(tmp_path):
    import asyncio
    import threading

    from nemo_evaluator.harbor.archive import run_validation

    started = threading.Event()
    finish = threading.Event()

    def work():
        started.set()
        finish.wait(timeout=5)
        (tmp_path / "finished").write_text("done")

    task = asyncio.create_task(run_validation(work))
    while not started.is_set():
        await asyncio.sleep(0.001)
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (tmp_path / "finished").read_text() == "done"


def test_trailing_nonzero_bytes_are_rejected(root, tmp_path):
    path = tmp_path / "task.tar.gz"
    archive.pack_task(root, path)
    path.write_bytes(gzip.compress(gzip.decompress(path.read_bytes()) + b"extra"))
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="trailing"):
        archive.extract_task(path, tmp_path / "out")


@pytest.mark.parametrize(
    "bad_name",
    ["../outside", "/outside", "task/../outside", "task//double", "other/file", "task/./file", "task/file\\name"],
)
def test_unsafe_archive_member_before_native(tmp_path, bad_name):
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz", format=tarfile.USTAR_FORMAT) as output:
        header = tarfile.TarInfo("task")
        header.type = tarfile.DIRTYPE
        output.addfile(header)
        output.addfile(tarfile.TarInfo(bad_name))
    (tmp_path / "out").mkdir()
    with patch("harbor.models.task.task.Task") as native:
        with pytest.raises(ValueError):
            archive.extract_task(path, tmp_path / "out")
        native.assert_not_called()


def test_pax_chain_bounded_before_next_member(tmp_path):
    header = tarfile.TarInfo("././@PaxHeader")
    header.type = tarfile.XHDTYPE
    data = header.tobuf() * (archive.MAX_EXTENSION_CHAIN + 1)
    path = tmp_path / "bad.tar.gz"
    path.write_bytes(gzip.compress(data))
    with pytest.raises(ValueError, match="metadata"):
        archive.extract_task(path, tmp_path)


def test_duplicate_casefolded_entry(tmp_path):
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz", format=tarfile.USTAR_FORMAT) as output:
        header = tarfile.TarInfo("task")
        header.type = tarfile.DIRTYPE
        output.addfile(header)
        output.addfile(tarfile.TarInfo("task/File"))
        output.addfile(tarfile.TarInfo("task/file"))
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="Duplicate"):
        archive.extract_task(path, tmp_path / "out")


def test_owned_cleanup_handles_untraversable_directories(tmp_path):
    owned = tmp_path / "owned"
    (owned / "nested").mkdir(parents=True)
    (owned / "nested/file").write_text("data")
    (owned / "nested").chmod(0)
    owned.chmod(0)
    archive.remove_owned_tree(owned)
    assert not owned.exists()


@pytest.mark.parametrize(
    ("headers", "decoder"),
    [
        ({"GNU.sparse.size": "1"}, "_proc_gnusparse_00"),
        ({"GNU.sparse.map": "0,1"}, "_proc_gnusparse_01"),
        ({"GNU.sparse.major": "1", "GNU.sparse.minor": "0"}, "_proc_gnusparse_10"),
    ],
)
def test_sparse_pax_rejected_before_decoder(tmp_path, headers, decoder):
    import io

    path = tmp_path / "sparse.tar.gz"
    with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT) as output:
        root = tarfile.TarInfo("task")
        root.type = tarfile.DIRTYPE
        output.addfile(root)
        member = tarfile.TarInfo("task/data")
        member.size = 512
        member.pax_headers = headers
        output.addfile(member, io.BytesIO(b"1\n0\n1\n".ljust(512, b"\0")))
    with patch.object(tarfile.TarInfo, decoder) as native_decoder:
        with pytest.raises(ValueError, match="Unsupported sparse"):
            archive.extract_task(path, tmp_path)
        native_decoder.assert_not_called()
    assert not (tmp_path / "task/data").exists()


@pytest.mark.parametrize("operation", ["capture", "pack"])
def test_unreadable_subtree_fails_inventory(root, tmp_path, operation):
    from pathlib import Path

    data = root / "data"
    data.mkdir()
    (data / "payload").write_text("required task content")
    scandir = os.scandir

    def unreadable(path):
        if Path(path) == data:
            raise PermissionError("Cannot enumerate task data")
        return scandir(path)

    # Simulate the actual scandir error even when tests run as root.
    with patch("os.scandir", side_effect=unreadable):
        with pytest.raises(PermissionError, match="Cannot enumerate"):
            if operation == "capture":
                archive.capture_task(root, tmp_path / "capture")
            else:
                archive.pack_task(root, tmp_path / "archive.tar.gz")
    assert not (tmp_path / "archive.tar.gz").exists()
    assert not (tmp_path / "capture").exists()
