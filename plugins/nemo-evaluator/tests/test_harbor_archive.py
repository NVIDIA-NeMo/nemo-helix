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
from nemo_evaluator_sdk.agent_eval.runtimes import harbor_archive

pytest.importorskip("harbor")


@pytest.fixture
def root(tmp_path):
    """Create a task with ignored files, an empty directory, and executable permissions for archive round trips."""
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
    """Verify archive bytes ignore timestamps while round trips preserve contents and permission bits."""
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
            harbor_archive.validate_native_task_inputs(root)
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
    monkeypatch.setattr(harbor_archive, constant, 4)
    (root / filename).write_text("x" * 5)
    with patch("harbor.models.task.task.Task") as native:
        with pytest.raises(ValueError):
            harbor_archive.validate_native_task_inputs(root)
        native.assert_not_called()


def test_internal_symlinks_round_trip(root, tmp_path):
    """Task-internal links survive capture, packing, and extraction as links, including PAX ``linkpath`` targets."""
    deep = root / "tests" / ("d" * 60) / ("e" * 60)
    deep.mkdir(parents=True)
    (deep / "input.txt").write_text("x")
    links = {
        "tests/run.sh": "test.sh",
        "tests/fixtures": "d" * 60,
        "tests/long": f"{'d' * 60}/{'e' * 60}/input.txt",
    }
    assert len(links["tests/long"]) > 100
    for name, target in links.items():
        (root / name).symlink_to(target)
    (tmp_path / "capture").mkdir()
    snapshot = harbor_archive.capture_task(root, tmp_path / "capture")
    packed = tmp_path / "task.tar.gz"
    archive.pack_task(snapshot, packed)
    (tmp_path / "out").mkdir()
    restored, _ = archive.extract_task(packed, tmp_path / "out")
    for name, target in links.items():
        assert (restored / name).is_symlink()
        assert os.readlink(restored / name) == target
    assert (restored / "tests/long").read_text() == "x"


@pytest.mark.parametrize("target", ["../outside", "/etc/hosts"])
def test_escaping_symlink_rejected(root, tmp_path, target):
    (tmp_path / "outside").write_text("secret")
    (root / "link").symlink_to(target)
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="Symlink escapes"):
        harbor_archive.capture_task(root, tmp_path / "out")


@pytest.mark.parametrize(
    ("links", "error"),
    [
        ([("task/l", "../../x")], "Symlink escapes"),
        ([("task/l", "/etc/passwd")], "Symlink escapes"),
        ([("task/b", "a/.."), ("task/a", ".")], "Symlink escapes"),
        ([("task/l", "../task/x")], "Symlink escapes"),
        ([("task/a", "b"), ("task/b", "a")], "Symlink escapes"),
        ([("task/d", "missing")], "Symlink escapes"),
        ([("task/a", "OUTSIDE"), ("task/a/x", None)], "Missing parent"),
    ],
)
def test_crafted_symlinks_rejected_before_native(tmp_path, links, error):
    """Escaping, order-dependent, cyclic, dangling, and write-through links never leave or touch the host tree."""
    outside = tmp_path / "outside"
    outside.mkdir()
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT) as output:
        header = tarfile.TarInfo("task")
        header.type = tarfile.DIRTYPE
        output.addfile(header)
        output.addfile(tarfile.TarInfo("task/x"))
        for name, target in links:
            header = tarfile.TarInfo(name)
            if target is not None:
                header.type = tarfile.SYMTYPE
                header.linkname = str(outside) if target == "OUTSIDE" else target
            output.addfile(header)
    (tmp_path / "out").mkdir()
    with patch("harbor.models.task.task.Task") as native:
        with pytest.raises(ValueError, match=error):
            archive.extract_task(path, tmp_path / "out")
        native.assert_not_called()
    assert not any(outside.iterdir())
    assert {child.name for child in tmp_path.iterdir()} == {"outside", "bad.tar.gz", "out"}


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


async def test_archive_operation_cancellation_drains_thread(tmp_path):
    """Cancel an archive operation and verify cancellation waits for its worker thread to finish."""
    import asyncio
    import threading

    from nemo_evaluator.harbor.archive import run_blocking_archive_operation

    started = threading.Event()
    finish = threading.Event()

    def work():
        started.set()
        finish.wait(timeout=5)
        (tmp_path / "finished").write_text("done")

    task = asyncio.create_task(run_blocking_archive_operation(work))
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
    harbor_archive.remove_owned_tree(owned)
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
    """Ensure sparse PAX headers are rejected before native sparse decoding or file extraction."""
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
    """Simulate unreadable task content and verify capture and packing fail before creating partial output."""
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
                harbor_archive.capture_task(root, tmp_path / "capture")
            else:
                archive.pack_task(root, tmp_path / "archive.tar.gz")
    assert not (tmp_path / "archive.tar.gz").exists()
    assert not (tmp_path / "capture").exists()
