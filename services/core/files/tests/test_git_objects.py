# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for bounded-memory git object reads, against packs git writes and packs built by hand."""

import hashlib
import os
import random
import subprocess
import tempfile
import zlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO, Any

import pytest
from dulwich.client import SubprocessGitClient
from dulwich.object_store import DiskObjectStore
from dulwich.objects import Blob, ObjectID
from dulwich.pack import OFS_DELTA, REF_DELTA, write_pack_index_v2
from dulwich.repo import Repo
from nhx.core.files.app.backends.git_objects import (
    GitObjectError,
    ObjectTooLargeError,
    check_pack_objects,
    iter_blob,
    object_size,
)

_GIT_ENV = {
    **{name: value for name, value in os.environ.items() if not name.startswith("GIT_")},
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}
_BLOB = Blob.type_num


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        env=_GIT_ENV,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _commit_history(work: Path) -> None:
    _git(work.parent, "init", "-q", "-b", "main", str(work))
    rng = random.Random(2617)
    binary = bytearray(rng.randbytes(100_000))
    lines = [f"line {number}\n" for number in range(2_000)]
    for step in range(8):
        for _ in range(4):
            at = rng.randrange(len(binary) - 16)
            binary[at : at + 16] = rng.randbytes(16)
        binary += rng.randbytes(500)
        lines.insert(rng.randrange(len(lines)), f"edit {step}\n")
        (work / "binary.bin").write_bytes(bytes(binary))
        (work / "notes.txt").write_text("".join(lines))
        (work / f"small-{step}.txt").write_text(f"small {step}\n")
        _git(work, "add", "-A")
        _git(work, "commit", "-q", "-m", f"step {step}")
    (work / "empty").write_bytes(b"")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "empty file")


def _ofs_store(source: Path, root: Path) -> Path:
    repo = root / "ofs.git"
    _git(root, "clone", "-q", "--bare", "--no-local", str(source), str(repo))
    _git(repo, "repack", "-adfq", "--window=50", "--depth=50")
    loose = root / "loose.txt"
    loose.write_bytes(b"a loose object\n" * 100)
    _git(repo, "hash-object", "-w", str(loose))
    return repo / "objects"


def _ref_store(source: Path, root: Path) -> Path:
    repo = root / "ref.git"
    _git(root, "clone", "-q", "--bare", "--no-local", str(source), str(repo))
    _git(repo, "-c", "repack.useDeltaBaseOffset=false", "repack", "-adfq", "--window=50", "--depth=50")
    return repo / "objects"


def _fetched_store(source: Path, root: Path) -> Path:
    repo = root / "fetched.git"
    with Repo.init_bare(str(repo), mkdir=True) as local:
        SubprocessGitClient().fetch(str(source), local)
    return repo / "objects"


_STORE_BUILDERS: dict[str, Callable[[Path, Path], Path]] = {
    "ofs": _ofs_store,
    "ref": _ref_store,
    "fetched": _fetched_store,
}


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, DiskObjectStore]]:
    root = tmp_path_factory.mktemp("git-objects")
    source = root / "source"
    _commit_history(source)
    opened = {name: DiskObjectStore(str(build(source, root))) for name, build in _STORE_BUILDERS.items()}
    yield opened
    for store in opened.values():
        store.close()


@pytest.fixture(params=sorted(_STORE_BUILDERS))
def store(request: pytest.FixtureRequest, stores: dict[str, DiskObjectStore]) -> DiskObjectStore:
    return stores[request.param]


def _pack_type_nums(store: DiskObjectStore) -> set[int]:
    return {unpacked.pack_type_num for pack in store.packs for unpacked in pack.data.iter_unpacked()}


def _blobs(store: DiskObjectStore) -> list[tuple[ObjectID, bytes]]:
    blobs = []
    for sha in store:
        type_num, body = store.get_raw(sha)
        if type_num == _BLOB:
            blobs.append((sha, body))
    return blobs


def _deepest_blob(store: DiskObjectStore) -> tuple[ObjectID, int]:
    depths: dict[bytes, int] = {}
    for pack in store.packs:
        index = os.fspath(pack.data.path).removesuffix(".pack") + ".idx"
        for line in _git(Path(store.path), "verify-pack", "-v", index).splitlines():
            fields = line.split()
            if len(fields) == 7 and fields[1] == "blob":
                depths[fields[0].encode()] = int(fields[5])
    deepest = max(depths, key=depths.__getitem__)
    return ObjectID(deepest), depths[deepest]


def test_git_stores_hold_every_storage_form(stores: dict[str, DiskObjectStore]) -> None:
    ofs, ref = stores["ofs"], stores["ref"]

    assert {_BLOB, OFS_DELTA} <= _pack_type_nums(ofs)
    assert REF_DELTA not in _pack_type_nums(ofs)
    assert {_BLOB, REF_DELTA} <= _pack_type_nums(ref)
    assert list(Path(ofs.path).glob("[0-9a-f][0-9a-f]/*"))
    assert _deepest_blob(ofs)[1] >= 3
    assert max(len(body) for _sha, body in _blobs(ofs)) > 64 * 1024


@pytest.mark.parametrize("chunk_size", [1, 7, 64 * 1024])
def test_streamed_blobs_match_dulwich(store: DiskObjectStore, tmp_path: Path, chunk_size: int) -> None:
    for sha, body in _blobs(store):
        chunks = list(iter_blob(store, sha, tmp_path, chunk_size))

        assert b"".join(chunks) == body
        assert all(len(chunk) == chunk_size for chunk in chunks[:-1])
        assert all(0 < len(chunk) <= chunk_size for chunk in chunks)


def test_sizes_match_dulwich_for_every_object(store: DiskObjectStore) -> None:
    for sha in store:
        assert object_size(store, sha) == len(store.get_raw(sha)[1])


def test_missing_object_raises_key_error(store: DiskObjectStore, tmp_path: Path) -> None:
    missing = ObjectID(b"0" * 40)

    with pytest.raises(KeyError):
        object_size(store, missing)
    with pytest.raises(KeyError):
        next(iter_blob(store, missing, tmp_path))


def test_non_blob_is_rejected(store: DiskObjectStore, tmp_path: Path) -> None:
    tree = next(sha for sha in store if store.get_raw(sha)[0] != _BLOB)

    with pytest.raises(GitObjectError, match="not a blob"):
        list(iter_blob(store, tree, tmp_path))


def test_malformed_id_is_a_value_error(store: DiskObjectStore, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        object_size(store, ObjectID(b"../../HEAD"))
    with pytest.raises(ValueError):
        next(iter_blob(store, ObjectID(b"A" * 40), tmp_path))


class _RecordedTemporaryFiles:
    def __init__(self) -> None:
        self.files: list[IO[bytes]] = []
        self.dirs: list[Any] = []
        self._create = tempfile.TemporaryFile

    def __call__(self, *args: Any, **kwargs: Any) -> IO[bytes]:
        created = self._create(*args, **kwargs)
        self.files.append(created)
        self.dirs.append(kwargs.get("dir"))
        return created


def test_closing_a_delta_stream_early_closes_its_temp_files(
    stores: dict[str, DiskObjectStore], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = stores["ofs"]
    deepest, _depth = _deepest_blob(store)
    recorded = _RecordedTemporaryFiles()
    monkeypatch.setattr(tempfile, "TemporaryFile", recorded)
    stream = iter_blob(store, deepest, tmp_path, chunk_size=1)

    next(stream)
    assert len(recorded.files) == 2
    assert recorded.dirs == [tmp_path, tmp_path]
    assert not any(created.closed for created in recorded.files)

    stream.close()
    assert all(created.closed for created in recorded.files)


def test_finished_delta_stream_closes_its_temp_files(
    stores: dict[str, DiskObjectStore], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = stores["ofs"]
    deepest, _depth = _deepest_blob(store)
    recorded = _RecordedTemporaryFiles()
    monkeypatch.setattr(tempfile, "TemporaryFile", recorded)

    list(iter_blob(store, deepest, tmp_path))

    assert len(recorded.files) == 2
    assert all(created.closed for created in recorded.files)


def test_check_pack_objects_caps_git_packs_at_their_largest_object(store: DiskObjectStore) -> None:
    for pack in store.packs:
        largest = max(object_size(store, sha) for sha in pack)
        with open(pack.data.path, "rb") as handle:
            check_pack_objects(handle.fileno(), largest)
            with pytest.raises(ObjectTooLargeError):
                check_pack_objects(handle.fileno(), largest - 1)


def test_check_pack_objects_ignores_an_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pack"
    empty.write_bytes(b"")

    with open(empty, "rb") as handle:
        check_pack_objects(handle.fileno(), 0)


def _entry_header(type_num: int, size: int) -> bytes:
    out = bytearray([(type_num << 4) | (size & 0x0F)])
    size >>= 4
    while size:
        out[-1] |= 0x80
        out.append(size & 0x7F)
        size >>= 7
    return bytes(out)


def _ofs_distance(distance: int) -> bytes:
    out = [distance & 0x7F]
    distance >>= 7
    while distance:
        distance -= 1
        out.insert(0, 0x80 | (distance & 0x7F))
        distance >>= 7
    return bytes(out)


def _delta_varint(value: int) -> bytes:
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _copy(offset: int, length: int) -> bytes:
    return b"\xff" + offset.to_bytes(4, "little") + length.to_bytes(3, "little")


def _insert(data: bytes) -> bytes:
    return bytes([len(data)]) + data


def _delta(source_size: int, result_size: int, *ops: bytes) -> bytes:
    return _delta_varint(source_size) + _delta_varint(result_size) + b"".join(ops)


def _whole(body: bytes, declared: int | None = None) -> bytes:
    return _entry_header(_BLOB, len(body) if declared is None else declared) + zlib.compress(body)


def _ofs(distance: int, delta: bytes) -> bytes:
    return _entry_header(OFS_DELTA, len(delta)) + _ofs_distance(distance) + zlib.compress(delta)


def _self_referencing_ofs() -> bytes:
    delta = _delta(5, 1, _insert(b"x"))
    return _entry_header(OFS_DELTA, len(delta)) + b"\x00" + zlib.compress(delta)


def _ref(base: bytes, delta: bytes) -> bytes:
    return _entry_header(REF_DELTA, len(delta)) + bytes.fromhex(base.decode()) + zlib.compress(delta)


def _blob_id(body: bytes) -> bytes:
    return Blob.from_string(body).id


def _pack_bytes(entries: list[bytes], *, cut: int = 0, count: int | None = None) -> tuple[bytes, list[int]]:
    body = bytearray(b"PACK" + (2).to_bytes(4, "big") + (len(entries) if count is None else count).to_bytes(4, "big"))
    offsets = []
    for entry in entries:
        offsets.append(len(body))
        body += entry
    trailer = hashlib.sha1(body).digest()
    return bytes(body[: len(body) - cut]) + trailer, offsets


def _crafted_store(root: Path, entries: list[tuple[bytes, bytes]], *, cut: int = 0) -> DiskObjectStore:
    objects = root / "objects"
    store = DiskObjectStore.init(str(objects))
    data, offsets = _pack_bytes([entry for _sha, entry in entries], cut=cut)
    (objects / "pack" / "pack-crafted.pack").write_bytes(data)
    index = sorted((bytes.fromhex(sha.decode()), offset, 0) for (sha, _entry), offset in zip(entries, offsets))
    with open(objects / "pack" / "pack-crafted.idx", "wb") as handle:
        write_pack_index_v2(handle, index, data[-20:])
    return store


def _check(root: Path, data: bytes, cap: int) -> None:
    path = root / "check.pack"
    path.write_bytes(data)
    with open(path, "rb") as handle:
        check_pack_objects(handle.fileno(), cap)


def test_check_pack_objects_rejects_a_whole_object_over_the_cap(tmp_path: Path) -> None:
    data, _offsets = _pack_bytes([_whole(random.Random(1).randbytes(1_000))])

    _check(tmp_path, data, 1_000)
    with pytest.raises(ObjectTooLargeError, match="1000 bytes"):
        _check(tmp_path, data, 999)


def test_check_pack_objects_rejects_a_delta_result_over_the_cap(tmp_path: Path) -> None:
    base = _whole(b"0123456789" * 20)
    grows = _ofs(len(base), _delta(200, 1_000, *[_copy(0, 200)] * 5))
    data, _offsets = _pack_bytes([base, grows])
    assert len(grows) < 500

    _check(tmp_path, data, 1_000)
    with pytest.raises(ObjectTooLargeError, match="1000 bytes"):
        _check(tmp_path, data, 500)


def _malformed_packs() -> dict[str, bytes]:
    base = _whole(b"hello")
    noise = random.Random(2).randbytes(1_000)
    return {
        "truncated zlib stream": _pack_bytes([_whole(noise)], cut=500)[0],
        "count past the last entry": _pack_bytes([base], count=2)[0],
        "data after the last entry": _pack_bytes([base, b"junk"], count=1)[0],
        "self-referencing ofs delta": _pack_bytes([_self_referencing_ofs()])[0],
        "ofs delta before the pack": _pack_bytes([_ofs(100, _delta(5, 1, _insert(b"x")))])[0],
        "whole object longer than declared": _pack_bytes([_whole(b"hello", declared=3)])[0],
        "whole object shorter than declared": _pack_bytes([_whole(b"hello", declared=9)])[0],
        "delta longer than its result needs": _pack_bytes(
            [base, _ofs(len(base), _delta(5, 1, _insert(b"x")) + b"\x00" * 40)]
        )[0],
        "unknown type": _pack_bytes([_entry_header(5, 1) + zlib.compress(b"x")])[0],
        "not a pack": b"NOPE" + bytes(40),
    }


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("truncated zlib stream", "runs past the end"),
        ("count past the last entry", "outside its pack"),
        ("data after the last entry", "after its last object"),
        ("self-referencing ofs delta", "points at 12, which is not an earlier"),
        ("ofs delta before the pack", "points at -88"),
        ("whole object longer than declared", "does not inflate to the 3 bytes"),
        ("whole object shorter than declared", "ends before its object"),
        ("delta longer than its result needs", "too long to produce 1 bytes"),
        ("unknown type", "unknown type 5"),
        ("not a pack", "no valid pack header"),
    ],
)
def test_check_pack_objects_rejects_malformed_packs(tmp_path: Path, case: str, reason: str) -> None:
    with pytest.raises(GitObjectError, match=reason):
        _check(tmp_path, _malformed_packs()[case], 10_000)


def test_crafted_delta_streams_and_sizes(tmp_path: Path) -> None:
    base = _whole(b"hello")
    patched = _ofs(len(base), _delta(5, 11, _copy(0, 5), _insert(b" world")))
    wanted = ObjectID(_blob_id(b"hello world"))
    store = _crafted_store(tmp_path, [(_blob_id(b"hello"), base), (wanted, patched)])

    assert list(iter_blob(store, wanted, tmp_path, 4)) == [b"hell", b"o wo", b"rld"]
    assert object_size(store, wanted) == 11
    store.close()


def _broken_objects() -> dict[str, tuple[list[tuple[bytes, bytes]], int]]:
    base = _whole(b"hello")
    hello = _blob_id(b"hello")
    target = _blob_id(b"target")
    other = _blob_id(b"other")
    noise = random.Random(3).randbytes(1_000)

    def patched(delta: bytes) -> list[tuple[bytes, bytes]]:
        return [(hello, base), (target, _ofs(len(base), delta))]

    return {
        "truncated pack": ([(target, _whole(noise))], 500),
        "self-referencing ofs delta": ([(target, _self_referencing_ofs())], 0),
        "copy past the base": (patched(_delta(5, 10, _copy(0, 10))), 0),
        "copy past the result": (patched(_delta(5, 3, _copy(0, 5))), 0),
        "base size mismatch": (patched(_delta(6, 1, _insert(b"x"))), 0),
        "result shorter than declared": (patched(_delta(5, 10, _insert(b"abc"))), 0),
        "result longer than declared": (patched(_delta(5, 2, _insert(b"abc"))), 0),
        "trailing delta instructions": (patched(_delta(5, 1, _insert(b"x"), _insert(b"y"))), 0),
        "reserved delta instruction": (patched(_delta(5, 1, b"\x00")), 0),
        "ref delta cycle": (
            [(other, _ref(target, _delta(1, 1, _insert(b"x")))), (target, _ref(other, _delta(1, 1, _insert(b"x"))))],
            0,
        ),
        "missing ref base": ([(target, _ref(other, _delta(1, 1, _insert(b"x"))))], 0),
        "content that does not match its id": ([(target, _whole(b"hello"))], 0),
    }


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("truncated pack", "runs past the end"),
        ("self-referencing ofs delta", "points at 12, which is not an earlier"),
        ("copy past the base", "copies 10 bytes at 0 from a 5 byte base"),
        ("copy past the result", "with 3 bytes left to produce"),
        ("base size mismatch", "expects a base of 6 bytes"),
        ("result shorter than declared", "ends before its object"),
        ("result longer than declared", "inserts 3 bytes with 2 bytes left"),
        ("trailing delta instructions", "does not inflate to the 6 bytes"),
        ("reserved delta instruction", "reserved instruction 0"),
        ("ref delta cycle", "deeper than 512"),
        ("missing ref base", "is missing"),
        ("content that does not match its id", "hashes to"),
    ],
)
def test_malformed_objects_raise_git_object_error(tmp_path: Path, case: str, reason: str) -> None:
    entries, cut = _broken_objects()[case]
    store = _crafted_store(tmp_path, entries, cut=cut)

    with pytest.raises(GitObjectError, match=reason):
        list(iter_blob(store, ObjectID(_blob_id(b"target")), tmp_path))
    store.close()


def test_mismatched_content_withholds_its_last_chunk(tmp_path: Path) -> None:
    target = ObjectID(_blob_id(b"target"))
    store = _crafted_store(tmp_path, [(target, _whole(b"hello"))])
    received: list[bytes] = []

    with pytest.raises(GitObjectError, match="hashes to"):
        for chunk in iter_blob(store, target, tmp_path, chunk_size=1):
            received.append(chunk)

    assert b"".join(received) == b"hell"
    store.close()


def test_object_size_rejects_a_self_referencing_delta(tmp_path: Path) -> None:
    target = ObjectID(_blob_id(b"target"))
    store = _crafted_store(tmp_path, [(target, _self_referencing_ofs())])

    with pytest.raises(GitObjectError):
        object_size(store, target)
    store.close()
