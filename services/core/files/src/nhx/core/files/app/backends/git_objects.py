# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded-memory git object reads, parsing pack and loose files directly since every dulwich read inflates whole objects."""

import hashlib
import os
import re
import tempfile
import zlib
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from typing import IO, Self

from dulwich.object_store import DiskObjectStore
from dulwich.objects import Blob, Commit, ObjectID, RawObjectID, Tag, Tree, hex_to_filename
from dulwich.pack import OFS_DELTA, REF_DELTA, Pack

_BLOCK = 64 * 1024
_PEEK = 256
_ZLIB_OVERHEAD = 64
_PACK_HEADER_BYTES = 12
_PACK_TRAILER_BYTES = 20
_PACK_VERSIONS = (2, 3)
_RAW_ID_BYTES = 20
_MAX_VARINT_BITS = 64
_MAX_ENTRY_HEADER_BYTES = 32
_MAX_LOOSE_HEADER_BYTES = 64
# Twice git's old --aggressive depth of 250; each level rewrites the object to a temporary file.
_MAX_DELTA_DEPTH = 512
# A copy takes at most 8 bytes and writes at least one; an insert takes one more byte than it writes.
_MAX_DELTA_BYTES_PER_RESULT_BYTE = 8
_DEFAULT_COPY_LENGTH = 0x10000
_HEX_ID = re.compile(rb"[0-9a-f]{40}")
_TYPE_NUMS = {cls.type_name: cls.type_num for cls in (Commit, Tree, Blob, Tag)}


class GitObjectError(Exception):
    """Raised when a pack or loose object is malformed or does not match its id."""


class ObjectTooLargeError(Exception):
    """Raised when a received pack holds an object larger than the caller allows."""


class _Inflater:
    def __init__(self, fd: int, start: int, end: int, first_read: int) -> None:
        self._fd = fd
        self._next = start
        self._end = end
        self._read_size = max(first_read, 1)
        self._zlib = zlib.decompressobj()
        self._input = b""
        self._output = b""
        self._at = 0
        self.consumed = 0

    def _fill(self, want: int) -> bool:
        while len(self._output) - self._at < want:
            if self._zlib.eof:
                return False
            if not self._input:
                length = min(self._read_size, self._end - self._next)
                self._input = os.pread(self._fd, length, self._next) if length > 0 else b""
                if not self._input:
                    raise GitObjectError("A zlib stream runs past the end of its file")
                self._next += len(self._input)
                self._read_size = _BLOCK
            try:
                inflated = self._zlib.decompress(self._input, _BLOCK)
            except zlib.error as exc:
                raise GitObjectError(f"A zlib stream is corrupt: {exc}") from None
            self._input = self._zlib.unconsumed_tail
            if inflated:
                self._output = self._output[self._at :] + inflated
                self._at = 0
        return True

    def byte(self) -> int:
        if not self._fill(1):
            raise GitObjectError("A zlib stream ends before its object does")
        value = self._output[self._at]
        self._at += 1
        self.consumed += 1
        return value

    def read(self, length: int) -> bytes:
        if not self._fill(length):
            raise GitObjectError("A zlib stream ends before its object does")
        return self._take(length)

    def read_some(self, limit: int) -> bytes:
        if not self._fill(1):
            raise GitObjectError("A zlib stream ends before its object does")
        return self._take(limit)

    def _take(self, limit: int) -> bytes:
        piece = self._output[self._at : self._at + limit]
        self._at += len(piece)
        self.consumed += len(piece)
        return piece

    def finish(self, total: int) -> None:
        if self._fill(1) or self.consumed != total:
            raise GitObjectError(f"An object does not inflate to the {total} bytes its header declares")

    @property
    def end_offset(self) -> int:
        return self._next - len(self._input) - len(self._zlib.unused_data)


@dataclass(frozen=True)
class _PackFile:
    fd: int
    end: int


@dataclass(frozen=True)
class _Entry:
    pack: _PackFile
    type_num: int
    size: int
    data_at: int
    base: int | RawObjectID | None

    def inflater(self, first_read: int = _BLOCK) -> _Inflater:
        return _Inflater(self.pack.fd, self.data_at, self.pack.end, min(first_read, self.size + _ZLIB_OVERHEAD))


@dataclass(frozen=True)
class _LooseObject:
    type_num: int
    size: int
    header_bytes: int
    data: _Inflater


class _Cursor:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.pos = 0

    def byte(self) -> int:
        if self.pos >= len(self._data):
            raise GitObjectError("A pack entry header is truncated")
        self.pos += 1
        return self._data[self.pos - 1]

    def take(self, length: int) -> bytes:
        if self.pos + length > len(self._data):
            raise GitObjectError("A pack entry header is truncated")
        self.pos += length
        return self._data[self.pos - length : self.pos]


def _varint(next_byte: Callable[[], int]) -> int:
    value = shift = 0
    while True:
        byte = next_byte()
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value
        shift += 7
        if shift >= _MAX_VARINT_BITS:
            raise GitObjectError("A size in a git object is longer than 64 bits")


def _open_pack(fd: int) -> tuple[int, _PackFile]:
    size = os.fstat(fd).st_size
    header = os.pread(fd, _PACK_HEADER_BYTES, 0)
    if (
        size < _PACK_HEADER_BYTES + _PACK_TRAILER_BYTES
        or header[:4] != b"PACK"
        or int.from_bytes(header[4:8], "big") not in _PACK_VERSIONS
    ):
        raise GitObjectError("A pack file has no valid pack header")
    return int.from_bytes(header[8:12], "big"), _PackFile(fd, size - _PACK_TRAILER_BYTES)


def _entry_at(pack: _PackFile, offset: int) -> _Entry:
    if not _PACK_HEADER_BYTES <= offset < pack.end:
        raise GitObjectError(f"A pack entry offset {offset} is outside its pack")
    cursor = _Cursor(os.pread(pack.fd, min(_MAX_ENTRY_HEADER_BYTES, pack.end - offset), offset))
    first = cursor.byte()
    type_num = (first >> 4) & 0x07
    size = first & 0x0F
    if first & 0x80:
        size |= _varint(cursor.byte) << 4
    base: int | RawObjectID | None = None
    if type_num == OFS_DELTA:
        byte = cursor.byte()
        distance = byte & 0x7F
        while byte & 0x80:
            byte = cursor.byte()
            distance = ((distance + 1) << 7) | (byte & 0x7F)
        base = offset - distance
        # Bases come earlier in the pack, so a chain of offsets always ends.
        if not _PACK_HEADER_BYTES <= base < offset:
            raise GitObjectError(f"A delta at {offset} points at {base}, which is not an earlier pack entry")
    elif type_num == REF_DELTA:
        base = RawObjectID(cursor.take(_RAW_ID_BYTES))
    elif type_num not in _TYPE_NUMS.values():
        raise GitObjectError(f"A pack entry at {offset} has unknown type {type_num}")
    return _Entry(pack, type_num, size, offset + cursor.pos, base)


def _loose_object(fd: int) -> _LooseObject:
    data = _Inflater(fd, 0, os.fstat(fd).st_size, _PEEK)
    header = bytearray()
    while (byte := data.byte()) != 0:
        if len(header) == _MAX_LOOSE_HEADER_BYTES:
            raise GitObjectError("A loose object's header is too long")
        header.append(byte)
    type_name, _, size = bytes(header).partition(b" ")
    if type_name not in _TYPE_NUMS or not size.isdigit():
        raise GitObjectError("A loose object's header is malformed")
    return _LooseObject(_TYPE_NUMS[type_name], int(size), len(header) + 1, data)


def _delta_header(entry: _Entry, data: _Inflater) -> tuple[int, int]:
    source_size = _varint(data.byte)
    result_size = _varint(data.byte)
    if entry.size > data.consumed + _MAX_DELTA_BYTES_PER_RESULT_BYTE * result_size:
        raise GitObjectError(f"A delta of {entry.size} bytes is too long to produce {result_size} bytes")
    return source_size, result_size


class _ObjectFiles:
    def __init__(self, store: DiskObjectStore) -> None:
        self._store = store
        self._packs = store.packs
        self._pack_files: dict[str, _PackFile] = {}
        self._fds: list[int] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        for fd in self._fds:
            os.close(fd)

    def _open(self, path: str) -> int:
        fd = os.open(path, os.O_RDONLY)
        self._fds.append(fd)
        return fd

    def _pack_file(self, pack: Pack) -> _PackFile:
        try:
            path = os.fspath(pack.data.path)
        except AssertionError as exc:
            # dulwich asserts that a pack's index and data count the same objects.
            raise GitObjectError(f"A pack's index does not match its data: {exc}") from None
        if path not in self._pack_files:
            self._pack_files[path] = _open_pack(self._open(path))[1]
        return self._pack_files[path]

    def find(self, sha: ObjectID | RawObjectID) -> _Entry | _LooseObject:
        for pack in self._packs:
            try:
                offset = pack.index.object_offset(sha)
            except KeyError:
                continue
            return _entry_at(self._pack_file(pack), offset)
        hex_sha = sha.decode() if len(sha) == 2 * _RAW_ID_BYTES else sha.hex()
        try:
            fd = self._open(hex_to_filename(os.fspath(self._store.path), hex_sha))
        except FileNotFoundError:
            raise KeyError(hex_sha) from None
        return _loose_object(fd)

    def size_of(self, sha: ObjectID) -> int:
        found = self.find(sha)
        if isinstance(found, _LooseObject):
            # Listings size every file, so a loose object's descriptor cannot wait for __exit__.
            os.close(self._fds.pop())
            return found.size
        if found.base is None:
            return found.size
        return _delta_header(found, found.inflater(_PEEK))[1]


def _require_hex_id(sha: ObjectID) -> None:
    if not _HEX_ID.fullmatch(sha):
        raise ValueError(f"{sha!r} is not a 40-character lowercase hex object id")


def object_size(store: DiskObjectStore, sha: ObjectID) -> int:
    """Return an object's uncompressed size from its headers, without inflating its content."""
    _require_hex_id(sha)
    with _ObjectFiles(store) as files:
        return files.size_of(sha)


def object_sizes(store: DiskObjectStore, shas: Iterable[ObjectID]) -> Iterator[int]:
    """Yield each object's uncompressed size, as object_size does, opening each pack once for all of them."""
    with _ObjectFiles(store) as files:
        for sha in shas:
            _require_hex_id(sha)
            yield files.size_of(sha)


def _inflated(data: _Inflater, size: int, total: int) -> Iterator[bytes]:
    remaining = size
    while remaining:
        piece = data.read_some(min(remaining, _BLOCK))
        remaining -= len(piece)
        yield piece
    data.finish(total)


class _Delta:
    def __init__(self, entry: _Entry) -> None:
        self._size = entry.size
        self._data = entry.inflater()
        self.source_size, self.result_size = _delta_header(entry, self._data)

    def _field(self, flags: int, length: int) -> int:
        value = 0
        for index in range(length):
            if flags & (1 << index):
                value |= self._data.byte() << (8 * index)
        return value

    def apply(self, source_fd: int, source_size: int) -> Iterator[bytes]:
        if source_size != self.source_size:
            raise GitObjectError(f"A delta expects a base of {self.source_size} bytes, not {source_size}")
        remaining = self.result_size
        out = bytearray()
        while remaining:
            command = self._data.byte()
            if command & 0x80:
                offset = self._field(command, 4)
                length = self._field(command >> 4, 3) or _DEFAULT_COPY_LENGTH
                if length > remaining or offset + length > source_size:
                    raise GitObjectError(
                        f"A delta copies {length} bytes at {offset} from a {source_size} byte base "
                        f"with {remaining} bytes left to produce"
                    )
                remaining -= length
                while length:
                    piece = os.pread(source_fd, min(length, _BLOCK), offset)
                    if not piece:
                        raise GitObjectError("A delta's base ended while it was being read")
                    out += piece
                    offset += len(piece)
                    length -= len(piece)
                    if len(out) >= _BLOCK:
                        yield bytes(out)
                        out.clear()
            elif command:
                if command > remaining:
                    raise GitObjectError(f"A delta inserts {command} bytes with {remaining} bytes left to produce")
                remaining -= command
                out += self._data.read(command)
                if len(out) >= _BLOCK:
                    yield bytes(out)
                    out.clear()
            else:
                raise GitObjectError("A delta uses the reserved instruction 0")
        self._data.finish(self._size)
        if out:
            yield bytes(out)


def _delta_chain(files: _ObjectFiles, sha: ObjectID) -> tuple[list[_Entry], _Entry | _LooseObject]:
    found = files.find(sha)
    deltas: list[_Entry] = []
    while isinstance(found, _Entry):
        base = found.base
        if base is None:
            break
        if len(deltas) == _MAX_DELTA_DEPTH:
            raise GitObjectError(f"{sha.decode()} has a delta chain deeper than {_MAX_DELTA_DEPTH}")
        deltas.append(found)
        if isinstance(base, int):
            found = _entry_at(found.pack, base)
            continue
        try:
            found = files.find(base)
        except KeyError:
            raise GitObjectError(f"The delta base {base.hex()} of {sha.decode()} is missing") from None
    return deltas, found


def _write(target: IO[bytes], pieces: Iterator[bytes]) -> None:
    for piece in pieces:
        target.write(piece)
    target.flush()


def _patched(
    top: _Delta,
    lower: list[_Entry],
    base: Iterator[bytes],
    base_size: int,
    temp_dir: str | os.PathLike[str],
    temp_files: ExitStack,
) -> Iterator[bytes]:
    source: IO[bytes] = temp_files.enter_context(tempfile.TemporaryFile(dir=temp_dir))
    _write(source, base)
    source_size = base_size
    spare: IO[bytes] | None = None
    for entry in reversed(lower):
        delta = _Delta(entry)
        if spare is None:
            spare = temp_files.enter_context(tempfile.TemporaryFile(dir=temp_dir))
        spare.seek(0)
        spare.truncate()
        _write(spare, delta.apply(source.fileno(), source_size))
        source, spare = spare, source
        source_size = delta.result_size
    yield from top.apply(source.fileno(), source_size)


def _rechunked(pieces: Iterator[bytes], chunk_size: int) -> Iterator[bytes]:
    pending = bytearray()
    for piece in pieces:
        pending += piece
        while len(pending) >= chunk_size:
            yield bytes(pending[:chunk_size])
            del pending[:chunk_size]
    if pending:
        yield bytes(pending)


def iter_blob(
    store: DiskObjectStore, sha: ObjectID, temp_dir: str | os.PathLike[str], chunk_size: int = _BLOCK
) -> Generator[bytes]:
    """Yield a blob's content in *chunk_size* pieces, holding the last back until its SHA-1 checks out."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, not {chunk_size}")
    _require_hex_id(sha)
    with _ObjectFiles(store) as files, ExitStack() as temp_files:
        deltas, base = _delta_chain(files, sha)
        if base.type_num != Blob.type_num:
            raise GitObjectError(f"{sha.decode()} is not a blob")
        if isinstance(base, _LooseObject):
            pieces = _inflated(base.data, base.size, base.header_bytes + base.size)
        else:
            pieces = _inflated(base.inflater(), base.size, base.size)
        size = base.size
        if deltas:
            top = _Delta(deltas[0])
            pieces = _patched(top, deltas[1:], pieces, size, temp_dir, temp_files)
            size = top.result_size

        digest = hashlib.sha1(b"blob %d\x00" % size)
        held = b""
        for chunk in _rechunked(pieces, chunk_size):
            if held:
                yield held
            digest.update(chunk)
            held = chunk
        if digest.hexdigest().encode() != sha:
            raise GitObjectError(f"The content stored for {sha.decode()} hashes to {digest.hexdigest()}")
        if held:
            yield held


def check_pack_objects(fd: int, max_object_bytes: int) -> None:
    """Raise ObjectTooLargeError if the pack at *fd* holds an object, or a delta result, over *max_object_bytes*."""
    if os.fstat(fd).st_size == 0:
        return
    count, pack = _open_pack(fd)
    offset = _PACK_HEADER_BYTES
    depths: dict[int, int] = {}
    for _ in range(count):
        entry = _entry_at(pack, offset)
        if isinstance(entry.base, int):
            if entry.base not in depths:
                raise GitObjectError(f"A delta at {offset} points into the middle of an earlier pack entry")
            depths[offset] = depths[entry.base] + 1
            # Refused here, so reading the pack can never rewrite one object through more levels than iter_blob allows.
            if depths[offset] > _MAX_DELTA_DEPTH:
                raise GitObjectError(f"A delta chain in the pack is deeper than {_MAX_DELTA_DEPTH}")
        else:
            depths[offset] = 0
        data = entry.inflater()
        size = entry.size if entry.base is None else _delta_header(entry, data)[1]
        if size > max_object_bytes:
            raise ObjectTooLargeError(
                f"The pack holds an object of {size} bytes, over the {max_object_bytes} byte limit"
            )
        while data.consumed < entry.size:
            data.read_some(min(entry.size - data.consumed, _BLOCK))
        data.finish(entry.size)
        offset = data.end_offset
    if offset != pack.end:
        raise GitObjectError("A pack has data after its last object")
