# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Stainless-free transfer helpers in ``filesets.transfer``.

A small in-memory fileset server answers both the sync client and the async
client the filesystem builds from it, so the tests pin the request sequence,
paths, query params, and bodies that each helper produces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from filesets import transfer
from filesets.filesystem.filesystem import FilesetFileSystem
from filesets.transfer import ListFilesResponse
from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_platform_plugin.files.types import CacheStatus, FilesetFileOutput
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

BASE = "http://test"
WORKSPACE = "default"


def _fileset_json(workspace: str, name: str) -> dict:
    return {
        "id": f"id-{name}",
        "name": name,
        "workspace": workspace,
        "description": "",
        "purpose": "generic",
        "storage": {"type": "local", "path": f"/data/{name}"},
        "metadata": {},
        "custom_fields": {},
        "project": "",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }


@dataclass
class FakeFilesServer:
    """In-memory fileset store keyed by ``(workspace, fileset)``; records every request."""

    filesets: dict[tuple[str, str], dict[str, bytes]] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)

    def calls(self) -> list[tuple[str, str]]:
        return [(request.method, request.url.path) for request in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        parts = request.url.path.lstrip("/").split("/")
        # apis/files/v2/workspaces/{ws}/filesets[/{name}[/files | /-/{path}]]
        workspace = parts[4]
        if len(parts) == 6:
            if request.method == "POST":
                name = json.loads(request.content)["name"]
                if (workspace, name) in self.filesets:
                    return httpx.Response(409, json={"detail": "exists"})
                self.filesets[(workspace, name)] = {}
                return httpx.Response(201, json=_fileset_json(workspace, name))
            return httpx.Response(405)
        name = parts[6]
        files = self.filesets.get((workspace, name))
        if files is None:
            return httpx.Response(404, json={"detail": f"Fileset '{name}' not found"})
        if len(parts) == 7:
            return httpx.Response(200, json=_fileset_json(workspace, name))
        if parts[7] == "files":
            prefix = request.url.params.get("path", "")
            data = [
                {
                    "file_ref": f"{workspace}/{name}#{path}",
                    "file_url": f"/apis/files/v2/workspaces/{workspace}/filesets/{name}/-/{path}",
                    "path": path,
                    "size": len(content),
                }
                for path, content in sorted(files.items())
                if path.startswith(prefix)
            ]
            return httpx.Response(200, json={"data": data})
        path = "/".join(parts[8:])
        file_json = {
            "file_ref": f"{workspace}/{name}#{path}",
            "file_url": request.url.path,
            "path": path,
            "size": len(files.get(path, b"")),
        }
        if request.method == "PUT":
            files[path] = request.content
            return httpx.Response(200, json={**file_json, "size": len(request.content)})
        if path not in files:
            return httpx.Response(404, json={"detail": "File not found"})
        if request.method == "GET":
            return httpx.Response(200, content=files[path], headers={"content-length": str(len(files[path]))})
        if request.method == "DELETE":
            del files[path]
            return httpx.Response(200, json=file_json)
        return httpx.Response(405)

    async def asgi(self, scope: Scope, receive: Receive, send: Send) -> None:
        incoming = Request(scope, receive)
        body = await incoming.body()
        response = self(httpx.Request(incoming.method, str(incoming.url), headers=incoming.headers.raw, content=body))
        await Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))(
            scope, receive, send
        )


class _HttpClient(httpx.Client):
    def __init__(self, server: FakeFilesServer) -> None:
        super().__init__(transport=httpx.MockTransport(server), base_url=BASE)
        self._server = server

    @property
    def asgi_app(self):
        return self._server.asgi


@pytest.fixture
def server() -> FakeFilesServer:
    return FakeFilesServer()


@pytest.fixture
def client(server: FakeFilesServer) -> FilesClient:
    return FilesClient(base_url=BASE, workspace=WORKSPACE, http_client=_HttpClient(server))


@pytest.fixture
def async_client(server: FakeFilesServer) -> AsyncFilesClient:
    return AsyncFilesClient(
        base_url=BASE, workspace=WORKSPACE, http_client=httpx.AsyncClient(transport=httpx.ASGITransport(server.asgi))
    )


def _put(request: httpx.Request) -> tuple[str, str]:
    return request.method, request.url.path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filepath", "pattern", "expected"),
    [
        ("train.json", "*.json", True),
        ("subdir/nested.json", "*.json", False),
        ("subdir/nested.json", "subdir/*.json", True),
        ("subdir/nested.json", "*/*.json", True),
        ("a/b/c.txt", "b/*.txt", True),
        ("a/b/c.txt", "*.md", False),
    ],
)
def test_matches_glob(filepath: str, pattern: str, expected: bool) -> None:
    assert transfer.matches_glob(filepath, pattern) is expected


def test_generate_fileset_name_is_unique_and_prefixed() -> None:
    names = {transfer.generate_fileset_name() for _ in range(5)}
    assert len(names) == 5
    assert all(name.startswith("fileset-") and len(name) == len("fileset-") + 8 for name in names)


def _file(path: str, cache_status: CacheStatus | None) -> FilesetFileOutput:
    return FilesetFileOutput(file_ref=f"ws/fs#{path}", file_url="/x", path=path, size=1, cache_status=cache_status)


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ([], None),
        ([None, None], None),
        ([CacheStatus.CACHED, CacheStatus.CACHING], CacheStatus.CACHING),
        ([CacheStatus.CACHED, CacheStatus.NOT_CACHED], CacheStatus.NOT_CACHED),
        ([CacheStatus.CACHED, CacheStatus.CACHED], CacheStatus.CACHED),
        ([CacheStatus.NOT_CACHEABLE, CacheStatus.NOT_CACHEABLE], CacheStatus.NOT_CACHEABLE),
        ([CacheStatus.CACHED, CacheStatus.NOT_CACHEABLE], CacheStatus.CACHED),
    ],
)
def test_list_files_response_cache_status(statuses: list[CacheStatus | None], expected: CacheStatus | None) -> None:
    response = ListFilesResponse(data=[_file(f"f{i}", status) for i, status in enumerate(statuses)])
    assert response.cache_status == expected


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------


def test_list_files_root(server: FakeFilesServer, client: FilesClient) -> None:
    server.filesets[(WORKSPACE, "fs")] = {"a.txt": b"a", "d/b.txt": b"bb"}

    response = transfer.list_files(client, fileset="fs")

    assert [(f.path, f.size) for f in response.data] == [("a.txt", 1), ("d/b.txt", 2)]
    assert server.calls() == [("GET", "/apis/files/v2/workspaces/default/filesets/fs/files")]
    assert dict(server.requests[0].url.params) == {}


def test_list_files_prefix_is_sent_as_path_param(server: FakeFilesServer, client: FilesClient) -> None:
    server.filesets[(WORKSPACE, "fs")] = {"a.txt": b"a", "d/b.txt": b"bb"}

    response = transfer.list_files(client, fileset="fs", remote_path="d/", include_cache_status=True)

    assert [f.path for f in response.data] == ["d/b.txt"]
    assert dict(server.requests[0].url.params) == {"path": "d/", "include_cache_status": "true"}


def test_list_files_glob_filters_client_side(server: FakeFilesServer, client: FilesClient) -> None:
    server.filesets[(WORKSPACE, "fs")] = {"a.json": b"a", "b.txt": b"b", "d/c.json": b"c"}

    response = transfer.list_files(client, fileset="fs", remote_path="*.json")

    assert [f.path for f in response.data] == ["a.json"]
    assert dict(server.requests[0].url.params) == {}


def test_list_files_parses_fileset_ref_in_remote_path(server: FakeFilesServer, client: FilesClient) -> None:
    server.filesets[("other", "fs")] = {"d/x.txt": b"x"}

    response = transfer.list_files(client, remote_path="other/fs#d/")

    assert [f.path for f in response.data] == ["d/x.txt"]
    assert server.calls() == [("GET", "/apis/files/v2/workspaces/other/filesets/fs/files")]


def test_list_files_requires_fileset(client: FilesClient) -> None:
    with pytest.raises(ValueError, match="Fileset must be specified"):
        transfer.list_files(client, remote_path="d/")


def test_list_files_missing_fileset_raises_not_found(client: FilesClient) -> None:
    with pytest.raises(NotFoundError):
        transfer.list_files(client, fileset="nope")


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------


def test_upload_single_file(server: FakeFilesServer, client: FilesClient, tmp_path: Path) -> None:
    server.filesets[(WORKSPACE, "fs")] = {}
    local = tmp_path / "a.txt"
    local.write_bytes(b"hello")

    result = transfer.upload(client, local_path=str(local), fileset="fs", remote_path="data/")

    assert result.name == "fs"
    assert server.filesets[(WORKSPACE, "fs")] == {"data/a.txt": b"hello"}
    put = next(r for r in server.requests if r.method == "PUT")
    assert put.url.path == "/apis/files/v2/workspaces/default/filesets/fs/-/data/a.txt"
    assert put.headers["content-length"] == "5"
    assert server.calls()[-1] == ("GET", "/apis/files/v2/workspaces/default/filesets/fs")


def test_upload_directory_keeps_name_without_trailing_slash(
    server: FakeFilesServer, client: FilesClient, tmp_path: Path
) -> None:
    server.filesets[(WORKSPACE, "fs")] = {}
    src = tmp_path / "src"
    (src / "n").mkdir(parents=True)
    (src / "a.txt").write_bytes(b"a")
    (src / "n" / "b.txt").write_bytes(b"b")

    transfer.upload(client, local_path=str(src), fileset="fs")

    assert server.filesets[(WORKSPACE, "fs")] == {"src/a.txt": b"a", "src/n/b.txt": b"b"}


def test_upload_directory_contents_with_trailing_slash(
    server: FakeFilesServer, client: FilesClient, tmp_path: Path
) -> None:
    server.filesets[(WORKSPACE, "fs")] = {}
    src = tmp_path / "src"
    (src / "n").mkdir(parents=True)
    (src / "a.txt").write_bytes(b"a")
    (src / "n" / "b.txt").write_bytes(b"b")

    transfer.upload(client, local_path=f"{src}/", fileset="fs", remote_path="up/")

    assert server.filesets[(WORKSPACE, "fs")] == {"up/a.txt": b"a", "up/n/b.txt": b"b"}


def test_upload_fileset_from_remote_path_ref(server: FakeFilesServer, client: FilesClient, tmp_path: Path) -> None:
    server.filesets[("other", "fs")] = {}
    local = tmp_path / "a.txt"
    local.write_bytes(b"x")

    result = transfer.upload(client, local_path=str(local), remote_path="other/fs#dir/")

    assert result.workspace == "other"
    assert server.filesets[("other", "fs")] == {"dir/a.txt": b"x"}


def test_upload_requires_fileset_without_auto_create(client: FilesClient, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fileset_auto_create is False"):
        transfer.upload(client, local_path=str(tmp_path))


def test_upload_auto_create_named_fileset_is_idempotent(
    server: FakeFilesServer, client: FilesClient, tmp_path: Path
) -> None:
    local = tmp_path / "a.txt"
    local.write_bytes(b"x")

    first = transfer.upload(client, local_path=str(local), fileset="new", fileset_auto_create=True)
    second = transfer.upload(client, local_path=str(local), fileset="new", fileset_auto_create=True)

    assert first.name == second.name == "new"
    posts = [r for r in server.requests if r.method == "POST"]
    assert [json.loads(r.content) for r in posts] == [{"name": "new"}, {"name": "new"}]
    # The second create 409s and is resolved by re-fetching the fileset (exist_ok).
    assert server.filesets[(WORKSPACE, "new")] == {"a.txt": b"x"}


def test_upload_auto_create_generates_name(server: FakeFilesServer, client: FilesClient, tmp_path: Path) -> None:
    local = tmp_path / "a.txt"
    local.write_bytes(b"x")

    result = transfer.upload(client, local_path=str(local), fileset_auto_create=True)

    assert result.name.startswith("fileset-")
    assert server.filesets[(WORKSPACE, result.name)] == {"a.txt": b"x"}
    assert server.calls()[0] == ("POST", "/apis/files/v2/workspaces/default/filesets")


def test_upload_uses_supplied_filesystem(server: FakeFilesServer, client: FilesClient, tmp_path: Path) -> None:
    server.filesets[(WORKSPACE, "fs")] = {}
    local = tmp_path / "a.txt"
    local.write_bytes(b"x")
    fs = FilesetFileSystem(client=client)

    transfer.upload(client, local_path=str(local), fileset="fs", filesystem=fs)

    assert server.filesets[(WORKSPACE, "fs")] == {"a.txt": b"x"}


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


@pytest.fixture
def populated(server: FakeFilesServer) -> dict[str, bytes]:
    files = {"a/file1.txt": b"content1", "a/b/file2.txt": b"content2", "a/b/file3.txt": b"content3", "r.json": b"{}"}
    server.filesets[(WORKSPACE, "fs")] = dict(files)
    return files


def test_download_single_file_into_existing_directory(
    populated: dict[str, bytes], client: FilesClient, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    out.mkdir()

    transfer.download(client, fileset="fs", remote_path="a/b/file2.txt", local_path=str(out))

    assert (out / "file2.txt").read_bytes() == b"content2"


def test_download_single_file_to_exact_path(populated: dict[str, bytes], client: FilesClient, tmp_path: Path) -> None:
    dest = tmp_path / "renamed.txt"

    transfer.download(client, fileset="fs", remote_path="a/b/file2.txt", local_path=str(dest))

    assert dest.read_bytes() == b"content2"


def test_download_directory_contents_with_trailing_slash(
    populated: dict[str, bytes], client: FilesClient, tmp_path: Path
) -> None:
    out = tmp_path / "out"

    transfer.download(client, fileset="fs", remote_path="a/", local_path=f"{out}/")

    assert (out / "file1.txt").read_bytes() == b"content1"
    assert (out / "b" / "file2.txt").read_bytes() == b"content2"
    assert (out / "b" / "file3.txt").read_bytes() == b"content3"


def test_download_directory_keeps_name_without_trailing_slash(
    populated: dict[str, bytes], client: FilesClient, tmp_path: Path
) -> None:
    out = tmp_path / "out"

    transfer.download(client, fileset="fs", remote_path="a/b", local_path=str(out))

    assert (out / "b" / "file2.txt").read_bytes() == b"content2"
    assert (out / "b" / "file3.txt").read_bytes() == b"content3"
    assert not (out / "file1.txt").exists()


def test_download_fileset_root_copies_contents(
    populated: dict[str, bytes], client: FilesClient, tmp_path: Path
) -> None:
    out = tmp_path / "out"

    transfer.download(client, fileset="fs", local_path=str(out))

    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == sorted(populated)


def test_download_glob_preserves_relative_paths(
    populated: dict[str, bytes], server: FakeFilesServer, client: FilesClient, tmp_path: Path
) -> None:
    out = tmp_path / "out"

    transfer.download(client, fileset="fs", remote_path="a/b/*.txt", local_path=str(out))

    assert (out / "a" / "b" / "file2.txt").read_bytes() == b"content2"
    assert (out / "a" / "b" / "file3.txt").read_bytes() == b"content3"
    assert not (out / "a" / "file1.txt").exists()
    downloads = sorted(r.url.path for r in server.requests if r.method == "GET" and "/-/" in r.url.path)
    assert downloads == [
        "/apis/files/v2/workspaces/default/filesets/fs/-/a/b/file2.txt",
        "/apis/files/v2/workspaces/default/filesets/fs/-/a/b/file3.txt",
    ]


def test_download_glob_without_matches_is_noop(
    populated: dict[str, bytes], server: FakeFilesServer, client: FilesClient, tmp_path: Path
) -> None:
    transfer.download(client, fileset="fs", remote_path="*.parquet", local_path=str(tmp_path))

    assert server.calls() == [("GET", "/apis/files/v2/workspaces/default/filesets/fs/files")]


def test_download_list_of_paths(populated: dict[str, bytes], client: FilesClient, tmp_path: Path) -> None:
    out = tmp_path / "out"

    transfer.download(client, fileset="fs", remote_path=["a/file1.txt", "r.json"], local_path=str(out))

    assert (out / "a" / "file1.txt").read_bytes() == b"content1"
    assert (out / "r.json").read_bytes() == b"{}"


def test_download_list_requires_fileset_and_workspace(server: FakeFilesServer, tmp_path: Path) -> None:
    no_workspace = FilesClient(base_url=BASE, http_client=_HttpClient(server))

    with pytest.raises(ValueError, match="fileset must be provided"):
        transfer.download(no_workspace, remote_path=["a"], local_path=str(tmp_path))
    with pytest.raises(ValueError, match="workspace must be provided"):
        transfer.download(no_workspace, fileset="fs", remote_path=["a"], local_path=str(tmp_path))


def test_download_empty_list_is_noop(server: FakeFilesServer, client: FilesClient, tmp_path: Path) -> None:
    transfer.download(client, fileset="fs", remote_path=[], local_path=str(tmp_path))

    assert server.requests == []


def test_download_requires_fileset(client: FilesClient, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Fileset must be specified"):
        transfer.download(client, remote_path="a/", local_path=str(tmp_path))


def test_download_missing_fileset_raises_not_found(client: FilesClient, tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        transfer.download(client, fileset="nope", local_path=str(tmp_path))


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete_file(populated: dict[str, bytes], server: FakeFilesServer, client: FilesClient) -> None:
    transfer.delete(client, fileset="fs", remote_path="a/b/file2.txt")

    assert server.calls() == [("DELETE", "/apis/files/v2/workspaces/default/filesets/fs/-/a/b/file2.txt")]
    assert "a/b/file2.txt" not in server.filesets[(WORKSPACE, "fs")]


def test_delete_with_fileset_ref(populated: dict[str, bytes], server: FakeFilesServer, client: FilesClient) -> None:
    transfer.delete(client, remote_path="default/fs#r.json")

    assert server.calls() == [("DELETE", "/apis/files/v2/workspaces/default/filesets/fs/-/r.json")]


def test_delete_requires_fileset(client: FilesClient) -> None:
    with pytest.raises(ValueError, match="Fileset must be specified"):
        transfer.delete(client, remote_path="a.txt")


def test_delete_missing_file_raises_not_found(populated: dict[str, bytes], client: FilesClient) -> None:
    with pytest.raises(NotFoundError):
        transfer.delete(client, fileset="fs", remote_path="nope.txt")


# ---------------------------------------------------------------------------
# async twins
# ---------------------------------------------------------------------------


async def test_async_roundtrip(server: FakeFilesServer, async_client: AsyncFilesClient, tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_bytes(b"a")
    (src / "b.json").write_bytes(b"{}")

    result = await transfer.async_upload(
        async_client, local_path=f"{src}/", fileset="fs", remote_path="up/", fileset_auto_create=True
    )
    assert result.name == "fs"
    assert server.filesets[(WORKSPACE, "fs")] == {"up/a.txt": b"a", "up/b.json": b"{}"}

    listed = await transfer.async_list_files(async_client, fileset="fs", remote_path="up/*.json")
    assert [f.path for f in listed.data] == ["up/b.json"]

    out = tmp_path / "out"
    await transfer.async_download(async_client, fileset="fs", remote_path="up/", local_path=f"{out}/")
    assert (out / "a.txt").read_bytes() == b"a"
    assert (out / "b.json").read_bytes() == b"{}"

    await transfer.async_download(async_client, fileset="fs", remote_path=["up/a.txt"], local_path=str(out / "l"))
    assert (out / "l" / "up" / "a.txt").read_bytes() == b"a"

    await transfer.async_delete(async_client, fileset="fs", remote_path="up/a.txt")
    assert server.filesets[(WORKSPACE, "fs")] == {"up/b.json": b"{}"}


async def test_async_requires_fileset(async_client: AsyncFilesClient, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Fileset must be specified"):
        await transfer.async_list_files(async_client, remote_path="x/")
    with pytest.raises(ValueError, match="Fileset must be specified"):
        await transfer.async_download(async_client, remote_path="x/", local_path=str(tmp_path))
    with pytest.raises(ValueError, match="Fileset must be specified"):
        await transfer.async_delete(async_client, remote_path="x")
    with pytest.raises(ValueError, match="fileset_auto_create is False"):
        await transfer.async_upload(async_client, local_path=str(tmp_path))
