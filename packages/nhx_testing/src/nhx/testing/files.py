# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An in-memory Files service for unit tests.

Tests that need a Files client get a real :class:`AsyncFilesClient` whose requests are
answered from memory, instead of a hand-written subclass that can drift from the client's
signatures and skips its request building, response parsing, and error mapping::

    files = InMemoryFilesService()
    await store_bundle(files.client(), "default", "my-metric", bundle)
    assert ("default", fileset_name) in files.filesets

Each request is recorded in :attr:`InMemoryFilesService.calls`, and any operation can be made to fail::

    files.failures["upload"] = 500  # or an exception to raise, e.g. RuntimeError("boom")
"""

import json
import re
from collections.abc import Awaitable, Callable
from urllib.parse import unquote

import httpx
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.metadata import FilesetMetadata
from nemo_helix_plugin.files.storage_config import LocalStorageConfig
from nemo_helix_plugin.files.types import FilesetFileOutput, FilesetOutput, FilesetPurpose
from nhx.testing.client import mock_async_nemo_client

#: How a configured call fails: an HTTP status to answer with, or an exception to raise.
Failure = int | Exception

_FILESETS = re.compile(r"^/apis/files/v2/workspaces/(?P<workspace>[^/]+)/filesets(?:/(?P<name>[^/]+))?$")
_FILE = re.compile(r"^/apis/files/v2/workspaces/(?P<workspace>[^/]+)/filesets/(?P<name>[^/]+)/-/(?P<path>.+)$")


class InMemoryFilesService:
    """Answers Files API requests from an in-memory store of filesets and their files.

    Covers fileset create, get, and delete plus file upload and download — the calls the
    platform's typed :class:`AsyncFilesClient` makes for those endpoints.
    """

    def __init__(self) -> None:
        #: Stored filesets, keyed by ``(workspace, name)``, mapping file path to content.
        self.filesets: dict[tuple[str, str], dict[str, bytes]] = {}
        #: Every request as ``(operation, workspace, name)``, in order. Operations are ``"create"``,
        #: ``"get"``, ``"delete"``, ``"upload"`` and ``"download"``.
        self.calls: list[tuple[str, str, str]] = []
        #: Failures by operation, applied before the request touches storage: an ``int`` answers
        #: with that HTTP status, an exception is raised from the transport.
        self.failures: dict[str, Failure] = {}
        #: Called with each ``(operation, workspace, name)`` not covered by :attr:`failures`; a
        #: returned failure is applied the same way, ``None`` lets the request through.
        self.fail_when: Callable[[str, str, str], Failure | None] | None = None
        #: Awaited after each upload is stored and before it is answered, e.g. to hold concurrent
        #: uploads at a barrier.
        self.after_upload: Callable[[], Awaitable[None]] | None = None

    def client(self, *, workspace: str = "default") -> AsyncFilesClient:
        """Return a real async Files client backed by this service."""
        return mock_async_nemo_client(self, AsyncFilesClient, workspace=workspace)

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        # Match the still-encoded path and decode each captured segment once, so a literal "%2F"
        # in a file name isn't decoded a second time into a "/".
        path = request.url.raw_path.partition(b"?")[0].decode("ascii")
        if match := _FILE.match(path):
            workspace, name, file_path = (unquote(match[group]) for group in ("workspace", "name", "path"))
            return await self._file(request, workspace, name, file_path)
        if match := _FILESETS.match(path):
            name = match["name"]
            return self._fileset(request, unquote(match["workspace"]), unquote(name) if name else None)
        return httpx.Response(404, json={"detail": f"no in-memory Files route for {request.method} {path}"})

    def _failure(self, operation: str, workspace: str, name: str) -> httpx.Response | None:
        """Record the call and answer with its configured failure, if any."""
        self.calls.append((operation, workspace, name))
        failure = self.failures.get(operation)
        if failure is None and self.fail_when is not None:
            failure = self.fail_when(operation, workspace, name)
        if failure is None:
            return None
        if isinstance(failure, Exception):
            raise failure
        return httpx.Response(failure, json={"detail": f"{operation} {workspace}/{name} failed"})

    def _fileset(self, request: httpx.Request, workspace: str, name: str | None) -> httpx.Response:
        if name is None and request.method == "POST":
            name = json.loads(request.content)["name"]
            if failed := self._failure("create", workspace, name):
                return failed
            if (workspace, name) in self.filesets:
                return httpx.Response(409, json={"detail": f"fileset {workspace}/{name} already exists"})
            self.filesets[(workspace, name)] = {}
            return httpx.Response(201, json=_fileset_json(workspace, name))
        operation = {"GET": "get", "DELETE": "delete"}.get(request.method)
        if name is None or operation is None:
            return httpx.Response(405, json={"detail": f"{request.method} is not supported here"})
        if failed := self._failure(operation, workspace, name):
            return failed
        if (workspace, name) not in self.filesets:
            return httpx.Response(404, json={"detail": f"fileset {workspace}/{name} not found"})
        if request.method == "GET":
            return httpx.Response(200, json=_fileset_json(workspace, name))
        del self.filesets[(workspace, name)]
        return httpx.Response(200, json=_fileset_json(workspace, name))

    async def _file(self, request: httpx.Request, workspace: str, name: str, path: str) -> httpx.Response:
        operation = {"PUT": "upload", "GET": "download"}.get(request.method)
        if operation is None:
            return httpx.Response(405, json={"detail": f"{request.method} is not supported here"})
        if failed := self._failure(operation, workspace, name):
            return failed
        files = self.filesets.get((workspace, name))
        if files is None:
            return httpx.Response(404, json={"detail": f"fileset {workspace}/{name} not found"})
        if request.method == "PUT":
            files[path] = request.content
            if self.after_upload is not None:
                await self.after_upload()
            file = FilesetFileOutput(
                file_ref=f"{workspace}/{name}/{path}",
                file_url=f"file://{name}/{path}",
                path=path,
                size=len(files[path]),
            )
            return httpx.Response(200, json=file.model_dump(mode="json"))
        if path not in files:
            return httpx.Response(404, json={"detail": f"file {path} not found in {workspace}/{name}"})
        return httpx.Response(200, content=files[path])


def _fileset_json(workspace: str, name: str) -> dict[str, object]:
    fileset = FilesetOutput(
        id=f"fileset-{workspace}-{name}",
        name=name,
        workspace=workspace,
        description="",
        purpose=FilesetPurpose.GENERIC,
        storage=LocalStorageConfig(path=f"/filesets/{workspace}/{name}"),
        metadata=FilesetMetadata(),
        custom_fields={},
        project="",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )
    return fileset.model_dump(mode="json")
