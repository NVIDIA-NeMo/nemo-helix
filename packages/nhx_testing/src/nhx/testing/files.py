# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An in-memory Files service for unit tests.

Tests that need a Files client get a real :class:`AsyncFilesClient` whose requests are
answered from memory, instead of a hand-written subclass that can drift from the client's
signatures and skips its request building, response parsing, and error mapping::

    files = InMemoryFilesService()
    await store_bundle(files.client(), "default", "my-metric", bundle)
    assert ("default", fileset_name) in files.filesets
"""

import json
import re
from urllib.parse import unquote

import httpx
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.metadata import FilesetMetadata
from nemo_helix_plugin.files.storage_config import LocalStorageConfig
from nemo_helix_plugin.files.types import FilesetFileOutput, FilesetOutput, FilesetPurpose
from nhx.testing.client import mock_async_nemo_client

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
        #: When set, uploads fail with this HTTP status (e.g. 500) instead of storing content.
        self.upload_failure_status: int | None = None

    def client(self, *, workspace: str = "default") -> AsyncFilesClient:
        """Return a real async Files client backed by this service."""
        return mock_async_nemo_client(self, AsyncFilesClient, workspace=workspace)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = unquote(request.url.path)
        if match := _FILE.match(path):
            return self._file(request, match["workspace"], match["name"], match["path"])
        if match := _FILESETS.match(path):
            return self._fileset(request, match["workspace"], match["name"])
        return httpx.Response(404, json={"detail": f"no in-memory Files route for {request.method} {path}"})

    def _fileset(self, request: httpx.Request, workspace: str, name: str | None) -> httpx.Response:
        if name is None and request.method == "POST":
            name = json.loads(request.content)["name"]
            if (workspace, name) in self.filesets:
                return httpx.Response(409, json={"detail": f"fileset {workspace}/{name} already exists"})
            self.filesets[(workspace, name)] = {}
            return httpx.Response(201, json=_fileset_json(workspace, name))
        if name is not None and (workspace, name) not in self.filesets:
            return httpx.Response(404, json={"detail": f"fileset {workspace}/{name} not found"})
        if name is not None and request.method == "GET":
            return httpx.Response(200, json=_fileset_json(workspace, name))
        if name is not None and request.method == "DELETE":
            del self.filesets[(workspace, name)]
            return httpx.Response(200, json=_fileset_json(workspace, name))
        return httpx.Response(405, json={"detail": f"{request.method} is not supported here"})

    def _file(self, request: httpx.Request, workspace: str, name: str, path: str) -> httpx.Response:
        files = self.filesets.get((workspace, name))
        if files is None:
            return httpx.Response(404, json={"detail": f"fileset {workspace}/{name} not found"})
        if request.method == "PUT":
            if self.upload_failure_status is not None:
                return httpx.Response(self.upload_failure_status, json={"detail": "upload failed"})
            files[path] = request.read()
            file = FilesetFileOutput(
                file_ref=f"{workspace}/{name}/{path}",
                file_url=f"file://{name}/{path}",
                path=path,
                size=len(files[path]),
            )
            return httpx.Response(200, json=file.model_dump(mode="json"))
        if request.method == "GET":
            if path not in files:
                return httpx.Response(404, json={"detail": f"file {path} not found in {workspace}/{name}"})
            return httpx.Response(200, content=files[path])
        return httpx.Response(405, json={"detail": f"{request.method} is not supported here"})


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
