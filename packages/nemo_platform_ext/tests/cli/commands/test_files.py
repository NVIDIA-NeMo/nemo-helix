# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-level tests for ``nemo files`` against a scripted typed client.

Each test drives the real Typer command with a ``NemoClient`` whose transport
records the HTTP requests, so the assertions pin the exact method, path, query
string, and JSON body the command sends.

The transfer commands (``upload``/``download``/``delete``) run through the
fileset filesystem, which issues its requests from an ``httpx.AsyncClient``
built off the sync client's transport. The recorder therefore also serves as
an ASGI app so both the sync and the async request paths hit the same script.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from nemo_platform_ext.cli.app import app
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_plugin.client.client import NemoClient
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Receive, Scope, Send
from typer.testing import CliRunner

FILESET = {
    "id": "fileset-1",
    "name": "my-fileset",
    "workspace": "default",
    "description": "",
    "purpose": "generic",
    "storage": {"type": "local", "path": "/data/my-fileset"},
    "metadata": {},
    "custom_fields": {},
    "project": "",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
}

FILE = {
    "file_ref": "default/my-fileset#data/train.jsonl",
    "file_url": "/apis/files/v2/workspaces/default/filesets/my-fileset/-/data/train.jsonl",
    "path": "data/train.jsonl",
    "size": 12,
    "cache_status": None,
}


@dataclass
class Recorder:
    """Records requests and answers each with the next scripted response or a route handler."""

    responses: list[httpx.Response] = field(default_factory=list)
    handler: Callable[[httpx.Request], httpx.Response] | None = None
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        if self.handler is not None:
            return self.handler(request)
        if not self.responses:
            raise AssertionError(f"unexpected request {request.method} {request.url}")
        return self.responses.pop(0)

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    def calls(self) -> list[tuple[str, str]]:
        return [(request.method, request.url.path) for request in self.requests]

    async def asgi_app(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Serve the same script to the filesystem's async client."""
        incoming = Request(scope, receive)
        body = await incoming.body()
        request = httpx.Request(incoming.method, str(incoming.url), headers=incoming.headers.raw, content=body)
        response = self(request)
        await Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))(
            scope, receive, send
        )


class _RecordingHttpClient(httpx.Client):
    """Sync httpx client over the recorder that also exposes it as an ASGI app."""

    def __init__(self, recorder: Recorder) -> None:
        super().__init__(transport=httpx.MockTransport(recorder), base_url="http://test")
        self._recorder = recorder

    @property
    def asgi_app(self):
        return self._recorder.asgi_app


def make_runner(recorder: Recorder, *, workspace: str | None = "default") -> tuple[CliRunner, CLIContext]:
    client = NemoClient(base_url="http://test", workspace=workspace, http_client=_RecordingHttpClient(recorder))
    state = CLIContext(overrides={"base_url": "http://test", "output_format": "json"}, _client=client)
    return CliRunner(), state


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.touch()
    monkeypatch.setenv("NMP_CONFIG_FILE", str(config_file))
    monkeypatch.delenv("NMP_ACCESS_TOKEN", raising=False)


def _page(items: list[dict], page: int, total_pages: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": items,
            "pagination": {
                "page": page,
                "page_size": 1,
                "current_page_size": len(items),
                "total_pages": total_pages,
                "total_results": total_pages,
            },
        },
    )


# ---------------------------------------------------------------------------
# filesets
# ---------------------------------------------------------------------------


def test_filesets_get_uses_client_default_workspace() -> None:
    recorder = Recorder([httpx.Response(200, json=FILESET)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "filesets", "get", "my-fileset"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "GET"
    assert recorder.last.url.path == "/apis/files/v2/workspaces/default/filesets/my-fileset"
    assert json.loads(result.stdout)["name"] == "my-fileset"


def test_filesets_get_explicit_workspace_overrides_default() -> None:
    recorder = Recorder([httpx.Response(200, json={**FILESET, "workspace": "other"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "filesets", "get", "my-fileset", "--workspace", "other"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/files/v2/workspaces/other/filesets/my-fileset"


def test_filesets_get_without_any_workspace_is_usage_error() -> None:
    recorder = Recorder()
    runner, state = make_runner(recorder, workspace=None)

    result = runner.invoke(app, ["files", "filesets", "get", "my-fileset"], obj=state)

    assert result.exit_code == 2
    assert "Missing workspace" in result.stderr
    assert recorder.requests == []


def test_filesets_refresh_posts_to_the_refresh_route() -> None:
    recorder = Recorder([httpx.Response(200, json=FILESET)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "filesets", "refresh", "my-fileset", "--workspace", "other"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == "/apis/files/v2/workspaces/other/filesets/my-fileset/refresh"
    assert recorder.last.content == b""
    assert json.loads(result.stdout)["name"] == "my-fileset"


def test_filesets_refresh_conflict_maps_to_remote_error_exit_code() -> None:
    recorder = Recorder([httpx.Response(409, json={"detail": "Fileset does not track a mutable revision"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "filesets", "refresh", "my-fileset"], obj=state)

    assert result.exit_code == 3
    assert "mutable revision" in result.stderr


def test_filesets_get_not_found_maps_to_remote_error_exit_code() -> None:
    recorder = Recorder([httpx.Response(404, json={"detail": "Fileset 'nope' not found"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "filesets", "get", "nope"], obj=state)

    assert result.exit_code == 3
    assert "Not found: (404) Fileset 'nope' not found" in result.stderr
    assert "nemo files filesets list" in result.stderr


def test_filesets_create_sends_only_provided_fields() -> None:
    recorder = Recorder([httpx.Response(201, json=FILESET)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "files",
            "filesets",
            "create",
            "my-fileset",
            "--description",
            "training data",
            "--purpose",
            "dataset",
            "--custom-fields",
            '{"team": "nlp"}',
            "--metadata",
            '{"dataset": {"schema": {"columns": ["id"]}}}',
            "--storage",
            '{"type": "local", "path": "/data/my-fileset"}',
            "--project",
            "proj",
            "--cache",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert len(recorder.requests) == 1
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == "/apis/files/v2/workspaces/default/filesets"
    assert json.loads(recorder.last.content) == {
        "name": "my-fileset",
        "description": "training data",
        "purpose": "dataset",
        "custom_fields": {"team": "nlp"},
        "metadata": {"dataset": {"schema": {"columns": ["id"]}}},
        "storage": {"type": "local", "path": "/data/my-fileset"},
        "project": "proj",
        "cache": True,
    }
    assert json.loads(result.stdout)["name"] == "my-fileset"


def test_filesets_create_minimal_body() -> None:
    recorder = Recorder([httpx.Response(201, json=FILESET)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "filesets", "create", "my-fileset"], obj=state)

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"name": "my-fileset"}


def test_filesets_create_from_input_data_with_flag_override() -> None:
    recorder = Recorder([httpx.Response(201, json=FILESET)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "files",
            "filesets",
            "create",
            "--input-data",
            '{"name": "from-input", "description": "from input", "workspace": "other"}',
            "--description",
            "from flag",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/files/v2/workspaces/other/filesets"
    assert json.loads(recorder.last.content) == {"name": "from-input", "description": "from flag"}


def test_filesets_create_from_stdin(tmp_path: Path) -> None:
    recorder = Recorder([httpx.Response(201, json=FILESET)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["files", "filesets", "create", "my-fileset", "--input-file", "-"],
        obj=state,
        input='{"description": "piped"}\n',
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"name": "my-fileset", "description": "piped"}


def test_filesets_create_requires_name() -> None:
    recorder = Recorder()
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "filesets", "create", "--description", "x"], obj=state)

    assert result.exit_code == 2
    assert "name" in result.stderr
    assert recorder.requests == []


def test_filesets_create_rejects_invalid_name_locally() -> None:
    recorder = Recorder()
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "filesets", "create", "X"], obj=state)

    assert result.exit_code == 2
    assert "Invalid input" in result.stderr
    assert recorder.requests == []


def test_filesets_create_exist_ok_returns_existing_on_conflict() -> None:
    recorder = Recorder([httpx.Response(409, json={"detail": "exists"}), httpx.Response(200, json=FILESET)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "filesets", "create", "my-fileset", "--exist-ok"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.calls() == [
        ("POST", "/apis/files/v2/workspaces/default/filesets"),
        ("GET", "/apis/files/v2/workspaces/default/filesets/my-fileset"),
    ]
    assert json.loads(result.stdout)["name"] == "my-fileset"


def test_filesets_create_conflict_without_exist_ok_is_remote_error() -> None:
    recorder = Recorder([httpx.Response(409, json={"detail": "exists"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "filesets", "create", "my-fileset"], obj=state)

    assert result.exit_code == 3
    assert "Conflict" in result.stderr


def test_filesets_update_only_sends_provided_fields() -> None:
    recorder = Recorder([httpx.Response(200, json=FILESET)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["files", "filesets", "update", "my-fileset", "--description", "new", "--purpose", "model"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "PATCH"
    assert recorder.last.url.path == "/apis/files/v2/workspaces/default/filesets/my-fileset"
    assert json.loads(recorder.last.content) == {"description": "new", "purpose": "model"}


def test_filesets_update_from_input_data() -> None:
    recorder = Recorder([httpx.Response(200, json=FILESET)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["files", "filesets", "update", "my-fileset", "--input-data", '{"custom_fields": {"a": 1}, "project": "p"}'],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {"custom_fields": {"a": 1}, "project": "p"}


def test_filesets_delete() -> None:
    recorder = Recorder([httpx.Response(200, json=FILESET)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "filesets", "delete", "my-fileset"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "DELETE"
    assert recorder.last.url.path == "/apis/files/v2/workspaces/default/filesets/my-fileset"
    assert "Deleted successfully" in result.stdout


def test_filesets_list_passes_query_params_and_warns() -> None:
    recorder = Recorder([_page([FILESET], 1, 2)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["files", "filesets", "list", "--page", "1", "--page-size", "1", "--sort", "-created_at"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/files/v2/workspaces/default/filesets"
    assert dict(recorder.last.url.params) == {"page": "1", "page_size": "1", "sort": "-created_at"}
    body = json.loads(result.stdout)
    assert [item["name"] for item in body["data"]] == ["my-fileset"]
    assert body["pagination"]["total_pages"] == 2
    assert "More pages" in result.stderr


def test_filesets_list_merges_filter_fields_into_json_filter() -> None:
    recorder = Recorder([_page([FILESET], 1, 1)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "files",
            "filesets",
            "list",
            "--filter",
            '{"created_at": {"$gte": "2026-01-01"}}',
            "--filter.name",
            "my-fileset",
            "--filter.purpose",
            "dataset",
            "--filter.storage-type",
            "local",
            "--filter.description",
            "training",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert json.loads(dict(recorder.last.url.params)["filter"]) == {
        "created_at": {"$gte": "2026-01-01"},
        "name": "my-fileset",
        "purpose": "dataset",
        "storage_type": "local",
        "description": "training",
    }


def test_filesets_list_text_filter_is_forwarded_verbatim() -> None:
    recorder = Recorder([_page([FILESET], 1, 1)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "filesets", "list", "--filter", 'name~"my"'], obj=state)

    assert result.exit_code == 0, result.output
    assert dict(recorder.last.url.params) == {"filter": 'name~"my"'}


def test_filesets_list_all_pages_follows_every_page() -> None:
    recorder = Recorder([_page([FILESET], 1, 2), _page([{**FILESET, "name": "second"}], 2, 2)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "filesets", "list", "--page-size", "1", "--all-pages"], obj=state)

    assert result.exit_code == 0, result.output
    assert [dict(r.url.params).get("page") for r in recorder.requests] == [None, "2"]
    body = json.loads(result.stdout)
    assert [item["name"] for item in body["data"]] == ["my-fileset", "second"]
    assert body["pagination"]["total_results"] == 2
    assert "More pages" not in result.stderr


def test_filesets_list_table_output_uses_default_columns() -> None:
    recorder = Recorder([_page([FILESET], 1, 1)])
    runner, state = make_runner(recorder)
    state.overrides["output_format"] = "table"

    result = runner.invoke(app, ["files", "filesets", "list", "-f", "table"], obj=state)

    assert result.exit_code == 0, result.output
    output = result.stdout.lower()
    assert "name" in output and "workspace" in output and "created_at" in output
    assert "purpose" not in output


def test_filesets_list_stream_requires_json_output() -> None:
    recorder = Recorder()
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "filesets", "list", "-f", "table", "--stream"], obj=state)

    assert result.exit_code == 2
    assert recorder.requests == []


def test_filesets_create_code_output_renders_typed_client_call() -> None:
    recorder = Recorder()
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["files", "filesets", "create", "my-fileset", "--purpose", "dataset", "--exist-ok", "-f", "code"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "from nemo_platform_plugin.files.client import FilesClient" in result.stdout
    assert "from nemo_platform_plugin.files.types import CreateFilesetRequest, FilesetPurpose" in result.stdout
    assert 'client = FilesClient(base_url="http://test/")' in result.stdout
    assert "client.create_fileset(" in result.stdout
    assert 'CreateFilesetRequest(name="my-fileset", purpose=FilesetPurpose.DATASET)' in result.stdout
    assert "exist_ok=True" in result.stdout
    assert "NeMoPlatform" not in result.stdout


def test_filesets_list_code_output() -> None:
    recorder = Recorder()
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "filesets", "list", "--page-size", "5", "-f", "code"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert 'client.list_filesets(query_params={"page_size": 5})' in result.stdout
    assert "for item in response.page().items:" in result.stdout
    assert "NeMoPlatform" not in result.stdout


# ---------------------------------------------------------------------------
# files list / delete
# ---------------------------------------------------------------------------


def test_files_list_lists_fileset_root() -> None:
    recorder = Recorder([httpx.Response(200, json={"data": [FILE]})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "list", "my-fileset"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "GET"
    assert recorder.last.url.path == "/apis/files/v2/workspaces/default/filesets/my-fileset/files"
    assert dict(recorder.last.url.params) == {}
    assert [item["path"] for item in json.loads(result.stdout)] == ["data/train.jsonl"]


def test_files_list_passes_remote_path_as_path_query_param() -> None:
    recorder = Recorder([httpx.Response(200, json={"data": [FILE]})])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app, ["files", "list", "my-fileset", "--remote-path", "data/", "--workspace", "other"], obj=state
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/files/v2/workspaces/other/filesets/my-fileset/files"
    assert dict(recorder.last.url.params) == {"path": "data/"}


def test_files_list_glob_filters_client_side() -> None:
    other = {**FILE, "path": "data/readme.md", "file_ref": "default/my-fileset#data/readme.md"}
    recorder = Recorder([httpx.Response(200, json={"data": [FILE, other]})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "list", "my-fileset", "--remote-path", "data/*.jsonl"], obj=state)

    assert result.exit_code == 0, result.output
    assert dict(recorder.last.url.params) == {}
    assert [item["path"] for item in json.loads(result.stdout)] == ["data/train.jsonl"]


def test_files_list_code_output_matches_the_prefix_query_the_command_sends() -> None:
    recorder = Recorder()
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "list", "my-fileset", "--remote-path", "data/", "-f", "code"], obj=state)

    assert result.exit_code == 0, result.output
    assert 'query_params={"path": "data/"}' in result.stdout or "query_params={'path': 'data/'}" in result.stdout
    assert recorder.requests == []


def test_files_list_code_output_refuses_a_glob_it_cannot_express() -> None:
    """The glob is matched client-side after listing, so no single typed call reproduces it."""
    recorder = Recorder()
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "list", "my-fileset", "--remote-path", "*.json", "-f", "code"], obj=state)

    assert result.exit_code == 2, result.output
    assert "glob" in result.stderr
    assert recorder.requests == []


def test_files_list_table_output_uses_path_and_size_columns() -> None:
    recorder = Recorder([httpx.Response(200, json={"data": [FILE]})])
    runner, state = make_runner(recorder)
    state.overrides["output_format"] = "table"

    result = runner.invoke(app, ["files", "list", "my-fileset", "-f", "table"], obj=state)

    assert result.exit_code == 0, result.output
    assert "PATH" in result.stdout and "SIZE" in result.stdout
    assert "file_ref" not in result.stdout


def test_files_list_without_any_workspace_is_usage_error() -> None:
    recorder = Recorder()
    runner, state = make_runner(recorder, workspace=None)

    result = runner.invoke(app, ["files", "list", "my-fileset"], obj=state)

    assert result.exit_code == 2
    assert "Missing workspace" in result.stderr
    assert recorder.requests == []


def test_files_list_not_found() -> None:
    recorder = Recorder([httpx.Response(404, json={"detail": "Fileset 'nope' not found"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "list", "nope"], obj=state)

    assert result.exit_code == 3
    assert "Not found: (404)" in result.stderr


def test_files_list_code_output() -> None:
    recorder = Recorder()
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "list", "my-fileset", "--remote-path", "data/", "-f", "code"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert 'client.list_files(name="my-fileset", workspace="default", query_params={"path": "data/"})' in result.stdout
    assert "NeMoPlatform" not in result.stdout


def test_files_delete() -> None:
    recorder = Recorder([httpx.Response(200, json=FILE)])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "delete", "my-fileset", "--remote-path", "data/train.jsonl"], obj=state)

    assert result.exit_code == 0, result.output
    assert recorder.calls() == [("DELETE", "/apis/files/v2/workspaces/default/filesets/my-fileset/-/data/train.jsonl")]
    assert "Deleted my-fileset#data/train.jsonl" in result.stdout


def test_files_delete_requires_remote_path() -> None:
    recorder = Recorder()
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "delete", "my-fileset"], obj=state)

    assert result.exit_code == 2
    assert recorder.requests == []


def test_files_delete_not_found() -> None:
    recorder = Recorder([httpx.Response(404, json={"detail": "File not found"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "delete", "my-fileset", "--remote-path", "nope.txt"], obj=state)

    assert result.exit_code == 3
    assert "Not found: (404) File not found" in result.stderr


# ---------------------------------------------------------------------------
# files upload / download
# ---------------------------------------------------------------------------


def _fileset_store(
    files: dict[str, bytes], *, fileset: str = "my-fileset"
) -> Callable[[httpx.Request], httpx.Response]:
    """Route handler emulating one fileset: GET fileset, list, upload, download."""
    base = f"/apis/files/v2/workspaces/default/filesets/{fileset}"

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path == base:
            return httpx.Response(200, json={**FILESET, "name": fileset})
        if request.method == "POST" and path == "/apis/files/v2/workspaces/default/filesets":
            return httpx.Response(201, json={**FILESET, "name": json.loads(request.content)["name"]})
        if request.method == "GET" and path == f"{base}/files":
            prefix = request.url.params.get("path", "")
            data = [
                {
                    "file_ref": f"default/{fileset}#{name}",
                    "file_url": f"{base}/-/{name}",
                    "path": name,
                    "size": len(content),
                }
                for name, content in files.items()
                if name.startswith(prefix)
            ]
            return httpx.Response(200, json={"data": data})
        if path.startswith(f"{base}/-/"):
            name = path[len(f"{base}/-/") :]
            if request.method == "PUT":
                files[name] = request.content
                return httpx.Response(
                    200,
                    json={
                        "file_ref": f"default/{fileset}#{name}",
                        "file_url": path,
                        "path": name,
                        "size": len(request.content),
                    },
                )
            if request.method == "GET":
                if name not in files:
                    return httpx.Response(404, json={"detail": "File not found"})
                return httpx.Response(200, content=files[name], headers={"content-length": str(len(files[name]))})
        return httpx.Response(404, json={"detail": f"{request.method} {path} not found"})

    return handle


def test_upload_single_file_to_remote_path(tmp_path: Path) -> None:
    store: dict[str, bytes] = {}
    recorder = Recorder(handler=_fileset_store(store))
    runner, state = make_runner(recorder)
    local = tmp_path / "train.jsonl"
    local.write_bytes(b'{"a": 1}\n')

    result = runner.invoke(app, ["files", "upload", str(local), "my-fileset", "--remote-path", "data/"], obj=state)

    assert result.exit_code == 0, result.output
    assert "Completed upload to my-fileset#data/" in result.stdout
    assert store == {"data/train.jsonl": b'{"a": 1}\n'}
    # The fileset is validated first; the file is PUT under the remote path.
    assert recorder.calls()[0] == ("GET", "/apis/files/v2/workspaces/default/filesets/my-fileset")
    puts = [request for request in recorder.requests if request.method == "PUT"]
    assert [request.url.path for request in puts] == [
        "/apis/files/v2/workspaces/default/filesets/my-fileset/-/data/train.jsonl"
    ]
    assert puts[0].headers["content-length"] == "9"


def test_upload_directory_contents_with_trailing_slash(tmp_path: Path) -> None:
    store: dict[str, bytes] = {}
    recorder = Recorder(handler=_fileset_store(store))
    runner, state = make_runner(recorder)
    source = tmp_path / "src"
    (source / "nested").mkdir(parents=True)
    (source / "a.txt").write_text("a")
    (source / "nested" / "b.txt").write_text("b")

    result = runner.invoke(app, ["files", "upload", f"{source}/", "my-fileset"], obj=state)

    assert result.exit_code == 0, result.output
    assert "Completed upload to my-fileset" in result.stdout
    assert store == {"a.txt": b"a", "nested/b.txt": b"b"}


def test_upload_directory_without_trailing_slash_keeps_directory_name(tmp_path: Path) -> None:
    store: dict[str, bytes] = {}
    recorder = Recorder(handler=_fileset_store(store))
    runner, state = make_runner(recorder)
    source = tmp_path / "src"
    source.mkdir()
    (source / "a.txt").write_text("a")

    result = runner.invoke(app, ["files", "upload", str(source), "my-fileset"], obj=state)

    assert result.exit_code == 0, result.output
    assert store == {"src/a.txt": b"a"}


def test_upload_without_fileset_auto_creates_one(tmp_path: Path) -> None:
    created: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/apis/files/v2/workspaces/default/filesets":
            body = json.loads(request.content)
            assert body == {"name": body["name"]}
            created.append(body["name"])
            return httpx.Response(201, json={**FILESET, "name": body["name"]})
        if request.method == "GET" and request.url.path.endswith("/files"):
            return httpx.Response(200, json={"data": []})
        if request.method == "PUT":
            return httpx.Response(200, json={**FILE, "path": "train.jsonl"})
        if request.method == "GET":
            return httpx.Response(200, json={**FILESET, "name": created[0]})
        raise AssertionError(f"unexpected {request.method} {request.url}")

    recorder = Recorder(handler=handle)
    runner, state = make_runner(recorder)
    local = tmp_path / "train.jsonl"
    local.write_text("x")

    result = runner.invoke(app, ["files", "upload", str(local)], obj=state)

    assert result.exit_code == 0, result.output
    assert len(created) == 1 and created[0].startswith("fileset-")
    assert f"Completed upload to {created[0]}" in result.stdout
    assert recorder.calls()[0] == ("POST", "/apis/files/v2/workspaces/default/filesets")


def test_upload_to_missing_fileset_fails_before_transfer(tmp_path: Path) -> None:
    recorder = Recorder([httpx.Response(404, json={"detail": "Fileset 'nope' not found"})])
    runner, state = make_runner(recorder)
    local = tmp_path / "train.jsonl"
    local.write_text("x")

    result = runner.invoke(app, ["files", "upload", str(local), "nope"], obj=state)

    assert result.exit_code == 3
    assert "Not found: (404) Fileset 'nope' not found" in result.stderr
    assert recorder.calls() == [("GET", "/apis/files/v2/workspaces/default/filesets/nope")]


def test_upload_without_any_workspace_is_usage_error(tmp_path: Path) -> None:
    recorder = Recorder()
    runner, state = make_runner(recorder, workspace=None)
    local = tmp_path / "train.jsonl"
    local.write_text("x")

    result = runner.invoke(app, ["files", "upload", str(local), "my-fileset"], obj=state)

    assert result.exit_code == 2
    assert "Missing workspace" in result.stderr
    assert recorder.requests == []


def test_upload_missing_local_path_is_usage_error(tmp_path: Path) -> None:
    recorder = Recorder()
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "upload", str(tmp_path / "missing.txt"), "my-fileset"], obj=state)

    assert result.exit_code == 2
    assert recorder.requests == []


def test_download_single_file_into_directory(tmp_path: Path) -> None:
    recorder = Recorder(handler=_fileset_store({"a/b/file2.txt": b"content2", "a/file1.txt": b"content1"}))
    runner, state = make_runner(recorder)
    out = tmp_path / "out"
    out.mkdir()

    result = runner.invoke(
        app, ["files", "download", "my-fileset", "--remote-path", "a/b/file2.txt", "-o", str(out)], obj=state
    )

    assert result.exit_code == 0, result.output
    assert f"Downloaded my-fileset#a/b/file2.txt to {str(out)!r}" in result.stdout
    assert (out / "file2.txt").read_bytes() == b"content2"
    assert ("GET", "/apis/files/v2/workspaces/default/filesets/my-fileset/-/a/b/file2.txt") in recorder.calls()
    listing = next(r for r in recorder.requests if r.url.path.endswith("/files"))
    assert dict(listing.url.params) == {"path": "a/b/file2.txt"}


def test_download_directory_tree(tmp_path: Path) -> None:
    recorder = Recorder(handler=_fileset_store({"a/b/file2.txt": b"content2", "a/file1.txt": b"content1"}))
    runner, state = make_runner(recorder)
    out = tmp_path / "out"

    result = runner.invoke(app, ["files", "download", "my-fileset", "--remote-path", "a/", "-o", f"{out}/"], obj=state)

    assert result.exit_code == 0, result.output
    assert f"Downloaded my-fileset#a/ to {f'{out}/'!r}" in result.stdout
    assert (out / "file1.txt").read_bytes() == b"content1"
    assert (out / "b" / "file2.txt").read_bytes() == b"content2"


def test_download_whole_fileset_reports_root(tmp_path: Path) -> None:
    recorder = Recorder(handler=_fileset_store({"file1.txt": b"content1"}))
    runner, state = make_runner(recorder)
    out = tmp_path / "out"

    result = runner.invoke(app, ["files", "download", "my-fileset", "-o", str(out)], obj=state)

    assert result.exit_code == 0, result.output
    assert f"Downloaded my-fileset#/ to {str(out)!r}" in result.stdout
    assert (out / "file1.txt").read_bytes() == b"content1"


def test_download_requires_output_option() -> None:
    recorder = Recorder()
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "download", "my-fileset"], obj=state)

    assert result.exit_code == 2
    assert recorder.requests == []


def test_download_missing_fileset_is_remote_error(tmp_path: Path) -> None:
    recorder = Recorder(handler=lambda _request: httpx.Response(404, json={"detail": "Fileset 'nope' not found"}))
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "download", "nope", "-o", str(tmp_path)], obj=state)

    assert result.exit_code == 3
    assert "Not found: (404) Fileset 'nope' not found" in result.stderr


# ---------------------------------------------------------------------------
# otlp logs
# ---------------------------------------------------------------------------


def test_otlp_logs_create_sends_payload_and_artifact_base_path(tmp_path: Path) -> None:
    recorder = Recorder([httpx.Response(200, json={"partialSuccess": None})])
    runner, state = make_runner(recorder)
    payload = {"resourceLogs": [{"scopeLogs": []}]}
    payload_file = tmp_path / "logs.json"
    payload_file.write_text(json.dumps(payload))

    result = runner.invoke(
        app,
        [
            "files",
            "otlp",
            "logs",
            "create",
            "my-fileset",
            "--input-file",
            str(payload_file),
            "--artifact-base-path",
            "runs/1",
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == "/apis/files/v2/workspaces/default/filesets/my-fileset/otlp/v1/logs"
    assert dict(recorder.last.url.params) == {"artifact_base_path": "runs/1"}
    assert recorder.last.headers["content-type"] == "application/json"
    assert json.loads(recorder.last.content) == payload
    assert json.loads(result.stdout) == {"partial_success": None}


def test_otlp_logs_create_from_input_data_with_workspace_in_payload() -> None:
    recorder = Recorder([httpx.Response(200, json={})])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "files",
            "otlp",
            "logs",
            "create",
            "my-fileset",
            "--input-data",
            '{"resourceLogs": [], "workspace": "other", "artifact_base_path": "from-input"}',
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.url.path == "/apis/files/v2/workspaces/other/filesets/my-fileset/otlp/v1/logs"
    assert dict(recorder.last.url.params) == {"artifact_base_path": "from-input"}
    assert json.loads(recorder.last.content) == {"resourceLogs": []}


def test_otlp_logs_create_without_input_sends_empty_object() -> None:
    recorder = Recorder([httpx.Response(200, json={})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "otlp", "logs", "create", "my-fileset"], obj=state)

    assert result.exit_code == 0, result.output
    assert dict(recorder.last.url.params) == {}
    assert json.loads(recorder.last.content) == {}


def test_otlp_logs_create_bad_request_maps_to_remote_error() -> None:
    recorder = Recorder([httpx.Response(400, json={"detail": "Invalid JSON format"})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "otlp", "logs", "create", "my-fileset"], obj=state)

    assert result.exit_code == 3
    assert "Bad request: (400) Invalid JSON format" in result.stderr


def test_otlp_logs_query_sends_only_provided_fields() -> None:
    entry = {"timestamp": "2026-01-01T00:00:00Z", "job": "j", "job_step": "s", "job_task": "t", "message": "hi"}
    page = {"data": [entry], "total": 1, "next_page": None, "prev_page": None}
    recorder = Recorder([httpx.Response(200, json=page)])
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        [
            "files",
            "otlp",
            "logs",
            "query",
            "my-fileset",
            "--limit",
            "10",
            "--tail",
            "5",
            "--page-cursor",
            "abc",
            "--artifact-base-path",
            "runs/1",
            "--filters",
            '{"level": "ERROR"}',
        ],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.last.method == "POST"
    assert recorder.last.url.path == "/apis/files/v2/workspaces/default/filesets/my-fileset/otlp/v1/logs/query"
    assert json.loads(recorder.last.content) == {
        "limit": 10,
        "tail": 5,
        "page_cursor": "abc",
        "artifact_base_path": "runs/1",
        "filters": {"level": "ERROR"},
    }
    assert json.loads(result.stdout)["total"] == 1


def test_otlp_logs_query_minimal_body() -> None:
    recorder = Recorder([httpx.Response(200, json={"data": [], "total": 0, "next_page": None, "prev_page": None})])
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "otlp", "logs", "query", "my-fileset"], obj=state)

    assert result.exit_code == 0, result.output
    assert json.loads(recorder.last.content) == {}


def test_otlp_logs_query_rejects_non_mapping_filters_locally() -> None:
    recorder = Recorder()
    runner, state = make_runner(recorder)

    result = runner.invoke(app, ["files", "otlp", "logs", "query", "my-fileset", "--filters", "level"], obj=state)

    assert result.exit_code == 2
    assert "Invalid input" in result.stderr
    assert recorder.requests == []


def test_otlp_logs_query_code_output() -> None:
    recorder = Recorder()
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app, ["files", "otlp", "logs", "query", "my-fileset", "--limit", "3", "-f", "code"], obj=state
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "from nemo_platform_plugin.files.types import OtlpLogQueryRequest" in result.stdout
    assert 'client.query_otlp_logs(name="my-fileset", body=OtlpLogQueryRequest(limit=3))' in result.stdout
    assert "NeMoPlatform" not in result.stdout


def test_otlp_logs_create_code_output() -> None:
    recorder = Recorder()
    runner, state = make_runner(recorder)

    result = runner.invoke(
        app,
        ["files", "otlp", "logs", "create", "my-fileset", "--artifact-base-path", "runs/1", "-f", "code"],
        obj=state,
    )

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "client.upload_otlp_logs(" in result.stdout
    assert 'query_params={"artifact_base_path": "runs/1"}' in result.stdout
    assert "NeMoPlatform" not in result.stdout
