# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo agents skills`` against the typed clients over ``httpx.MockTransport``."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote

import httpx
import pytest
from nemo_agents_plugin.cli import AgentsCLI
from nemo_agents_plugin.skills_cli import DEFAULT_UPLOAD_FILESET
from typer.testing import CliRunner

runner = CliRunner()
SKILLS = "/apis/agents/v2/workspaces/default/skills"
DIGEST = "c" * 64


def _skill(revision: int = 1) -> dict[str, Any]:
    return {
        "name": "my-skill",
        "workspace": "default",
        "revision": revision,
        "content_hash": DIGEST,
        "tags": {"latest": revision},
        "frontmatter": {"name": "my-skill", "description": "Use it.", "tags": ["pdf"], "extra": {}},
        "archive": {
            "archive_ref": f"default/nemo-agents-skills#my-skill/{'d' * 64}/skill_archive",
            "files_hash": "d" * 64,
            "skill_name": "my-skill",
        },
    }


def _page(*items: dict[str, Any]) -> dict[str, Any]:
    return {
        "data": list(items),
        "pagination": {
            "page": 1,
            "page_size": 100,
            "current_page_size": len(items),
            "total_pages": 1,
            "total_results": len(items),
        },
    }


@dataclass
class Recorder:
    """Records requests and answers skill and Files routes."""

    respond: Callable[[httpx.Request], httpx.Response]
    requests: list[httpx.Request] = field(default_factory=list)
    uploads: dict[str, bytes] = field(default_factory=dict)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = unquote(request.url.path)
        if path.startswith("/apis/files/"):
            if request.method == "PUT":
                self.uploads[path.split("/-/", 1)[1]] = request.read()
                return httpx.Response(200, json={"file_ref": "x", "file_url": "x", "path": "x", "size": 1})
            return httpx.Response(200, json=_fileset_json())
        return self.respond(request)

    def skill_calls(self) -> list[httpx.Request]:
        return [r for r in self.requests if "/apis/agents/" in r.url.path]


def _fileset_json() -> dict[str, Any]:
    return {
        "id": "id-fs",
        "name": DEFAULT_UPLOAD_FILESET,
        "workspace": "default",
        "description": "",
        "purpose": "generic",
        "storage": {"type": "local", "path": "/data/fs"},
        "metadata": {},
        "custom_fields": {},
        "project": "",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }


@pytest.fixture
def invoke(make_cli_state) -> Callable[..., tuple[Any, Recorder]]:
    def run(args: list[str], respond: Callable[[httpx.Request], httpx.Response] | None = None):
        recorder = Recorder(respond or (lambda _: httpx.Response(200, json=_skill())))
        result = runner.invoke(AgentsCLI().get_cli(), ["skills", *args], obj=make_cli_state(recorder))
        return result, recorder

    return run


def _skill_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "my-skill"
    (directory / "scripts").mkdir(parents=True)
    (directory / "SKILL.md").write_text("---\nname: my-skill\ndescription: Use it.\n---\nBody\n")
    (directory / "scripts" / "run.sh").write_text("echo hi\n")
    return directory


def test_create_from_fileset_posts_the_body(invoke) -> None:
    result, recorder = invoke(
        [
            "create",
            "my-skill",
            "--fileset",
            "default/src#skills/my-skill",
            "--tag",
            "v1",
            "--taskset",
            f"default/ts#{'a' * 64}",
            "-f",
            "json",
        ],
        lambda _: httpx.Response(201, json=_skill()),
    )

    assert result.exit_code == 0, result.output
    (request,) = recorder.requests
    assert (request.method, request.url.path) == ("POST", f"{SKILLS}/my-skill")
    assert json.loads(request.content) == {
        "source": {"fileset": "default/src#skills/my-skill"},
        "taskset": f"default/ts#{'a' * 64}",
        "tags": ["v1"],
    }
    assert json.loads(result.stdout)["content_hash"] == DIGEST


def test_publish_from_dir_uploads_under_a_content_path_then_puts(invoke, tmp_path: Path) -> None:
    directory = _skill_dir(tmp_path)

    result, recorder = invoke(["publish", "my-skill", "--from-dir", str(directory), "-f", "json"])

    assert result.exit_code == 0, result.output
    (fileset_create,) = [r for r in recorder.requests if r.method == "POST"]
    assert fileset_create.url.path == "/apis/files/v2/workspaces/default/filesets"
    (put,) = recorder.skill_calls()
    assert (put.method, put.url.path) == ("PUT", f"{SKILLS}/my-skill")
    source = json.loads(put.content)["source"]["fileset"]
    assert source.startswith(f"default/{DEFAULT_UPLOAD_FILESET}#my-skill/")
    prefix = source.partition("#")[2]
    assert len(prefix.split("/")[1]) == 64
    assert recorder.uploads == {
        f"{prefix}/SKILL.md": (directory / "SKILL.md").read_bytes(),
        f"{prefix}/scripts/run.sh": b"echo hi\n",
    }


def test_from_dir_path_depends_only_on_content(invoke, tmp_path: Path) -> None:
    def prefix_for(directory: Path) -> str:
        result, recorder = invoke(["publish", "my-skill", "--from-dir", str(directory), "-f", "json"])
        assert result.exit_code == 0, result.output
        return json.loads(recorder.skill_calls()[0].content)["source"]["fileset"]

    first = prefix_for(_skill_dir(tmp_path / "a"))
    assert prefix_for(_skill_dir(tmp_path / "b")) == first
    changed = _skill_dir(tmp_path / "c")
    (changed / "SKILL.md").write_text("---\nname: my-skill\ndescription: Changed.\n---\n")
    assert prefix_for(changed) != first


def test_from_dir_honours_upload_fileset(invoke, tmp_path: Path) -> None:
    result, recorder = invoke(
        ["create", "my-skill", "--from-dir", str(_skill_dir(tmp_path)), "--upload-fileset", "mine", "-f", "json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(recorder.skill_calls()[0].content)["source"]["fileset"].startswith("default/mine#my-skill/")


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["create", "my-skill"], "exactly one of --from-dir or --fileset"),
        (["create", "my-skill", "--fileset", "default/src#x", "--upload-fileset", "f"], "--upload-fileset"),
    ],
)
def test_source_options_are_validated(invoke, args: list[str], message: str) -> None:
    result, recorder = invoke(args)
    assert result.exit_code == 2
    assert message in result.output
    assert recorder.requests == []


def test_from_dir_refuses_symlinks_and_a_missing_skill_md(invoke, tmp_path: Path) -> None:
    directory = _skill_dir(tmp_path)
    (directory / "link").symlink_to(directory / "SKILL.md")
    result, recorder = invoke(["create", "my-skill", "--from-dir", str(directory)])
    assert result.exit_code == 2 and "symlink" in result.output
    assert recorder.requests == []

    empty = tmp_path / "empty"
    empty.mkdir()
    result, recorder = invoke(["create", "my-skill", "--from-dir", str(empty)])
    assert result.exit_code == 2 and "SKILL.md" in result.output
    assert recorder.requests == []


def test_from_dir_help_documents_lost_file_modes(invoke) -> None:
    result, _ = invoke(["create", "--help"])
    assert "executable bits" in " ".join(result.output.split())


def test_from_dir_cannot_emit_code(invoke, tmp_path: Path) -> None:
    result, recorder = invoke(["create", "my-skill", "--from-dir", str(_skill_dir(tmp_path)), "-f", "code"])
    assert result.exit_code == 2
    assert recorder.requests == []


def test_code_output_sends_nothing(invoke) -> None:
    result, recorder = invoke(["create", "my-skill", "--fileset", "default/src#skills/my-skill", "-f", "code"])
    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "create_skill" in result.stdout
    compile(result.stdout, "<snippet>", "exec")


@pytest.mark.parametrize(
    ("args", "path"),
    [
        (["get", "my-skill"], f"{SKILLS}/my-skill"),
        (["get", "my-skill", "--revision", "v1"], f"{SKILLS}/my-skill/revisions/v1"),
        (["get", "my-skill", "-r", DIGEST], f"{SKILLS}/my-skill/revisions/{DIGEST}"),
    ],
)
def test_get_latest_or_a_revision(invoke, args: list[str], path: str) -> None:
    result, recorder = invoke([*args, "-f", "json"])
    assert result.exit_code == 0, result.output
    (request,) = recorder.requests
    assert (request.method, request.url.path) == ("GET", path)


def test_list_merges_tag_options_into_the_filter(invoke) -> None:
    result, recorder = invoke(
        [
            "list",
            "--tag",
            "pdf",
            "--tag",
            "docs",
            "--filter",
            '{"description": {"$like": "read"}}',
            "--sort",
            "-name",
            "-f",
            "json",
        ],
        lambda _: httpx.Response(200, json=_page(_skill())),
    )

    assert result.exit_code == 0, result.output
    (request,) = recorder.requests
    assert request.url.path == SKILLS
    query = parse_qs(request.url.query.decode())
    assert query["sort"] == ["-name"]
    assert json.loads(query["filter"][0]) == {"description": {"$like": "read"}, "tags": {"$in": ["pdf", "docs"]}}
    assert [row["name"] for row in json.loads(result.stdout)["data"]] == ["my-skill"]


def test_revisions_lists_newest_first(invoke) -> None:
    result, recorder = invoke(
        ["revisions", "my-skill", "--page-size", "5", "-f", "json"],
        lambda _: httpx.Response(200, json=_page(_skill(2), _skill(1))),
    )
    assert result.exit_code == 0, result.output
    (request,) = recorder.requests
    assert request.url.path == f"{SKILLS}/my-skill/revisions"
    assert parse_qs(request.url.query.decode())["page_size"] == ["5"]
    assert [row["revision"] for row in json.loads(result.stdout)["data"]] == [2, 1]


def test_tag_puts_with_the_revision_query(invoke) -> None:
    result, recorder = invoke(["tag", "my-skill", "stable", "--revision", DIGEST, "-f", "json"])
    assert result.exit_code == 0, result.output
    (request,) = recorder.requests
    assert (request.method, request.url.path) == ("PUT", f"{SKILLS}/my-skill/tags/stable")
    assert parse_qs(request.url.query.decode()) == {"revision": [DIGEST]}


def test_delete_requires_confirmation_and_maps_404(invoke) -> None:
    result, recorder = invoke(["delete", "my-skill"])
    assert result.exit_code == 2
    assert recorder.requests == []

    result, recorder = invoke(["delete", "my-skill", "--yes"], lambda _: httpx.Response(204))
    assert result.exit_code == 0, result.output
    assert "deleted" in result.stdout
    assert [(r.method, r.url.path) for r in recorder.requests] == [("DELETE", f"{SKILLS}/my-skill")]

    result, _ = invoke(["get", "absent"], lambda _: httpx.Response(404, json={"detail": "Skill not found"}))
    assert result.exit_code == 3
