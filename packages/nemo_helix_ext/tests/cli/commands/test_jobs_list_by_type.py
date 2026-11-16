# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo jobs list --type`` — wire-level tests through the real ``nemo`` app.

``--type`` resolves plugins and job types from the ``nemo.jobs`` entry points (faked here) and lists
through each job type's own route instead of the core Jobs list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
from click.testing import Result
from nemo_helix_ext.cli.app import app
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_plugin import discovery
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.job import NemoJob
from pydantic import BaseModel
from typer.testing import CliRunner

BUILD_PATH = "/apis/demo/v2/workspaces/default/jobs/build"
TRAIN_PATH = "/apis/demo/v2/workspaces/default/train/jobs"


class _Spec(BaseModel):
    size: int = 1


class _BuildJob(NemoJob):
    name = "build"
    spec_schema = _Spec


class _TrainJob(NemoJob):
    name = "train"
    spec_schema = _Spec
    job_collection_path = "/train/jobs"


@dataclass
class _EntryPoint:
    name: str
    loaded: object

    def load(self) -> object:
        return self.loaded


@dataclass
class Recorder:
    routes: dict[str, list[httpx.Response]] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        queue = self.routes.get(request.url.path)
        if not queue:
            raise AssertionError(f"unexpected request {request.method} {request.url}")
        return queue.pop(0)

    def on(self, path: str, *jobs: dict[str, Any]) -> Recorder:
        body = {
            "data": list(jobs),
            "pagination": {
                "page": 1,
                "page_size": 10,
                "current_page_size": len(jobs),
                "total_pages": 1,
                "total_results": len(jobs),
            },
        }
        self.routes.setdefault(path, []).append(httpx.Response(200, json=body))
        return self


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.touch()
    monkeypatch.setenv("NHX_CONFIG_FILE", str(config_file))
    monkeypatch.delenv("NHX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("NHX_WORKSPACE", raising=False)


@pytest.fixture(autouse=True)
def _job_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = {
        "demo.build": _EntryPoint("demo.build", _BuildJob),
        "demo.train": _EntryPoint("demo.train", _TrainJob),
    }
    original = discovery.discover_entry_points

    def _discover(group: str) -> Any:
        return fake if group == "nemo.jobs" else original(group)

    monkeypatch.setattr(discovery, "discover_entry_points", _discover)


def _invoke(recorder: Recorder, *args: str) -> Result:
    client = NemoClient(
        base_url="http://test", workspace="default", http_client=httpx.Client(transport=httpx.MockTransport(recorder))
    )
    state = CLIContext(overrides={"base_url": "http://test", "output_format": "json"}, _client=client)
    return CliRunner().invoke(app, ["jobs", "list", *args], obj=state)


def _job(name: str, created_at: str) -> dict[str, Any]:
    return {"name": name, "workspace": "default", "created_at": created_at, "status": "active", "spec": {}}


def test_type_plugin_lists_every_job_type_through_its_own_route() -> None:
    recorder = (
        Recorder().on(BUILD_PATH, _job("b1", "2026-01-01T00:00:00Z")).on(TRAIN_PATH, _job("t1", "2026-01-02T00:00:00Z"))
    )

    result = _invoke(recorder, "--type", "demo")

    assert result.exit_code == 0, result.output
    assert [r.url.path for r in recorder.requests] == [BUILD_PATH, TRAIN_PATH]
    assert [(job["name"], job["type"]) for job in json.loads(result.stdout)["data"]] == [
        ("t1", "demo.train"),
        ("b1", "demo.build"),
    ]


def test_type_job_passes_filters_page_and_sort() -> None:
    recorder = Recorder().on(BUILD_PATH)

    result = _invoke(
        recorder,
        "-t",
        "demo.build",
        "--filter.name",
        "b1",
        "--filter.status",
        "active",
        "--filter.status",
        "pending",
        "--page",
        "2",
        "--page-size",
        "5",
        "--sort",
        "updated_at",
    )

    assert result.exit_code == 0, result.output
    query = parse_qs(recorder.requests[0].url.query.decode())
    assert json.loads(query.pop("filter")[0]) == {"name": "b1", "status": {"$in": ["active", "pending"]}}
    assert query == {"page": ["2"], "page_size": ["5"], "sort": ["updated_at"]}


def test_type_single_status_is_sent_as_a_value() -> None:
    recorder = Recorder().on(BUILD_PATH)

    result = _invoke(recorder, "-t", "demo.build", "--filter.status", "active")

    assert result.exit_code == 0, result.output
    query = parse_qs(recorder.requests[0].url.query.decode())
    assert json.loads(query["filter"][0]) == {"status": "active"}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--filter.source", "x"], "--filter.source"),
        (["--filter", '{"source": "x"}'], "a 'source' filter can't be combined"),
        (["--sort", "-source"], "can't be sorted by source"),
        (["--page", "2"], "--page needs exactly one job type"),
    ],
)
def test_type_rejects_options_it_cannot_honor(args: list[str], message: str) -> None:
    recorder = Recorder()

    result = _invoke(recorder, "--type", "demo", *args)

    assert result.exit_code == 2, result.output
    assert message in result.output
    assert recorder.requests == []


def test_unknown_type_lists_plugins_with_jobs() -> None:
    recorder = Recorder()

    result = _invoke(recorder, "--type", "nope")

    assert result.exit_code == 2, result.output
    assert "Plugins with jobs: demo" in result.output
    assert recorder.requests == []


def test_type_code_output_renders_plugin_jobs_client_call() -> None:
    recorder = Recorder()

    result = _invoke(recorder, "--type", "demo.train", "-f", "code")

    assert result.exit_code == 0, result.output
    assert recorder.requests == []
    assert "PluginJobsClient" in result.stdout
    assert "train/jobs" in result.stdout
