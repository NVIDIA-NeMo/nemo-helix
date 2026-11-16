# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo <plugin> list-jobs`` — wire-level tests of the generated command and its building blocks.

The command runs against a stand-in ``nemo`` CLI state whose typed client talks to a recording
``httpx.MockTransport``, so each test pins the requests sent and what was printed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar
from urllib.parse import parse_qs

import click
import httpx
import pytest
import typer
from click.testing import Result
from nemo_helix_plugin.cli_list_jobs import (
    JobCollection,
    add_list_jobs_command,
    has_command,
    job_collections,
    resolve_job_types,
    select_job_types,
)
from nemo_helix_plugin.cli_warnings import collect_warnings
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.commands import add_job_commands
from nemo_helix_plugin.job import NemoJob
from pydantic import BaseModel
from typer.testing import CliRunner

BASE_URL = "http://nhx.test"
WORKSPACE = "team-a"

ClientT = TypeVar("ClientT", bound=NemoClient)


class _Spec(BaseModel):
    size: int = 1


class _BuildJob(NemoJob):
    name = "build"
    description = "Build a thing."
    spec_schema = _Spec


class _TrainJob(NemoJob):
    name = "train.jobs"
    description = "Train on a custom collection path."
    spec_schema = _Spec
    job_collection_path: ClassVar[str | None] = "/train/jobs"


class _NotAJob:
    """A post-run action or function: registered under ``nemo.jobs`` but with no job collection."""


JOBS: dict[str, object] = {"demo.build": _BuildJob, "demo.train.jobs": _TrainJob, "demo.noop": _NotAJob}
BUILD_PATH = f"/apis/demo/v2/workspaces/{WORKSPACE}/jobs/build"
TRAIN_PATH = f"/apis/demo/v2/workspaces/{WORKSPACE}/train/jobs"


def _job(name: str, created_at: str | None, **extra: Any) -> dict[str, Any]:
    return {"name": name, "workspace": WORKSPACE, "created_at": created_at, "status": "completed", "spec": {}, **extra}


def _page(*jobs: dict[str, Any], page: int = 1, page_size: int = 10, total_pages: int = 1, total: int | None = None):
    return {
        "data": list(jobs),
        "pagination": {
            "page": page,
            "page_size": page_size,
            "current_page_size": len(jobs),
            "total_pages": total_pages,
            "total_results": len(jobs) if total is None else total,
        },
        "sort": "-created_at",
    }


@dataclass
class _Api:
    """Answers each request from ``routes[path]``, a queue of responses, and records it."""

    routes: dict[str, list[httpx.Response]] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        queue = self.routes.get(request.url.path)
        if not queue:
            return httpx.Response(404, json={"detail": "Not Found"})
        return queue.pop(0)

    def on(self, path: str, *bodies: dict[str, Any]) -> _Api:
        self.routes.setdefault(path, []).extend(httpx.Response(200, json=body) for body in bodies)
        return self

    def fail(self, path: str, status: int) -> _Api:
        self.routes.setdefault(path, []).append(httpx.Response(status, json={"detail": "nope"}))
        return self

    def query(self, index: int = 0) -> dict[str, list[str]]:
        return parse_qs(self.requests[index].url.query.decode())


@dataclass
class _State:
    """The slice of the ``nemo`` CLI state (``CLIState``) these commands use."""

    api: _Api
    output_format: str = "json"

    def typed_client(self, client_cls: type[ClientT], timeout: float = 60.0) -> ClientT:
        http_client = httpx.Client(transport=httpx.MockTransport(self.api))
        return client_cls.from_client(NemoClient(base_url=BASE_URL, http_client=http_client))

    def get_workspace(self) -> str | None:
        return WORKSPACE

    def get_base_url(self, default: str | None = None) -> str | None:
        return BASE_URL

    def get_output_format(self, override: str | None = None, *, apply_non_tty_default: bool = True) -> str:
        return override or self.output_format

    def get_no_truncate(self, override: bool | None = None) -> bool:
        return True

    def get_timestamp_format(self, override: str | None = None) -> str:
        return "iso8601"


@pytest.fixture
def app() -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def _root() -> None:
        """Force subcommand dispatch."""

    add_list_jobs_command(app, JOBS)
    return app


def _invoke(app: typer.Typer, api: _Api, *args: str, output_format: str = "json") -> Result:
    return CliRunner().invoke(app, ["list-jobs", *args], obj=_State(api, output_format))


def _names(result: Result) -> list[str]:
    return [job["name"] for job in json.loads(result.stdout)["data"]]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def test_job_collections_skips_entries_that_are_not_jobs() -> None:
    assert job_collections(JOBS) == [
        JobCollection(api="demo", name="build", path="jobs/build"),
        JobCollection(api="demo", name="train.jobs", path="train/jobs"),
    ]


def test_job_collections_skips_keys_without_a_plugin_prefix() -> None:
    """``add_job_commands`` callers sometimes key jobs by bare name; that must not break their CLI."""
    assert job_collections({"build": _BuildJob, ".build": _BuildJob}) == []

    app = typer.Typer()
    add_job_commands(app, {"build": _BuildJob})
    assert not has_command(app, "list-jobs")


def test_job_collection_type_is_plugin_and_job_name() -> None:
    collection = JobCollection.for_job("demo.train.jobs", _TrainJob)

    assert (collection.name, collection.job_type) == ("train.jobs", "demo.train.jobs")


def test_job_collection_type_follows_the_job_name_not_the_key() -> None:
    """A key registered under another name keeps the job's own name, which is its submit command."""
    assert JobCollection.for_job("demo.make", _BuildJob).job_type == "demo.build"


def test_job_collection_rejects_key_without_plugin_prefix() -> None:
    with pytest.raises(ValueError, match="'<plugin>.<job>'"):
        JobCollection.for_job("build", _BuildJob)


def test_select_job_types_accepts_job_names_and_full_keys_without_duplicates() -> None:
    collections = job_collections(JOBS)
    assert select_job_types(collections, None) == collections
    assert [c.job_type for c in select_job_types(collections, ["train.jobs", "demo.train.jobs", "build"])] == [
        "demo.train.jobs",
        "demo.build",
    ]


def test_select_job_types_rejects_unknown_name() -> None:
    with pytest.raises(click.BadParameter, match="Unknown job type 'nope'. Choose from: build, train.jobs"):
        select_job_types(job_collections(JOBS), ["nope"])


def test_has_command_sees_commands_and_groups() -> None:
    app = typer.Typer()

    @app.command()
    def list_jobs() -> None: ...

    app.add_typer(typer.Typer(), name="other")
    assert has_command(app, "list-jobs")
    assert has_command(app, "other")
    assert not has_command(app, "missing")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_add_job_commands_registers_list_jobs_in_jobs_panel() -> None:
    app = typer.Typer()
    add_job_commands(app, {"demo.build": _BuildJob})

    (info,) = [c for c in app.registered_commands if c.name == "list-jobs"]
    assert info.rich_help_panel == "Jobs"


def test_add_list_jobs_command_keeps_a_plugin_supplied_list_jobs() -> None:
    app = typer.Typer()

    @app.command("list-jobs")
    def custom() -> None:
        """The plugin's own."""

    add_list_jobs_command(app, JOBS)

    assert [c.callback for c in app.registered_commands if c.name == "list-jobs"] == [custom]


def test_add_list_jobs_command_skips_plugin_without_job_collections() -> None:
    app = typer.Typer()
    add_list_jobs_command(app, {"demo.noop": _NotAJob})
    assert app.registered_commands == []


def test_help_lists_the_plugins_job_types(app: typer.Typer) -> None:
    result = CliRunner().invoke(app, ["list-jobs", "--help"], obj=_State(_Api()))
    assert result.exit_code == 0, result.output
    assert "build, train.jobs" in result.stdout


# ---------------------------------------------------------------------------
# One job type: the server's own pagination
# ---------------------------------------------------------------------------


def test_single_type_lists_its_collection_and_tags_each_job(app: typer.Typer) -> None:
    api = _Api().on(BUILD_PATH, _page(_job("b1", "2026-01-02T00:00:00Z"), total_pages=3, total=21))

    result = _invoke(app, api, "--type", "build", "--page", "2", "--page-size", "1", "--sort", "updated_at")

    assert result.exit_code == 0, result.output
    assert [r.url.path for r in api.requests] == [BUILD_PATH]
    assert api.query() == {"page": ["2"], "page_size": ["1"], "sort": ["updated_at"]}
    body = json.loads(result.stdout)
    assert body["data"][0]["name"] == "b1"
    assert body["data"][0]["type"] == "demo.build"
    assert body["pagination"]["total_results"] == 21


def test_single_type_sends_merged_filter(app: typer.Typer) -> None:
    api = _Api().on(BUILD_PATH, _page())

    result = _invoke(
        app, api, "-t", "build", "--filter", '{"spec.size": 2}', "--filter.status", "active", "--filter.name", "b1"
    )

    assert result.exit_code == 0, result.output
    assert json.loads(api.query()["filter"][0]) == {"spec.size": 2, "status": "active", "name": "b1"}


def test_repeated_status_is_sent_as_in(app: typer.Typer) -> None:
    api = _Api().on(BUILD_PATH, _page())

    result = _invoke(app, api, "-t", "build", "--filter.status", "active", "--filter.status", "pending")

    assert result.exit_code == 0, result.output
    assert json.loads(api.query()["filter"][0]) == {"status": {"$in": ["active", "pending"]}}


def test_text_filter_expression_is_forwarded_as_is(app: typer.Typer) -> None:
    api = _Api().on(BUILD_PATH, _page())

    result = _invoke(app, api, "-t", "build", "--filter", 'name~"nightly"')

    assert result.exit_code == 0, result.output
    assert api.query()["filter"] == ['name~"nightly"']


def test_single_type_all_pages_follows_server_pages(app: typer.Typer) -> None:
    api = _Api().on(
        BUILD_PATH,
        _page(_job("b1", "2026-01-02T00:00:00Z"), page_size=1, total_pages=2, total=2),
        _page(_job("b2", "2026-01-01T00:00:00Z"), page=2, page_size=1, total_pages=2, total=2),
    )

    result = _invoke(app, api, "-t", "build", "--all-pages")

    assert result.exit_code == 0, result.output
    assert _names(result) == ["b1", "b2"]
    assert len(api.requests) == 2


def test_single_type_warns_when_more_pages_exist(app: typer.Typer) -> None:
    api = _Api().on(BUILD_PATH, _page(_job("b1", "2026-01-02T00:00:00Z"), total_pages=2, total=11))

    @collect_warnings
    def _run() -> Result:
        return _invoke(app, api, "-t", "build")

    result = _run()

    assert result.exit_code == 0, result.output
    assert "More pages of results are available" in result.stderr


# ---------------------------------------------------------------------------
# Several job types: merged client-side
# ---------------------------------------------------------------------------


def test_all_types_are_merged_newest_first_and_capped_at_page_size(app: typer.Typer) -> None:
    api = (
        _Api()
        .on(BUILD_PATH, _page(_job("b-new", "2026-01-03T00:00:00Z"), _job("b-old", "2026-01-01T00:00:00Z"), total=5))
        .on(TRAIN_PATH, _page(_job("t-mid", "2026-01-02T00:00:00Z"), total=1))
    )

    result = _invoke(app, api, "--page-size", "2")

    assert result.exit_code == 0, result.output
    assert [r.url.path for r in api.requests] == [BUILD_PATH, TRAIN_PATH]
    assert api.query(0) == {"page_size": ["2"]} and api.query(1) == {"page_size": ["2"]}
    body = json.loads(result.stdout)
    assert [(job["name"], job["type"]) for job in body["data"]] == [
        ("b-new", "demo.build"),
        ("t-mid", "demo.train.jobs"),
    ]
    assert body["pagination"] == {
        "page": 1,
        "page_size": 2,
        "current_page_size": 2,
        "total_pages": 3,
        "total_results": 6,
    }
    assert body["sort"] == "-created_at"


def test_merge_honors_ascending_sort_and_puts_missing_values_last(app: typer.Typer) -> None:
    api = (
        _Api()
        .on(BUILD_PATH, _page(_job("b2", None, updated_at="2026-01-02T00:00:00Z"), _job("nil", None)))
        .on(TRAIN_PATH, _page(_job("t1", None, updated_at="2026-01-01T00:00:00Z")))
    )

    result = _invoke(app, api, "--sort", "updated_at")

    assert result.exit_code == 0, result.output
    assert _names(result) == ["t1", "b2", "nil"]


def test_merge_compares_naive_and_aware_timestamps(app: typer.Typer) -> None:
    api = (
        _Api()
        .on(BUILD_PATH, _page(_job("naive", "2026-01-02T00:00:00")))
        .on(TRAIN_PATH, _page(_job("aware", "2026-01-03T00:00:00+00:00")))
    )

    result = _invoke(app, api)

    assert result.exit_code == 0, result.output
    assert _names(result) == ["aware", "naive"]


def test_merge_without_page_size_uses_the_servers_page_size(app: typer.Typer) -> None:
    builds = [_job(f"b{i}", f"2026-01-{10 + i:02d}T00:00:00Z") for i in range(2)]
    api = _Api().on(BUILD_PATH, _page(*builds, page_size=2)).on(TRAIN_PATH, _page(_job("t", "2026-02-01T00:00:00Z")))

    result = _invoke(app, api)

    assert result.exit_code == 0, result.output
    assert _names(result) == ["t", "b1"]


def test_all_pages_across_types_fetches_everything(app: typer.Typer) -> None:
    api = (
        _Api()
        .on(
            BUILD_PATH,
            _page(_job("b1", "2026-01-04T00:00:00Z"), page_size=1, total_pages=2, total=2),
            _page(_job("b2", "2026-01-01T00:00:00Z"), page=2, page_size=1, total_pages=2, total=2),
        )
        .on(TRAIN_PATH, _page(_job("t1", "2026-01-02T00:00:00Z")))
    )

    result = _invoke(app, api, "--all-pages")

    assert result.exit_code == 0, result.output
    assert _names(result) == ["b1", "t1", "b2"]
    body = json.loads(result.stdout)
    assert body["pagination"]["total_results"] == 3
    assert body["sort"] == "-created_at"


def test_all_pages_across_types_reports_the_requested_sort(app: typer.Typer) -> None:
    api = (
        _Api()
        .on(BUILD_PATH, _page(_job("b1", "2026-01-02T00:00:00Z")))
        .on(TRAIN_PATH, _page(_job("t1", "2026-01-01T00:00:00Z")))
    )

    result = _invoke(app, api, "--all-pages", "--sort", "created_at")

    assert result.exit_code == 0, result.output
    assert _names(result) == ["t1", "b1"]
    assert json.loads(result.stdout)["sort"] == "created_at"


def test_page_number_needs_one_type(app: typer.Typer) -> None:
    api = _Api()

    result = _invoke(app, api, "--page", "2")

    assert result.exit_code == 2
    assert "--page needs exactly one job type" in result.output
    assert api.requests == []


def test_missing_collection_is_skipped_with_a_warning(app: typer.Typer) -> None:
    api = _Api().on(TRAIN_PATH, _page(_job("t1", "2026-01-02T00:00:00Z")))

    @collect_warnings
    def _run() -> Result:
        return _invoke(app, api)

    result = _run()

    assert result.exit_code == 0, result.output
    assert _names(result) == ["t1"]
    assert "Skipping job type 'demo.build'" in result.stderr


def test_all_collections_missing_is_an_api_error(app: typer.Typer) -> None:
    result = _invoke(app, _Api())

    assert result.exit_code == 3, result.output


def test_forbidden_collection_is_skipped_with_a_warning(app: typer.Typer) -> None:
    """Job types can be authorized separately; list the ones the caller may read."""
    api = _Api().fail(BUILD_PATH, 403).on(TRAIN_PATH, _page(_job("t1", "2026-01-02T00:00:00Z")))

    @collect_warnings
    def _run() -> Result:
        return _invoke(app, api)

    result = _run()

    assert result.exit_code == 0, result.output
    assert _names(result) == ["t1"]
    assert "Skipping job type 'demo.build': you don't have permission to list its jobs." in result.stderr


def test_every_collection_forbidden_or_missing_is_an_error(app: typer.Typer) -> None:
    result = _invoke(app, _Api().fail(BUILD_PATH, 403))

    assert result.exit_code != 0, result.output
    assert result.stdout == ""


def test_forbidden_single_type_is_an_error(app: typer.Typer) -> None:
    result = _invoke(app, _Api().fail(BUILD_PATH, 403), "-t", "build")

    assert result.exit_code != 0, result.output
    assert result.stdout == ""


def test_server_error_on_one_collection_fails_the_listing(app: typer.Typer) -> None:
    """A partial list after a server error would look complete, so it fails instead."""
    api = _Api().fail(BUILD_PATH, 500).on(TRAIN_PATH, _page(_job("t1", "2026-01-02T00:00:00Z")))

    result = _invoke(app, api)

    assert result.exit_code != 0, result.output
    assert result.stdout == ""


def test_unknown_type_is_a_usage_error(app: typer.Typer) -> None:
    api = _Api()

    result = _invoke(app, api, "--type", "nope")

    assert result.exit_code == 2
    assert "Unknown job type 'nope'" in result.output
    assert api.requests == []


def test_table_output_shows_type_column(app: typer.Typer) -> None:
    api = _Api().on(BUILD_PATH, _page(_job("b1", "2026-01-02T00:00:00Z")))

    result = _invoke(app, api, "-t", "build", output_format="table")

    assert result.exit_code == 0, result.output
    assert "demo.build" in result.stdout


# ---------------------------------------------------------------------------
# --output-format code
# ---------------------------------------------------------------------------


def test_code_output_for_one_type_sends_nothing_and_prints_typed_client_call(app: typer.Typer) -> None:
    api = _Api()

    result = _invoke(app, api, "-t", "train.jobs", "--filter.status", "active", "--all-pages", output_format="code")

    assert result.exit_code == 0, result.output
    assert api.requests == []
    assert "PluginJobsClient" in result.stdout
    assert "list_plugin_jobs(" in result.stdout
    assert "api='demo'" in result.stdout or 'api="demo"' in result.stdout
    assert "train/jobs" in result.stdout
    code = result.stdout.strip().removeprefix("```python").removesuffix("```")
    compile(code, "<snippet>", "exec")


def test_code_output_needs_one_type(app: typer.Typer) -> None:
    api = _Api()

    result = _invoke(app, api, output_format="code")

    assert result.exit_code == 2
    assert "pass a single --type" in result.output
    assert api.requests == []


# ---------------------------------------------------------------------------
# nemo jobs list --type resolution
# ---------------------------------------------------------------------------


@dataclass
class _EntryPoint:
    name: str
    loaded: object

    def load(self) -> object:
        if isinstance(self.loaded, Exception):
            raise self.loaded
        return self.loaded


@pytest.fixture
def entry_points(monkeypatch: pytest.MonkeyPatch) -> dict[str, _EntryPoint]:
    entries = {
        "demo.build": _EntryPoint("demo.build", _BuildJob),
        "demo.train.jobs": _EntryPoint("demo.train.jobs", _TrainJob),
        "demo.noop": _EntryPoint("demo.noop", _NotAJob),
        "other.broken": _EntryPoint("other.broken", ImportError("missing dependency")),
        "other.build": _EntryPoint("other.build", _BuildJob),
    }
    monkeypatch.setattr("nemo_helix_plugin.discovery.discover_entry_points", lambda group: entries)
    return entries


@pytest.mark.usefixtures("entry_points")
def test_resolve_plugin_selects_its_job_types_only() -> None:
    assert [c.job_type for c in resolve_job_types(["demo"])] == ["demo.build", "demo.train.jobs"]


@pytest.mark.usefixtures("entry_points")
def test_resolve_exact_job_type_and_dedupes() -> None:
    assert [c.job_type for c in resolve_job_types(["demo.build", "demo", "other.build"])] == [
        "demo.build",
        "demo.train.jobs",
        "other.build",
    ]


@pytest.mark.usefixtures("entry_points")
def test_resolve_unknown_type_lists_plugins() -> None:
    with pytest.raises(
        click.BadParameter, match="No installed plugin or job type 'nope'. Plugins with jobs: demo, other"
    ):
        resolve_job_types(["nope"])


@pytest.mark.usefixtures("entry_points")
def test_resolve_exact_entry_that_is_not_a_job_is_rejected() -> None:
    with pytest.raises(
        click.BadParameter,
        match=re.escape("Unknown job type 'demo.noop'. demo job types: demo.build, demo.train.jobs."),
    ):
        resolve_job_types(["demo.noop"])


@pytest.mark.usefixtures("entry_points")
def test_resolve_unknown_job_type_names_entries_that_failed_to_load() -> None:
    with pytest.raises(
        click.BadParameter,
        match=re.escape("other job types: other.build. Failed to load: other.broken (missing dependency)."),
    ):
        resolve_job_types(["other.broken"])


def test_resolve_job_type_matches_the_job_name_not_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = {"demo.make": _EntryPoint("demo.make", _BuildJob)}
    monkeypatch.setattr("nemo_helix_plugin.discovery.discover_entry_points", lambda group: entries)

    assert [c.job_type for c in resolve_job_types(["demo.build"])] == ["demo.build"]
    with pytest.raises(click.BadParameter, match="Unknown job type 'demo.make'"):
        resolve_job_types(["demo.make"])


@pytest.mark.usefixtures("entry_points")
def test_resolve_plugin_skips_entries_that_fail_to_load(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr("nemo_helix_plugin.cli_list_jobs.add_warning", warnings.append)

    assert [c.job_type for c in resolve_job_types(["other"])] == ["other.build"]
    assert warnings == ["Skipping job type 'other.broken': it failed to load (missing dependency)."]
