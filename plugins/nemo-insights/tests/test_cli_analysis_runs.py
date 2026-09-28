# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo insights analysis-runs`` — submit and inspect analysis runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from _insights_cli import (
    CONFIGURED_DEFAULT,
    CONFIGURED_FAST,
    RUN_NAME,
    FakeInsightsAPI,
    WireState,
    app_with_state,
    page_json,
    run_json,
    run_response_json,
)
from click.testing import Result
from typer.testing import CliRunner

runner = CliRunner()

RUNS = "/analysis-runs"
RUN = f"/analysis-runs/{RUN_NAME}"

pytestmark = pytest.mark.usefixtures("configured_models")


def _invoke(app: typer.Typer, *args: str) -> Result:
    return runner.invoke(app, ["insights", "analysis-runs", *args])


def _create(app: typer.Typer, *args: str) -> Result:
    return runner.invoke(app, ["insights", "analysis-runs", "create", "--agent", "demo-agent", *args])


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_submits_a_run_and_prints_it_as_json(app: typer.Typer, api: FakeInsightsAPI) -> None:
    api.on("POST", RUNS, run_response_json())

    result = _create(app)

    assert result.exit_code == 0, result.output
    (request,) = api.requests
    assert (request.method, request.workspace, request.path) == ("POST", "default", RUNS)
    assert request.body["agent"] == "demo-agent"
    printed = json.loads(result.stdout)
    assert printed["run"]["name"] == RUN_NAME
    assert printed["run"]["created_at"] == "2026-09-01T12:00:00Z"


def test_create_falls_back_to_the_configured_model_pair(app: typer.Typer, api: FakeInsightsAPI) -> None:
    """The Platform process cannot read the operator's config, so the CLI supplies it."""
    api.on("POST", RUNS, run_response_json())

    result = _create(app)

    assert result.exit_code == 0, result.output
    assert api.requests[0].body["default_model"] == CONFIGURED_DEFAULT
    assert api.requests[0].body["fast_model"] == CONFIGURED_FAST


def test_explicit_model_refs_win_over_the_configured_pair(app: typer.Typer, api: FakeInsightsAPI) -> None:
    api.on("POST", RUNS, run_response_json())

    result = _create(app, "--default-model", "default/big", "--fast-model", "default/small")

    assert result.exit_code == 0, result.output
    assert api.requests[0].body["default_model"] == "default/big"
    assert api.requests[0].body["fast_model"] == "default/small"


@pytest.mark.parametrize(
    ("configured_fast", "expected_fast"),
    [("default/configured-fast", "default/configured-fast"), (None, "default/explicit-big")],
)
def test_default_model_flag_needs_no_configured_default(
    app: typer.Typer,
    api: FakeInsightsAPI,
    monkeypatch: pytest.MonkeyPatch,
    configured_fast: str | None,
    expected_fast: str,
) -> None:
    from nemo_insights_plugin import cli

    def no_configured_default() -> None:
        raise ValueError("No default model is configured. Run `nemo setup` and select agent models.")

    monkeypatch.setattr(cli, "configured_model_refs", no_configured_default)
    monkeypatch.setattr(cli, "configured_fast_model", lambda: configured_fast)
    api.on("POST", RUNS, run_response_json())

    result = _create(app, "--default-model", "default/explicit-big")

    assert result.exit_code == 0, result.output
    assert api.requests[0].body["default_model"] == "default/explicit-big"
    assert api.requests[0].body["fast_model"] == expected_fast


def test_create_passes_the_requested_read_scope(app: typer.Typer, api: FakeInsightsAPI) -> None:
    api.on("POST", RUNS, run_response_json())

    result = _create(
        app, "--since", "2026-08-01T00:00:00+00:00", "--evaluation-id", "eval-123", "--timeout-seconds", "60"
    )

    assert result.exit_code == 0, result.output
    body = api.requests[0].body
    assert body["since"] == "2026-08-01T00:00:00Z"
    assert body["evaluation_id"] == "eval-123"
    assert body["timeout_seconds"] == 60.0


def test_create_reads_the_ethos_file_and_sends_its_contents(
    app: typer.Typer, api: FakeInsightsAPI, tmp_path: Path
) -> None:
    """``--ethos`` takes a path like ``nemo insights analyze``; the API wants the Markdown."""
    api.on("POST", RUNS, run_response_json())
    ethos = tmp_path / "ETHOS.md"
    ethos.write_text("# Ethos\n\nBe careful.\n", encoding="utf-8")

    result = _create(app, "--ethos", str(ethos))

    assert result.exit_code == 0, result.output
    assert api.requests[0].body["ethos"] == "# Ethos\n\nBe careful."


def test_create_without_an_ethos_omits_it(app: typer.Typer, api: FakeInsightsAPI) -> None:
    api.on("POST", RUNS, run_response_json())

    result = _create(app)

    assert result.exit_code == 0, result.output
    assert "ethos" not in api.requests[0].body


def test_an_unreadable_ethos_fails_before_anything_is_submitted(
    app: typer.Typer, api: FakeInsightsAPI, tmp_path: Path
) -> None:
    """Submitting without it would analyze the agent against no contract at all."""
    result = _create(app, "--ethos", str(tmp_path / "missing.md"))

    assert result.exit_code == 2
    assert "--ethos" in result.output
    assert api.requests == []


def test_an_empty_ethos_file_fails_before_anything_is_submitted(
    app: typer.Typer, api: FakeInsightsAPI, tmp_path: Path
) -> None:
    ethos = tmp_path / "ETHOS.md"
    ethos.write_text("   \n", encoding="utf-8")

    result = _create(app, "--ethos", str(ethos))

    assert result.exit_code == 2
    assert api.requests == []


def test_a_malformed_since_fails_before_anything_is_submitted(app: typer.Typer, api: FakeInsightsAPI) -> None:
    result = _create(app, "--since", "last tuesday")

    assert result.exit_code == 2
    assert "ISO-8601" in result.output
    assert api.requests == []


def test_missing_configured_models_fail_before_anything_is_submitted(
    app: typer.Typer, api: FakeInsightsAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_insights_plugin import cli

    def missing() -> None:
        raise ValueError("No default model is configured. Run `nemo setup` and select agent models.")

    monkeypatch.setattr(cli, "configured_model_refs", missing)

    result = _create(app)

    assert result.exit_code == 1
    assert "No default model is configured" in result.output
    assert api.requests == []


def test_create_without_wait_does_not_poll(app: typer.Typer, api: FakeInsightsAPI) -> None:
    api.on("POST", RUNS, run_response_json())

    result = _create(app)

    assert result.exit_code == 0, result.output
    assert [request.method for request in api.requests] == ["POST"]


def test_create_with_wait_polls_the_run_it_just_created(app: typer.Typer, api: FakeInsightsAPI) -> None:
    api.on("POST", RUNS, run_response_json())
    api.on("GET", RUN, run_response_json("running"), run_response_json("completed"))

    result = _create(app, "--wait", "--poll-interval", "0")

    assert result.exit_code == 0, result.output
    assert [(request.method, request.path) for request in api.requests] == [
        ("POST", RUNS),
        ("GET", RUN),
        ("GET", RUN),
    ]
    assert json.loads(result.stdout)["job"]["status"] == "completed"
    assert "status: running" in result.stderr


def test_wait_exits_non_zero_when_the_job_fails(app: typer.Typer, api: FakeInsightsAPI) -> None:
    """A finished-but-failed job still prints; only the exit code carries the verdict."""
    api.on("POST", RUNS, run_response_json())
    api.on("GET", RUN, run_response_json("error"))

    result = _create(app, "--wait", "--poll-interval", "0")

    assert result.exit_code == 1
    assert json.loads(result.stdout)["job"]["status"] == "error"


def test_wait_timeout_is_reported_as_an_error(app: typer.Typer, api: FakeInsightsAPI) -> None:
    api.on("POST", RUNS, run_response_json())
    api.on("GET", RUN, run_response_json("running"))

    result = _create(app, "--wait", "--poll-timeout", "0", "--poll-interval", "0")

    assert result.exit_code == 1
    assert "did not finish" in " ".join(result.output.split())


def test_waiting_on_a_run_whose_job_never_landed_fails_fast(app: typer.Typer, api: FakeInsightsAPI) -> None:
    api.on("GET", RUN, run_response_json(None))

    result = _invoke(app, "get", RUN_NAME, "--wait", "--poll-interval", "0")

    assert result.exit_code == 1
    assert "Resubmit it" in result.output
    assert len(api.requests) == 1


def test_create_code_prints_the_typed_client_call_without_sending(app: typer.Typer, api: FakeInsightsAPI) -> None:
    result = _create(app, "-f", "code")

    assert result.exit_code == 0, result.output
    assert api.requests == []
    assert "from nemo_insights_plugin.client import InsightsClient" in result.stdout
    assert "client.create_analysis_run(" in result.stdout
    assert f'default_model="{CONFIGURED_DEFAULT}"' in result.stdout


def test_wait_cannot_be_rendered_as_code(app: typer.Typer, api: FakeInsightsAPI) -> None:
    result = _create(app, "--wait", "-f", "code")

    assert result.exit_code == 2
    assert "--wait cannot be combined with --output-format code" in result.output
    assert api.requests == []


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_sends_the_filters_and_prints_a_page_of_runs(app: typer.Typer, api: FakeInsightsAPI) -> None:
    api.on("GET", RUNS, page_json([run_json()]))

    result = _invoke(app, "list", "--agent", "demo-agent", "--page", "2", "--page-size", "5")

    assert result.exit_code == 0, result.output
    assert api.requests[0].params == {"agent": "demo-agent", "page": "2", "page_size": "5", "sort": "-created_at"}
    printed = json.loads(result.stdout)
    assert printed["data"][0]["name"] == RUN_NAME
    assert printed["data"][0]["id"] == "run-id-1"
    assert printed["data"][0]["created_at"] == "2026-09-01T12:00:00Z"


def test_list_forwards_an_explicit_sort(app: typer.Typer, api: FakeInsightsAPI) -> None:
    """The route accepts a sort, so the CLI has to be able to reach it."""
    api.on("GET", RUNS, page_json([run_json()]))

    result = _invoke(app, "list", "--sort", "created_at")

    assert result.exit_code == 0, result.output
    assert api.requests[0].params["sort"] == "created_at"


@pytest.mark.parametrize("flag", ["--output-format", "--output", "-f"])
def test_list_accepts_the_standard_output_format_flags(app: typer.Typer, api: FakeInsightsAPI, flag: str) -> None:
    api.on("GET", RUNS, page_json([run_json()]))

    result = _invoke(app, "list", flag, "yaml")

    assert result.exit_code == 0, result.output
    assert f"name: {RUN_NAME}" in result.stdout


def test_list_renders_a_table_with_the_default_columns(app: typer.Typer, api: FakeInsightsAPI) -> None:
    api.on("GET", RUNS, page_json([run_json()]))

    result = _invoke(app, "list", "-f", "table", "--no-truncate")

    assert result.exit_code == 0, result.output
    header = result.stdout.splitlines()[1]
    for column in ("name", "agent", "evaluation_id", "created_at"):
        assert column in header
    assert "demo-agent" in result.stdout
    assert "2026-09-01" in result.stdout


def test_list_uses_the_global_output_format_preference(api: FakeInsightsAPI) -> None:
    api.on("GET", RUNS, page_json([run_json()]))

    result = runner.invoke(app_with_state(WireState(api, output_format="csv")), ["insights", "analysis-runs", "list"])

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[0] == "name,agent,evaluation_id,created_at"


def test_list_warns_when_more_pages_exist(app: typer.Typer, api: FakeInsightsAPI) -> None:
    api.on("GET", RUNS, page_json([run_json()], total_pages=2))

    result = _invoke(app, "list")

    assert result.exit_code == 0, result.output
    assert "--all-pages" in result.stderr


def test_list_all_pages_fetches_every_page(app: typer.Typer, api: FakeInsightsAPI) -> None:
    api.on(
        "GET",
        RUNS,
        page_json([run_json(name="run-a")], page=1, total_pages=2),
        page_json([run_json(name="run-b")], page=2, total_pages=2),
    )

    result = _invoke(app, "list", "--all-pages")

    assert result.exit_code == 0, result.output
    assert [request.params["page"] for request in api.requests] == ["1", "2"]
    assert [run["name"] for run in json.loads(result.stdout)["data"]] == ["run-a", "run-b"]


def test_list_code_prints_the_typed_client_call_without_sending(app: typer.Typer, api: FakeInsightsAPI) -> None:
    result = _invoke(app, "list", "--agent", "demo-agent", "-f", "code")

    assert result.exit_code == 0, result.output
    assert api.requests == []
    assert "client.list_analysis_runs(" in result.stdout
    assert '"agent": "demo-agent"' in result.stdout


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


def test_get_prints_the_run_and_its_job(app: typer.Typer, api: FakeInsightsAPI) -> None:
    api.on("GET", RUN, run_response_json("completed"))

    result = _invoke(app, "get", RUN_NAME)

    assert result.exit_code == 0, result.output
    assert [(request.method, request.path) for request in api.requests] == [("GET", RUN)]
    assert json.loads(result.stdout)["job"]["status"] == "completed"


def test_get_reports_a_run_whose_job_never_landed(app: typer.Typer, api: FakeInsightsAPI) -> None:
    api.on("GET", RUN, run_response_json(None))

    result = _invoke(app, "get", RUN_NAME)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["job"] is None


def test_get_with_wait_polls_until_the_job_is_terminal(app: typer.Typer, api: FakeInsightsAPI) -> None:
    api.on("GET", RUN, run_response_json("running"), run_response_json("completed"))

    result = _invoke(app, "get", RUN_NAME, "--wait", "--poll-interval", "0")

    assert result.exit_code == 0, result.output
    assert len(api.requests) == 2
    assert json.loads(result.stdout)["job"]["status"] == "completed"


def test_get_missing_run_is_a_not_found_error(app: typer.Typer, api: FakeInsightsAPI) -> None:
    api.on("GET", RUN, {"detail": "Analysis run not found"}, status=404)

    result = _invoke(app, "get", RUN_NAME)

    assert result.exit_code == 3
    assert "not found" in result.output.lower()


def test_get_code_prints_the_typed_client_call_without_sending(app: typer.Typer, api: FakeInsightsAPI) -> None:
    result = _invoke(app, "get", RUN_NAME, "-f", "code")

    assert result.exit_code == 0, result.output
    assert api.requests == []
    assert "client.get_analysis_run(" in result.stdout
    assert f'name="{RUN_NAME}"' in result.stdout
