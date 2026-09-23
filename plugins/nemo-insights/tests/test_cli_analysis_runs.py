# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo insights analysis-runs`` — submit and inspect analysis runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer
from nemo_helix_plugin.nooa_model_client import ConfiguredModelRefs
from nemo_insights_plugin import cli
from nemo_insights_plugin.entities import AnalysisRun
from nemo_insights_plugin.schema import AnalysisRunPage, AnalysisRunResponse
from nemo_insights_plugin.sdk_resources.analysis_runs import AnalysisRunTimeoutError
from typer.testing import CliRunner

runner = CliRunner()

RUN_NAME = "insights-run-0123456789abcdef0123456789abcdef"
CONFIGURED_DEFAULT = "default/configured-big"
CONFIGURED_FAST = "default/configured-small"


def _run(**overrides: Any) -> AnalysisRun:
    return AnalysisRun(
        name=overrides.pop("name", RUN_NAME),
        workspace=overrides.pop("workspace", "default"),
        agent=overrides.pop("agent", "demo-agent"),
        **overrides,
    )


def _response(status: str | None = "created", **overrides: Any) -> AnalysisRunResponse:
    job = None if status is None else {"name": RUN_NAME, "status": status}
    return AnalysisRunResponse(run=_run(**overrides), job=job)


class _StubAnalysisRuns:
    """Stands in for ``client.insights.analysis_runs``, recording each call."""

    def __init__(
        self,
        *,
        create_response: AnalysisRunResponse | None = None,
        get_response: AnalysisRunResponse | None = None,
        wait_response: AnalysisRunResponse | None = None,
        wait_error: Exception | None = None,
    ) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.list_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.wait_calls: list[dict[str, Any]] = []
        self._create_response = create_response or _response()
        self._get_response = get_response or _response(status="completed")
        self._wait_response = wait_response or _response(status="completed")
        self._wait_error = wait_error

    async def create(self, **kwargs: Any) -> AnalysisRunResponse:
        self.create_calls.append(kwargs)
        return self._create_response

    async def list_runs(self, **kwargs: Any) -> AnalysisRunPage:
        self.list_calls.append(kwargs)
        return AnalysisRunPage(data=[_run()], pagination=None, sort="-created_at", filter=None)

    async def get(self, **kwargs: Any) -> AnalysisRunResponse:
        self.get_calls.append(kwargs)
        return self._get_response

    async def wait(self, **kwargs: Any) -> AnalysisRunResponse:
        self.wait_calls.append(kwargs)
        if self._wait_error is not None:
            raise self._wait_error
        on_status = kwargs.get("on_status")
        if on_status is not None:
            on_status(self._wait_response.job_status)
        return self._wait_response


class _StubClient:
    def __init__(self, runs: _StubAnalysisRuns) -> None:
        self.insights = SimpleNamespace(analysis_runs=runs)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def app() -> typer.Typer:
    return cli.InsightsCLI().get_cli()


@pytest.fixture(autouse=True)
def configured_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "configured_model_refs",
        lambda: ConfiguredModelRefs(default=CONFIGURED_DEFAULT, fast=CONFIGURED_FAST),
    )


def _install_client(monkeypatch: pytest.MonkeyPatch, runs: _StubAnalysisRuns) -> _StubClient:
    client = _StubClient(runs)
    monkeypatch.setattr(cli, "make_client", lambda base_url: client)
    return client


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_submits_a_run_and_prints_it_as_json(app: typer.Typer, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _StubAnalysisRuns()
    client = _install_client(monkeypatch, runs)

    result = runner.invoke(app, ["analysis-runs", "create", "--agent", "demo-agent"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["run"]["name"] == RUN_NAME
    assert runs.create_calls[0]["agent"] == "demo-agent"
    assert runs.create_calls[0]["workspace"] == "default"
    assert client.closed is True


def test_create_falls_back_to_the_configured_model_pair(app: typer.Typer, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Platform process cannot read the operator's config, so the CLI supplies it."""
    runs = _StubAnalysisRuns()
    _install_client(monkeypatch, runs)

    result = runner.invoke(app, ["analysis-runs", "create", "--agent", "demo-agent"])

    assert result.exit_code == 0, result.output
    assert runs.create_calls[0]["default_model"] == CONFIGURED_DEFAULT
    assert runs.create_calls[0]["fast_model"] == CONFIGURED_FAST


def test_explicit_model_refs_win_over_the_configured_pair(app: typer.Typer, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _StubAnalysisRuns()
    _install_client(monkeypatch, runs)

    result = runner.invoke(
        app,
        [
            "analysis-runs",
            "create",
            "--agent",
            "demo-agent",
            "--default-model",
            "default/big",
            "--fast-model",
            "default/small",
        ],
    )

    assert result.exit_code == 0, result.output
    assert runs.create_calls[0]["default_model"] == "default/big"
    assert runs.create_calls[0]["fast_model"] == "default/small"


def test_create_passes_the_requested_read_scope(app: typer.Typer, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _StubAnalysisRuns()
    _install_client(monkeypatch, runs)

    result = runner.invoke(
        app,
        [
            "analysis-runs",
            "create",
            "--agent",
            "demo-agent",
            "--since",
            "2026-08-01T00:00:00+00:00",
            "--evaluation-id",
            "eval-123",
            "--timeout-seconds",
            "60",
        ],
    )

    assert result.exit_code == 0, result.output
    assert runs.create_calls[0]["since"] == datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert runs.create_calls[0]["evaluation_id"] == "eval-123"
    assert runs.create_calls[0]["timeout_seconds"] == 60.0


def test_create_reads_the_ethos_file_and_sends_its_contents(
    app: typer.Typer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--ethos`` takes a path like ``nemo insights analyze``; the API wants the Markdown."""
    runs = _StubAnalysisRuns()
    _install_client(monkeypatch, runs)
    ethos = tmp_path / "ETHOS.md"
    ethos.write_text("# Ethos\n\nBe careful.\n", encoding="utf-8")

    result = runner.invoke(app, ["analysis-runs", "create", "--agent", "demo-agent", "--ethos", str(ethos)])

    assert result.exit_code == 0, result.output
    assert runs.create_calls[0]["ethos"] == "# Ethos\n\nBe careful.\n"


def test_create_without_an_ethos_sends_none(app: typer.Typer, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _StubAnalysisRuns()
    _install_client(monkeypatch, runs)

    result = runner.invoke(app, ["analysis-runs", "create", "--agent", "demo-agent"])

    assert result.exit_code == 0, result.output
    assert runs.create_calls[0]["ethos"] is None


def test_an_unreadable_ethos_fails_before_anything_is_submitted(
    app: typer.Typer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Submitting without it would analyze the agent against no contract at all."""
    runs = _StubAnalysisRuns()
    _install_client(monkeypatch, runs)

    result = runner.invoke(
        app, ["analysis-runs", "create", "--agent", "demo-agent", "--ethos", str(tmp_path / "missing.md")]
    )

    assert result.exit_code == 1
    assert "--ethos" in result.output
    assert runs.create_calls == []


def test_an_empty_ethos_file_fails_before_anything_is_submitted(
    app: typer.Typer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs = _StubAnalysisRuns()
    _install_client(monkeypatch, runs)
    ethos = tmp_path / "ETHOS.md"
    ethos.write_text("   \n", encoding="utf-8")

    result = runner.invoke(app, ["analysis-runs", "create", "--agent", "demo-agent", "--ethos", str(ethos)])

    assert result.exit_code == 1
    assert runs.create_calls == []


def test_a_malformed_since_fails_before_anything_is_submitted(
    app: typer.Typer, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = _StubAnalysisRuns()
    _install_client(monkeypatch, runs)

    result = runner.invoke(app, ["analysis-runs", "create", "--agent", "demo-agent", "--since", "last tuesday"])

    assert result.exit_code == 1
    assert "ISO-8601" in result.output
    assert runs.create_calls == []


def test_create_without_wait_does_not_poll(app: typer.Typer, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _StubAnalysisRuns()
    _install_client(monkeypatch, runs)

    result = runner.invoke(app, ["analysis-runs", "create", "--agent", "demo-agent"])

    assert result.exit_code == 0, result.output
    assert runs.wait_calls == []


def test_create_with_wait_polls_the_run_it_just_created(app: typer.Typer, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _StubAnalysisRuns()
    _install_client(monkeypatch, runs)

    result = runner.invoke(app, ["analysis-runs", "create", "--agent", "demo-agent", "--wait", "--poll-interval", "0"])

    assert result.exit_code == 0, result.output
    assert runs.wait_calls[0]["name"] == RUN_NAME
    assert json.loads(result.stdout)["job"]["status"] == "completed"


def test_wait_exits_non_zero_when_the_job_fails(app: typer.Typer, monkeypatch: pytest.MonkeyPatch) -> None:
    """A finished-but-failed job still prints; only the exit code carries the verdict."""
    runs = _StubAnalysisRuns(wait_response=_response(status="error"))
    _install_client(monkeypatch, runs)

    result = runner.invoke(app, ["analysis-runs", "create", "--agent", "demo-agent", "--wait"])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["job"]["status"] == "error"


def test_wait_timeout_is_reported_as_an_error(app: typer.Typer, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _StubAnalysisRuns(wait_error=AnalysisRunTimeoutError("did not finish within 1.0s"))
    _install_client(monkeypatch, runs)

    result = runner.invoke(app, ["analysis-runs", "create", "--agent", "demo-agent", "--wait"])

    assert result.exit_code == 1
    assert "did not finish" in result.output


# ---------------------------------------------------------------------------
# list and get
# ---------------------------------------------------------------------------


def test_list_prints_a_page_of_runs(app: typer.Typer, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _StubAnalysisRuns()
    _install_client(monkeypatch, runs)

    result = runner.invoke(app, ["analysis-runs", "list", "--agent", "demo-agent"])

    assert result.exit_code == 0, result.output
    assert runs.list_calls[0]["agent"] == "demo-agent"
    assert json.loads(result.stdout)["data"][0]["name"] == RUN_NAME


def test_list_defaults_to_newest_first(app: typer.Typer, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _StubAnalysisRuns()
    _install_client(monkeypatch, runs)

    result = runner.invoke(app, ["analysis-runs", "list"])

    assert result.exit_code == 0, result.output
    assert runs.list_calls[0]["sort"] == "-created_at"


def test_list_forwards_an_explicit_sort(app: typer.Typer, monkeypatch: pytest.MonkeyPatch) -> None:
    """The route accepts a sort, so the CLI has to be able to reach it."""
    runs = _StubAnalysisRuns()
    _install_client(monkeypatch, runs)

    result = runner.invoke(app, ["analysis-runs", "list", "--sort", "created_at"])

    assert result.exit_code == 0, result.output
    assert runs.list_calls[0]["sort"] == "created_at"


def test_get_prints_the_run_and_its_job(app: typer.Typer, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _StubAnalysisRuns()
    _install_client(monkeypatch, runs)

    result = runner.invoke(app, ["analysis-runs", "get", RUN_NAME])

    assert result.exit_code == 0, result.output
    assert runs.get_calls[0]["name"] == RUN_NAME
    assert json.loads(result.stdout)["job"]["status"] == "completed"


def test_get_reports_a_run_whose_job_never_landed(app: typer.Typer, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _StubAnalysisRuns(get_response=_response(status=None))
    _install_client(monkeypatch, runs)

    result = runner.invoke(app, ["analysis-runs", "get", RUN_NAME])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["job"] is None


def test_get_with_wait_polls_instead_of_reading_once(app: typer.Typer, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _StubAnalysisRuns()
    _install_client(monkeypatch, runs)

    result = runner.invoke(app, ["analysis-runs", "get", RUN_NAME, "--wait"])

    assert result.exit_code == 0, result.output
    assert runs.wait_calls[0]["name"] == RUN_NAME
    assert runs.get_calls == []
