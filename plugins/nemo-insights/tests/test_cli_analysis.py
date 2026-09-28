# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo insights analysis`` — periodic analysis opt-in state."""

from __future__ import annotations

import json

import pytest
import typer
from _insights_cli import CONFIGURED_DEFAULT, CONFIGURED_FAST, FakeInsightsAPI, config_json, page_json
from click.testing import Result
from nemo_insights_plugin import cli
from typer.testing import CliRunner

runner = CliRunner()

CONFIGS = "/analysis-configs"
CONFIG = "/analysis-configs/demo-agent"


def _invoke(app: typer.Typer, *args: str) -> Result:
    return runner.invoke(app, ["insights", "analysis", *args])


@pytest.mark.usefixtures("configured_models")
def test_enable_sends_the_locally_configured_models(app: typer.Typer, api: FakeInsightsAPI) -> None:
    """The Platform process cannot read the operator's config, so the CLI supplies it."""
    api.on("POST", f"{CONFIG}/enable", config_json())

    result = _invoke(app, "enable", "--agent", "demo-agent")

    assert result.exit_code == 0, result.output
    (request,) = api.requests
    assert request.body == {"default_model": CONFIGURED_DEFAULT, "fast_model": CONFIGURED_FAST}
    assert json.loads(result.stdout)["id"] == "config-id-1"


def test_enable_checks_local_models_before_sending(
    app: typer.Typer, api: FakeInsightsAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing() -> None:
        raise ValueError("No default model is configured. Run `nemo setup` and select agent models.")

    monkeypatch.setattr(cli, "configured_model_refs", missing)

    result = _invoke(app, "enable", "--agent", "demo-agent")

    assert result.exit_code == 1
    assert "Error: No default model is configured" in result.output
    assert "--default-model and --fast-model" in " ".join(result.output.split())
    assert "Traceback" not in result.output
    assert api.requests == []


def test_enable_uses_explicit_model_refs_without_reading_config(
    app: typer.Typer, api: FakeInsightsAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing() -> None:
        raise ValueError("No default model is configured. Run `nemo setup` and select agent models.")

    monkeypatch.setattr(cli, "configured_model_refs", missing)
    api.on("POST", f"{CONFIG}/enable", config_json())

    result = _invoke(
        app, "enable", "--agent", "demo-agent", "--default-model", "default/big", "--fast-model", "default/small"
    )

    assert result.exit_code == 0, result.output
    assert api.requests[0].body == {"default_model": "default/big", "fast_model": "default/small"}


@pytest.mark.usefixtures("configured_models")
def test_enable_code_prints_the_typed_client_call(app: typer.Typer, api: FakeInsightsAPI) -> None:
    result = _invoke(app, "enable", "--agent", "demo-agent", "-f", "code")

    assert result.exit_code == 0, result.output
    assert api.requests == []
    assert "client.enable_analysis_config(" in result.stdout
    assert "EnableAnalysisConfigRequest(" in result.stdout


def test_disable_posts_to_the_disable_route(app: typer.Typer, api: FakeInsightsAPI) -> None:
    api.on("POST", f"{CONFIG}/disable", config_json(enabled=False))

    result = _invoke(app, "disable", "--agent", "demo-agent", "-f", "yaml")

    assert result.exit_code == 0, result.output
    assert [(request.method, request.path) for request in api.requests] == [("POST", f"{CONFIG}/disable")]
    assert "enabled: false" in result.stdout


def test_status_for_one_agent_reads_its_config(app: typer.Typer, api: FakeInsightsAPI) -> None:
    api.on("GET", CONFIG, config_json())

    result = _invoke(app, "status", "--agent", "demo-agent")

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["agent"] == "demo-agent"


def test_status_without_an_agent_lists_every_config(app: typer.Typer, api: FakeInsightsAPI) -> None:
    api.on("GET", CONFIGS, page_json([config_json()], page_size=100))

    result = _invoke(app, "status", "-f", "table", "--no-truncate")

    assert result.exit_code == 0, result.output
    assert api.requests[0].params == {"page": "1", "page_size": "100", "sort": "-created_at"}
    assert "demo-agent" in result.stdout
    assert CONFIGURED_DEFAULT in result.stdout
