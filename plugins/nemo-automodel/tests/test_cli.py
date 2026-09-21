# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from nemo_automodel_plugin.cli.inputs import load_job_json
from nemo_automodel_plugin.contributor import AutomodelContributor
from nemo_automodel_plugin.jobs.jobs import AutomodelJob
from nemo_platform_plugin.scheduler import NemoJobScheduler, submit_path_for
from typer.testing import CliRunner

FIXTURES = Path(__file__).parent / "fixtures"


def test_submit_path_includes_workspace() -> None:
    path = submit_path_for(AutomodelJob, workspace="acme-corp")
    assert path == "/apis/customization/v2/workspaces/acme-corp/automodel/jobs"


def test_load_job_json_validates_fixture() -> None:
    job_path = FIXTURES / "minimal_sft_lora.json"
    spec = json.loads(load_job_json(job_path))
    assert spec["training"]["training_type"] == "sft"
    assert spec["dataset"]["training"] == "default/train-data"


def test_jobs_submit_posts_to_automodel_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    capture: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        capture["method"] = request.method
        capture["url"] = str(request.url)
        capture["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "job-1", "status": "queued"})

    monkeypatch.setattr(
        "nemo_platform_plugin.discovery.discover_jobs",
        lambda: {"customization.automodel.jobs": AutomodelJob},
    )
    scheduler = NemoJobScheduler()
    scheduler.submit_remote(
        AutomodelJob,
        json.loads(load_job_json(FIXTURES / "minimal_sft_lora.json")),
        base_url="https://nmp.test",
        workspace="ws-a",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert capture["method"] == "POST"
    assert capture["url"] == "https://nmp.test/apis/customization/v2/workspaces/ws-a/automodel/jobs"
    assert capture["body"]["spec"]["training"]["training_type"] == "sft"


def test_cli_submit_accepts_job_json_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Contributor CLI: ``submit JOB.json -w ws`` forwards workspace to submit_remote."""
    submitted: dict = {}

    def fake_submit_remote(
        _scheduler,
        job_cls: type,
        spec_data: dict,
        base_url: str | None,
        workspace: str,
        profile: str | None = None,
        options: dict | None = None,
        metadata: dict | None = None,
        http_client: httpx.Client | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict:
        submitted["workspace"] = workspace
        submitted["spec"] = spec_data
        submitted["base_url"] = base_url
        return {"id": "job-99"}

    monkeypatch.setattr(
        "nemo_platform_plugin.commands.NemoJobScheduler.submit_remote",
        fake_submit_remote,
    )
    monkeypatch.setattr(
        "nemo_platform_plugin.discovery.discover_jobs",
        lambda: {"customization.automodel.jobs": AutomodelJob},
    )

    automodel_cli = AutomodelContributor().get_cli()
    runner = CliRunner()
    result = runner.invoke(
        automodel_cli,
        [
            "submit",
            str(FIXTURES / "minimal_sft_lora.json"),
            "--workspace",
            "acme-corp",
            "--base-url",
            "https://nmp.test",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert submitted["workspace"] == "acme-corp"
    assert submitted["base_url"] == "https://nmp.test"
    assert submitted["spec"]["model"] == "default/qwen3-1.7b"


def test_cli_submit_prints_studio_link_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: the wired renderer emits the Studio deep link on submit.

    Proves the ``renderer_cls`` wiring through ``BaseContributor.get_cli`` reaches
    the real Typer CLI for the automodel backend. The ``/status`` probe is mocked
    to report Studio ready so the test makes no network call.
    """

    def fake_submit_remote(
        _scheduler,
        job_cls: type,
        spec_data: dict,
        base_url: str | None,
        workspace: str,
        profile: str | None = None,
        options: dict | None = None,
        metadata: dict | None = None,
        http_client: httpx.Client | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict:
        return {"id": "job-uuid", "name": "my-automodel-run", "status": "queued"}

    monkeypatch.setattr(
        "nemo_platform_plugin.commands.NemoJobScheduler.submit_remote",
        fake_submit_remote,
    )
    monkeypatch.setattr(
        "nemo_platform_plugin.discovery.discover_jobs",
        lambda: {"customization.automodel.jobs": AutomodelJob},
    )
    # Report Studio ready without a real /status call.
    monkeypatch.setattr(
        "nmp.customization_common.cli.renderer.studio_is_available",
        lambda base_url: True,
    )

    automodel_cli = AutomodelContributor().get_cli()
    runner = CliRunner()
    result = runner.invoke(
        automodel_cli,
        [
            "submit",
            str(FIXTURES / "minimal_sft_lora.json"),
            "--workspace",
            "acme-corp",
            "--base-url",
            "https://nmp.test",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    # Link + tracking hints go to stderr; stdout stays pure JSON.
    assert "https://nmp.test/studio/workspaces/acme-corp/customizations/my-automodel-run" in result.stderr
    assert "nemo jobs list --workspace acme-corp" in result.stderr
    assert '"name": "my-automodel-run"' in result.stdout
    assert "View in Studio" not in result.stdout


def test_cli_submit_omits_studio_link_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """When Studio is not a ready service, submit omits the link but keeps CLI hints."""

    def fake_submit_remote(
        _scheduler,
        job_cls: type,
        spec_data: dict,
        base_url: str | None,
        workspace: str,
        profile: str | None = None,
        options: dict | None = None,
        metadata: dict | None = None,
        http_client: httpx.Client | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict:
        return {"id": "job-uuid", "name": "my-automodel-run", "status": "queued"}

    monkeypatch.setattr(
        "nemo_platform_plugin.commands.NemoJobScheduler.submit_remote",
        fake_submit_remote,
    )
    monkeypatch.setattr(
        "nemo_platform_plugin.discovery.discover_jobs",
        lambda: {"customization.automodel.jobs": AutomodelJob},
    )
    monkeypatch.setattr(
        "nmp.customization_common.cli.renderer.studio_is_available",
        lambda base_url: False,
    )

    automodel_cli = AutomodelContributor().get_cli()
    runner = CliRunner()
    result = runner.invoke(
        automodel_cli,
        [
            "submit",
            str(FIXTURES / "minimal_sft_lora.json"),
            "--workspace",
            "acme-corp",
            "--base-url",
            "https://nmp.test",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "View in Studio" not in result.stderr
    assert "nemo jobs list --workspace acme-corp" in result.stderr
    assert '"name": "my-automodel-run"' in result.stdout


def test_cli_help_lists_submit_and_explain_only() -> None:
    automodel_cli = AutomodelContributor().get_cli()
    runner = CliRunner()
    result = runner.invoke(automodel_cli, ["--help"])
    assert result.exit_code == 0
    assert "submit" in result.stdout
    assert "explain" in result.stdout
    assert "run" not in result.stdout


def test_cli_expose_input_and_output_schemas() -> None:
    automodel_cli = AutomodelContributor().get_cli()
    runner = CliRunner()
    result = runner.invoke(automodel_cli, ["explain"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "input_spec_schema" in payload
    assert "spec_schema" in payload
    assert "/automodel/jobs" in payload["endpoint"]
    retrieval = payload["input_spec_schema"]["$defs"]["AutomodelRetrievalSpec"]["properties"]
    assert "do_distributed_inbatch_negative" in retrieval
