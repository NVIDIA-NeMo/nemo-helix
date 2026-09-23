# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Unsloth CLI overrides (``apply_unsloth_job_cli_overrides``).

Pins the post-2026 submit-only contract: generated ``run`` is removed, while
``submit`` accepts a positional ``JOB_JSON`` and delegates to the auto-generated
callback with ``--spec`` set to the validated JSON.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import typer
from nemo_helix_plugin.scheduler import NemoJobScheduler, submit_path_for
from nemo_unsloth_plugin.cli.inputs import apply_unsloth_job_cli_overrides, load_job_json
from nemo_unsloth_plugin.contributor import UnslothContributor
from nemo_unsloth_plugin.jobs.jobs import UnslothJob
from nemo_unsloth_plugin.schema import UnslothJobInput
from typer.testing import CliRunner

FIXTURES = Path(__file__).parent / "fixtures"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _build_app() -> typer.Typer:
    """Build a generated Typer app, then apply the contributor overrides."""
    from nemo_helix_plugin.commands import (
        _add_explain_command,
        _add_submit_command,
    )

    app = typer.Typer(no_args_is_help=True)

    @app.command("run")
    def generated_run() -> None:
        raise AssertionError("Customizer overrides should remove generated run commands")

    scheduler = NemoJobScheduler()
    _add_submit_command(app, UnslothJob, scheduler)
    _add_explain_command(app, UnslothJob, scheduler)
    apply_unsloth_job_cli_overrides(app)
    return app


def _minimal_payload() -> dict[str, object]:
    return {
        "model": {"name": "unsloth/Qwen2.5-0.5B-Instruct"},
        "dataset": {"path": "default/my-dataset"},
        "schedule": {"max_steps": 60},
    }


@dataclass
class _SubmittedCall:
    workspace: str | None = None
    spec: dict[str, object] | None = None
    base_url: str | None = None


class TestLoadJobJson:
    def test_validates_and_returns_canonical_json(self, tmp_path: Path) -> None:
        path = tmp_path / "job.json"
        path.write_text(json.dumps(_minimal_payload()))
        out = load_job_json(path)
        UnslothJobInput.model_validate(json.loads(out))

    def test_invalid_payload_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "job.json"
        path.write_text(json.dumps({"model": {"name": "x"}, "schedule": {"max_steps": 1}}))
        with pytest.raises(Exception):
            load_job_json(path)

    def test_validates_fixture(self) -> None:
        spec = json.loads(load_job_json(FIXTURES / "minimal_unsloth_sft.json"))
        assert spec["training"]["training_type"] == "sft"


class TestSubmitPath:
    def test_submit_path_includes_workspace(self) -> None:
        path = submit_path_for(UnslothJob, workspace="acme-corp")
        assert path == "/apis/customization/v2/workspaces/acme-corp/unsloth/jobs"


class TestCommandRegistration:
    def test_help_lists_submit_and_explain_only(self) -> None:
        app = _build_app()
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        plain = _plain(result.output)
        assert "submit" in plain
        assert "explain" in plain
        assert "run" not in plain


class TestSubmitOverride:
    def test_help_lists_job_json_workspace_and_profile(self) -> None:
        app = _build_app()
        runner = CliRunner()
        result = runner.invoke(app, ["submit", "--help"])
        assert result.exit_code == 0, result.output
        plain = _plain(result.output)
        assert "JOB_JSON" in plain
        assert "--workspace" in plain or "-w " in plain
        assert "--profile" in plain
        assert "--base-url" in plain

    def test_submit_delegates_with_validated_spec(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``submit JOB.json -w ws`` forwards workspace + base-url to submit_remote."""
        submitted = _SubmittedCall()

        def fake_submit_remote(
            _scheduler,
            _job_cls: type,
            spec_data: dict[str, object],
            base_url: str | None,
            workspace: str,
            profile: str | None = None,
            options: dict | None = None,
            metadata: dict | None = None,
            http_client: httpx.Client | None = None,
            headers: dict[str, str] | None = None,
        ) -> dict:
            submitted.workspace = workspace
            submitted.spec = spec_data
            submitted.base_url = base_url
            return {"id": "job-99"}

        monkeypatch.setattr(
            "nemo_helix_plugin.commands.NemoJobScheduler.submit_remote",
            fake_submit_remote,
        )
        monkeypatch.setattr(
            "nemo_helix_plugin.discovery.discover_jobs",
            lambda: {"customization.unsloth.jobs": UnslothJob},
        )

        path = tmp_path / "job.json"
        path.write_text(json.dumps(_minimal_payload()))

        unsloth_cli = UnslothContributor().get_cli()
        runner = CliRunner()
        result = runner.invoke(
            unsloth_cli,
            [
                "submit",
                str(path),
                "--workspace",
                "acme-corp",
                "--base-url",
                "https://nhx.test",
            ],
        )

        assert result.exit_code == 0, result.stdout + result.stderr
        assert submitted.workspace == "acme-corp"
        assert submitted.base_url == "https://nhx.test"
        # Raw input shape — to_spec runs inside the (mocked-out) submit_remote.
        spec = submitted.spec
        assert spec is not None
        captured_input = UnslothJobInput.model_validate(spec)
        assert captured_input.model.name == "unsloth/Qwen2.5-0.5B-Instruct"


class TestExplain:
    def test_explain_exposes_input_and_output_schemas(self) -> None:
        unsloth_cli = UnslothContributor().get_cli()
        runner = CliRunner()
        result = runner.invoke(unsloth_cli, ["explain"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert "input_spec_schema" in payload
        assert "spec_schema" in payload
        assert "/unsloth/jobs" in payload["endpoint"]


class TestJobsSubmitWire:
    def test_submit_remote_posts_to_unsloth_collection(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Job collection is /apis/customization/v2/workspaces/<ws>/unsloth/jobs."""
        capture: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            capture["method"] = request.method
            capture["url"] = str(request.url)
            capture["body"] = json.loads(request.content)
            return httpx.Response(200, json={"id": "job-1", "status": "queued"})

        monkeypatch.setattr(
            "nemo_helix_plugin.discovery.discover_jobs",
            lambda: {"customization.unsloth.jobs": UnslothJob},
        )

        path = tmp_path / "job.json"
        path.write_text(json.dumps(_minimal_payload()))

        scheduler = NemoJobScheduler()
        scheduler.submit_remote(
            UnslothJob,
            json.loads(load_job_json(path)),
            base_url="https://nhx.test",
            workspace="ws-a",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        assert capture["method"] == "POST"
        assert capture["url"] == "https://nhx.test/apis/customization/v2/workspaces/ws-a/unsloth/jobs"
        assert capture["body"]["spec"]["model"]["name"] == "unsloth/Qwen2.5-0.5B-Instruct"
