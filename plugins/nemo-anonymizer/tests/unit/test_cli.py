# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import httpx
import yaml
from nemo_anonymizer_plugin import cli as cli_module
from nemo_anonymizer_plugin.cli import AnonymizerCLI
from nemo_anonymizer_plugin.functions.preview import PreviewFunction
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.commands import add_function_commands, add_job_commands
from nemo_helix_plugin.job import NemoJob
from typer.testing import CliRunner


class _CliState(SimpleNamespace):
    """Stand-in CLI state: the platform comes from ``nemo --base-url`` / the active context."""

    def __init__(self, base_url: str) -> None:
        self._client = NemoClient(
            base_url=base_url,
            http_client=httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, request=req))),
        )

    def get_base_url(self, default: str | None = None) -> str:
        return self._client.base_url

    def get_client(self, timeout: float | httpx.Timeout | None = None) -> NemoClient:
        return self._client

    def typed_client(self, client_cls: type[NemoClient], timeout: float | httpx.Timeout | None = None) -> NemoClient:
        return client_cls.from_client(self._client)

    def get_workspace(self) -> str | None:
        return None

    def get_output_format(self, override: str | None = None, *, apply_non_tty_default: bool = True) -> str:
        return override or "json"


def _cli_state(base_url: str) -> _CliState:
    return _CliState(base_url)


class _RunJob(NemoJob):
    name: ClassVar[str] = "run"
    description: ClassVar[str] = "Run test job."

    def run(self, config: dict) -> dict:
        return {"config": config}


def _write_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "replace": {
                    "kind": "redact",
                    "format_template": "[REDACTED_{label}]",
                }
            }
        )
    )


def test_cli_only_registers_manual_validate_command() -> None:
    result = CliRunner().invoke(AnonymizerCLI().get_cli(), ["--help"])

    assert result.exit_code == 0, result.output
    assert "validate" in result.output
    assert "preview-local" not in result.output
    assert "run-local" not in result.output


def test_preview_function_uses_flat_remote_submit(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post_function_submit(url, body, *, headers, http_client, renderer_cls=None, cli_kwargs=None):
        captured["url"] = url
        captured["body"] = body
        captured["headers"] = headers
        captured["http_client"] = http_client
        captured["renderer_cls"] = renderer_cls
        captured["cli_kwargs"] = cli_kwargs

    monkeypatch.setattr(
        "nemo_helix_plugin.discovery.discover_functions",
        lambda: {"anonymizer.preview": PreviewFunction},
    )
    monkeypatch.setattr("nemo_helix_plugin.commands._post_function_submit", fake_post_function_submit)
    cli = AnonymizerCLI()
    app = cli.get_cli()
    runner = CliRunner()
    add_function_commands(app, {"anonymizer.preview": PreviewFunction}, cli=cli)

    help_result = runner.invoke(app, ["--help"])
    preview_result = runner.invoke(
        app,
        [
            "preview",
            "--spec",
            json.dumps(
                {
                    "config": {"replace": {"kind": "redact"}},
                    "data": {"source": "https://example.com/input.csv", "text_column": "text"},
                    "model_configs": [{"alias": "detector", "provider": "provider", "model": "test/model"}],
                    "num_records": 2,
                }
            ),
            "--workspace",
            "team-a",
        ],
        obj=_cli_state("http://platform.example"),
    )
    nested_result = runner.invoke(app, ["preview", "submit", "--spec", "{}"])

    assert help_result.exit_code == 0, help_result.output
    assert "preview" in help_result.output
    assert preview_result.exit_code == 0, preview_result.output
    assert captured["url"] == "http://platform.example/apis/anonymizer/v2/workspaces/team-a/preview"
    assert captured["body"] == {
        "config": {"replace": {"kind": "redact"}},
        "data": {"source": "https://example.com/input.csv", "text_column": "text"},
        "model_configs": [{"alias": "detector", "provider": "provider", "model": "test/model"}],
        "num_records": 2,
    }
    assert captured["headers"] == {}
    assert captured["http_client"] is not None
    assert captured["renderer_cls"] is None
    assert nested_result.exit_code == 2
    assert "unexpected extra argument" in nested_result.output


def test_run_job_uses_flat_remote_submit(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_submit_remote(self, job_cls, spec, **kwargs):
        captured["job_cls"] = job_cls
        captured["spec"] = spec
        captured["kwargs"] = kwargs
        return {"name": "anon-job-1", "workspace": "team-a"}

    monkeypatch.setattr("nemo_helix_plugin.scheduler.NemoJobScheduler.submit_remote", fake_submit_remote)

    cli = AnonymizerCLI()
    app = cli.get_cli()
    add_job_commands(app, {"anonymizer.run": _RunJob}, cli=cli)
    runner = CliRunner()

    run_result = runner.invoke(
        app,
        [
            "run",
            "--spec",
            '{"name": "Remote"}',
            "--workspace",
            "team-a",
        ],
        obj=_cli_state("http://platform.example"),
    )
    nested_result = runner.invoke(app, ["run", "run", "--spec", '{"name": "Nested"}'])
    help_result = runner.invoke(app, ["run", "--help"])

    assert run_result.exit_code == 0, run_result.output
    assert json.loads(run_result.output) == {"name": "anon-job-1", "workspace": "team-a"}
    assert captured["job_cls"] is _RunJob
    assert captured["spec"] == {"name": "Remote"}
    assert captured["kwargs"] == {
        "base_url": "http://platform.example",
        "workspace": "team-a",
        "profile": None,
        "options": None,
        "headers": None,
    }
    assert nested_result.exit_code == 2
    assert "No such command 'run'" in nested_result.output
    assert help_result.exit_code == 0, help_result.output
    assert "Run test job." in help_result.output
    assert "Pass the spec via --spec" in help_result.output
    assert "Run locally, in-process." not in help_result.output


def test_validate_command_runs_library_validation(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeAnonymizer:
        def validate_config(self, config: object) -> None:
            captured["config"] = config

    def fake_make_local_anonymizer(*, model_configs: str | Path | None, artifact_path: Path | None = None):
        captured["model_configs"] = model_configs
        captured["artifact_path"] = artifact_path
        return FakeAnonymizer()

    monkeypatch.setattr(cli_module, "_make_local_anonymizer", fake_make_local_anonymizer)

    config = tmp_path / "config.yaml"
    model_configs = tmp_path / "models.yaml"
    _write_config(config)
    model_configs.write_text("model_configs: []\n")

    result = CliRunner().invoke(
        AnonymizerCLI().get_cli(),
        [
            "validate",
            "--config",
            str(config),
            "--model-configs",
            str(model_configs),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["model_configs"] == str(model_configs)
    assert captured["artifact_path"] is None
    assert "Config is valid." in result.output
