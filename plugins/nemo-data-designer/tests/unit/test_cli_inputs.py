# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import typer
from data_designer.config.utils.constants import DEFAULT_NUM_RECORDS
from nemo_data_designer_plugin.cli import inputs
from nemo_data_designer_plugin.jobs.create import CreateJob
from nemo_helix_plugin.cli import NemoCLI
from nemo_helix_plugin.commands import add_job_commands
from nemo_helix_plugin.job import NemoJob
from typer.testing import CliRunner

_CONFIG_MODULE = """
import data_designer.config as dd


def load_config_builder():
    builder = dd.DataDesignerConfigBuilder()
    builder.add_column(dd.SamplerColumnConfig(name="id", sampler_type="uuid", params={}))
    return builder
"""


class _CreateOverrideCLI(NemoCLI):
    name: ClassVar[str] = "data-designer"

    def get_cli(self) -> typer.Typer:
        return typer.Typer()

    def update_job_cli(self, job_cls: type[NemoJob], group: typer.Typer) -> None:
        if job_cls is CreateJob:
            inputs.apply_create_cli_overrides(group)


def test_create_cli_override_delegates_to_generated_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    typer_ctx_sentinel = SimpleNamespace()
    group = typer.Typer()

    def original(
        typer_ctx: object,
        *,
        spec: str,
        spec_file: Path | None,
        options: list[str],
        options_file: Path | None,
        profile: str | None,
        workspace: str,
        config: object | None,
        config_file: object | None,
    ) -> None:
        captured.update(
            {
                "typer_ctx": typer_ctx,
                "spec": spec,
                "spec_file": spec_file,
                "options": options,
                "options_file": options_file,
                "profile": profile,
                "workspace": workspace,
                "config": config,
                "config_file": config_file,
            }
        )

    group.callback(invoke_without_command=True)(original)

    @contextlib.contextmanager
    def fake_spec_from_builder(config_source: str, num_records: int) -> Iterator[str]:
        captured["config_source"] = config_source
        captured["num_records"] = num_records
        yield '{"spec": true}'

    monkeypatch.setattr(inputs, "_spec_from_builder", fake_spec_from_builder)

    inputs.apply_create_cli_overrides(group)
    callback = group.registered_callback
    assert callback is not None
    assert callback.callback is not None

    callback.callback(
        typer_ctx_sentinel,
        "config.yaml",
        num_records=7,
        workspace="team-a",
        profile="training",
        options=["backend.key=value"],
        options_file=Path("options.yaml"),
    )

    assert captured == {
        "typer_ctx": typer_ctx_sentinel,
        "config_source": "config.yaml",
        "num_records": 7,
        "spec": '{"spec": true}',
        "spec_file": None,
        "options": ["backend.key=value"],
        "options_file": Path("options.yaml"),
        "profile": "training",
        "workspace": "team-a",
        "config": None,
        "config_file": None,
    }


def test_create_cli_override_accepts_config_before_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    group = typer.Typer(no_args_is_help=False)

    def original(
        typer_ctx: typer.Context,
        *,
        spec: str,
        spec_file: Path | None,
        options: list[str],
        options_file: Path | None,
        profile: str | None,
        workspace: str,
        config: object | None,
        config_file: object | None,
    ) -> None:
        captured.update(
            {
                "spec": spec,
                "spec_file": spec_file,
                "options": options,
                "options_file": options_file,
                "profile": profile,
                "workspace": workspace,
                "config": config,
                "config_file": config_file,
                "invoked_subcommand": typer_ctx.invoked_subcommand,
            }
        )

    group.callback(invoke_without_command=True)(original)

    @group.command("explain")
    def explain(
        profile: str | None = typer.Option(None, "--profile"),
    ) -> None:
        captured.update({"explain_profile": profile})

    @contextlib.contextmanager
    def fake_spec_from_builder(config_source: str, num_records: int) -> Iterator[str]:
        captured["config_source"] = config_source
        captured["num_records"] = num_records
        yield '{"spec": true}'

    monkeypatch.setattr(inputs, "_spec_from_builder", fake_spec_from_builder)

    inputs.apply_create_cli_overrides(group)
    app = typer.Typer()
    app.add_typer(group, name="create")

    result = CliRunner().invoke(
        app,
        [
            "create",
            "config.yaml",
            "--num-records",
            "7",
            "--workspace",
            "team-a",
            "--profile",
            "training",
            "-o",
            "backend.key=value",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "config_source": "config.yaml",
        "num_records": 7,
        "spec": '{"spec": true}',
        "spec_file": None,
        "options": ["backend.key=value"],
        "options_file": None,
        "profile": "training",
        "workspace": "team-a",
        "config": None,
        "config_file": None,
        "invoked_subcommand": None,
    }


def test_create_cli_override_preserves_explain_callback() -> None:
    captured: dict[str, object] = {}
    group = typer.Typer(no_args_is_help=False)

    def original(typer_ctx: typer.Context, **kwargs: object) -> None:
        del typer_ctx, kwargs
        captured["created"] = True

    group.callback(invoke_without_command=True)(original)

    @group.command("explain")
    def explain(
        profile: str | None = typer.Option(None, "--profile"),
    ) -> None:
        captured.update({"profile": profile})

    inputs.apply_create_cli_overrides(group)
    app = typer.Typer()
    app.add_typer(group, name="create")

    result = CliRunner().invoke(app, ["create", "explain", "--profile", "research"])

    assert result.exit_code == 0, result.output
    assert captured == {"profile": "research"}


@pytest.mark.parametrize(
    ("num_records_args", "expected_num_records"),
    [([], DEFAULT_NUM_RECORDS), (["--num-records", "5"], 5)],
)
def test_create_submits_config_source_with_or_without_num_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    num_records_args: list[str],
    expected_num_records: int,
) -> None:
    submitted: list[dict] = []

    def capture_submit(_self: object, _job_cls: type[NemoJob], spec: dict, **_kwargs: object) -> dict:
        submitted.append(spec)
        return {"id": "job-123"}

    monkeypatch.setattr("nemo_helix_plugin.scheduler.NemoJobScheduler.submit_remote", capture_submit)
    config_path = tmp_path / "config.py"
    config_path.write_text(_CONFIG_MODULE)
    app = typer.Typer()

    @app.callback()
    def _noop() -> None:
        pass

    add_job_commands(app, {"data-designer.create": CreateJob}, cli=_CreateOverrideCLI())

    result = CliRunner().invoke(app, ["create", str(config_path), "--workspace", "default", *num_records_args])

    assert result.exit_code == 0, result.output
    assert "Usage:" not in result.output
    assert len(submitted) == 1
    assert submitted[0]["num_records"] == expected_num_records
