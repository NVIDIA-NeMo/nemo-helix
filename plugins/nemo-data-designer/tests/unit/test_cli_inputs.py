# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

import pytest
import typer
from nemo_data_designer_plugin.cli import inputs
from typer.testing import CliRunner


def test_create_cli_override_supports_root_generated_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    typer_ctx_sentinel = object()
    group = typer.Typer()

    def original(
        typer_ctx: object,
        *,
        spec: str,
        spec_file: Path | None,
        options: list[str],
        options_file: Path | None,
        profile: str | None,
        cluster: str | None,
        base_url: str | None,
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
                "cluster": cluster,
                "base_url": base_url,
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
        cluster="local",
        base_url="http://nemo.test",
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
        "cluster": "local",
        "base_url": "http://nemo.test",
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
        cluster: str | None,
        base_url: str | None,
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
                "cluster": cluster,
                "base_url": base_url,
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
        cluster: str | None = typer.Option(None, "--cluster"),
    ) -> None:
        captured.update({"explain_profile": profile, "explain_cluster": cluster})

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
            "--cluster",
            "local",
            "--base-url",
            "http://nemo.test",
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
        "cluster": "local",
        "base_url": "http://nemo.test",
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
        cluster: str | None = typer.Option(None, "--cluster"),
    ) -> None:
        captured.update({"profile": profile, "cluster": cluster})

    inputs.apply_create_cli_overrides(group)
    app = typer.Typer()
    app.add_typer(group, name="create")

    result = CliRunner().invoke(app, ["create", "explain", "--profile", "research", "--cluster", "ignored"])

    assert result.exit_code == 0, result.output
    assert captured == {"profile": "research", "cluster": "ignored"}
