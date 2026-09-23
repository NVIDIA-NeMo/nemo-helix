# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
import typer
from nemo_customizer.cli import CustomizationCLI, CustomizationCLIError
from nemo_helix_plugin.customization_contributor import CustomizationCLISummary
from nemo_helix_plugin.service import RouterSpec
from nhx.customization_common.cli.uploads import UploadReport
from typer.testing import CliRunner


class _FakeContributor:
    name: ClassVar[str] = "fake"

    def get_routers(self) -> list[RouterSpec]:
        return []

    def get_cli(self) -> typer.Typer:
        app = typer.Typer()

        @app.command("info")
        def info() -> None:
            typer.echo("fake")

        return app


def test_cli_raises_without_contributors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nemo_customizer.cli.discover_customization_contributors",
        lambda: {},
    )
    with pytest.raises(CustomizationCLIError, match="no contributors"):
        CustomizationCLI()


def test_cli_mounts_contributor_subgroups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nemo_customizer.cli.discover_customization_contributors",
        lambda: {"fake": _FakeContributor()},
    )
    cli = CustomizationCLI()
    app = cli.get_cli()
    assert "fake" in {group.name for group in app.registered_groups}


class _SummaryContributor(_FakeContributor):
    """Stub that also contributes a top-level summary."""

    name: ClassVar[str] = "stub"

    def get_cli_summary(self) -> CustomizationCLISummary:
        return CustomizationCLISummary(
            trains="Widgets.",
            runs_on="a widget press.",
            job_json="widget, press.",
            use_when="you need a widget.",
            command="nemo customization stub submit JOB.json",
        )


def _root_help(contributors: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(
        "nemo_customizer.cli.discover_customization_contributors",
        lambda: contributors,
    )
    return CustomizationCLI().get_cli().info.help or ""


def test_root_help_aggregates_contributor_summaries(monkeypatch: pytest.MonkeyPatch) -> None:
    help_text = _root_help({"stub": _SummaryContributor()}, monkeypatch)
    assert "stub" in help_text
    assert "Trains: Widgets." in help_text
    assert "Submit: nemo customization stub submit JOB.json" in help_text


def test_root_help_lists_contributors_in_name_order(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Zebra(_SummaryContributor):
        name: ClassVar[str] = "zebra"

    help_text = _root_help({"zebra": _Zebra(), "stub": _SummaryContributor()}, monkeypatch)
    assert help_text.index("\nstub\n") < help_text.index("\nzebra\n")


def test_root_help_drops_uninstalled_contributors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Help follows discovery: the router never names a backend of its own."""
    help_text = _root_help({"stub": _SummaryContributor()}, monkeypatch)
    for backend in ("automodel", "unsloth", "rl"):
        assert backend not in help_text


def test_root_help_lists_contributor_without_a_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    help_text = _root_help({"fake": _FakeContributor()}, monkeypatch)
    assert "\nfake\n" in help_text
    assert "Trains:" not in help_text


def test_root_help_stays_within_the_rendered_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """Group help is printed through an 80-column Rich console, and longer lines re-wrap."""
    help_text = _root_help({"stub": _SummaryContributor()}, monkeypatch)
    assert [line for line in help_text.splitlines() if len(line) > 80] == []


class _HeadlessContributor:
    """Discovered, but contributes no CLI."""

    name: ClassVar[str] = "headless"

    def get_routers(self) -> list[RouterSpec]:
        return []

    def get_cli(self) -> typer.Typer | None:
        return None


def test_root_help_omits_contributors_without_a_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Help must not point at 'nemo customization headless', which would not exist."""
    contributors = {"fake": _FakeContributor(), "headless": _HeadlessContributor()}
    monkeypatch.setattr("nemo_customizer.cli.discover_customization_contributors", lambda: contributors)

    app = CustomizationCLI().get_cli()

    assert {group.name for group in app.registered_groups} == {"fake"}
    help_text = app.info.help or ""
    assert "\nfake\n" in help_text
    assert "headless" not in help_text


class _ContextState:
    """Minimal stand-in for ``CLIContext`` with a non-'default' workspace."""

    def get_workspace(self) -> str | None:
        return "my-team-ws"


def _upload_workspace(args: list[str], state: object | None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> object:
    """Run ``nemo customization --upload-dataset`` and return the workspace it uploaded to."""
    monkeypatch.setattr("nemo_customizer.cli.discover_customization_contributors", lambda: {"fake": _FakeContributor()})
    captured: dict[str, object] = {}

    def fake_create(_typer_ctx: typer.Context, **kwargs: object) -> UploadReport:
        captured.update(kwargs)
        return UploadReport()

    monkeypatch.setattr("nemo_customizer.cli._create_customization_resources", fake_create)
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("{}\n")

    parent = typer.Typer()

    @parent.callback()
    def _root(ctx: typer.Context) -> None:
        ctx.obj = state

    parent.add_typer(CustomizationCLI().get_cli(), name="customization")
    result = CliRunner().invoke(parent, ["customization", "--upload-dataset", str(dataset), *args])
    assert result.exit_code == 0, result.output
    return captured["workspace"]


@pytest.mark.parametrize(
    ("args", "state", "env", "expected"),
    [
        ([], _ContextState(), None, "my-team-ws"),
        (["--workspace", "acme-corp"], _ContextState(), None, "acme-corp"),
        (["-w", "default"], _ContextState(), None, "default"),
        ([], None, "env-ws", "env-ws"),
        ([], None, None, "default"),
    ],
    ids=["context", "explicit-flag", "explicit-default", "env-var", "fallback"],
)
def test_upload_resolves_workspace_like_submit(
    args: list[str],
    state: object | None,
    env: str | None,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Uploads land in the same workspace ``submit`` would use, not a literal 'default'."""
    if env is None:
        monkeypatch.delenv("NHX_WORKSPACE", raising=False)
    else:
        monkeypatch.setenv("NHX_WORKSPACE", env)
    assert _upload_workspace(args, state, monkeypatch, tmp_path) == expected


def _flat(output: str) -> str:
    """Undo the Rich error box, which wraps a usage error at 80 columns."""
    return " ".join(output.replace("│", " ").split())


def _invoke_root(args: list[str], monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr("nemo_customizer.cli.discover_customization_contributors", lambda: {"fake": _FakeContributor()})
    return CliRunner().invoke(CustomizationCLI().get_cli(), args)


@pytest.mark.parametrize(
    "flag_args",
    [
        ["--upload-model", "org/model"],
        ["--upload-dataset", "train.jsonl"],
        ["--upload-environment", "env/"],
        ["--exist-ok"],
        ["--workspace", "team-a"],
        ["-w", "team-a"],
        ["--base-url", "https://nhx.test"],
        ["--cluster", "prod"],
        ["--hf-token-secret", "hf-token"],
    ],
    ids=lambda args: args[0],
)
def test_group_flags_before_a_subcommand_are_rejected(flag_args: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """A flag the group would drop on the way to a subcommand is an error, not a silent no-op."""
    result = _invoke_root([*flag_args, "fake", "info"], monkeypatch)

    assert result.exit_code == 2
    assert "must come after the subcommand" in _flat(result.output)
    assert "nemo customization fake submit [OPTIONS] JOB_JSON" in _flat(result.output)
    assert "fake\n" not in result.output  # the subcommand never ran


def test_error_names_every_misplaced_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _invoke_root(["--upload-dataset", "train.jsonl", "-w", "team-a", "fake", "info"], monkeypatch)

    assert result.exit_code == 2
    assert "--upload-dataset, --workspace must come after the subcommand" in _flat(result.output)


def test_subcommand_without_group_flags_still_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """NHX_WORKSPACE is resolved by the subcommand; it is not a misplaced flag."""
    monkeypatch.setenv("NHX_WORKSPACE", "env-ws")

    result = _invoke_root(["fake", "info"], monkeypatch)

    assert result.exit_code == 0, result.output
    assert result.output == "fake\n"
