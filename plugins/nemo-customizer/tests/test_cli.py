# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import ClassVar

import pytest
import typer
from nemo_customizer.cli import CustomizationCLI, CustomizationCLIError
from nemo_platform_plugin.customization_contributor import CustomizationCLISummary
from nemo_platform_plugin.service import RouterSpec


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
    """Stub that also contributes a top-level blurb."""

    name: ClassVar[str] = "stub"

    def get_cli_summary(self) -> CustomizationCLISummary:
        return CustomizationCLISummary(
            trains="Widgets.",
            runs_on="a widget press.",
            job_json="widget, press.",
            pick_when="you need a widget.",
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
    """Group help is printed through an 80-column Rich console; longer lines re-wrap."""
    help_text = _root_help({"stub": _SummaryContributor()}, monkeypatch)
    assert [line for line in help_text.splitlines() if len(line) > 80] == []
