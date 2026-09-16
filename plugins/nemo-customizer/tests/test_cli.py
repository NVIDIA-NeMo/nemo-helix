# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import ClassVar

import pytest
import typer
from nemo_customizer.cli import CustomizationCLI, CustomizationCLIError
from nemo_helix_plugin.service import RouterSpec


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
