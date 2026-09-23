# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ngcsdk==4.20.1",
#   "pytest>=9.0.3,<10",
#   "pyyaml>=6.0.2",
#   "typer>=0.24.1,<0.25",
# ]
# ///

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import tomllib
from importlib.metadata import version
from inspect import signature
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ngcbase.errors import ResourceNotFoundException
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parents[1]))

from ngc_metadata import (  # noqa: E402
    DEFAULT_LABELS,
    DEFAULT_LOGO,
    PUBLISHER,
    Client,
    app,
    default_display_name,
    discover_assets,
    load_asset,
    sync_chart,
    sync_container,
)


def test_ngc_sdk_pin_matches_production_script() -> None:
    script_path = Path(__file__).parents[1] / "ngc_metadata.py"
    metadata_block = script_path.read_text(encoding="utf-8").split("# ///", maxsplit=2)[1]
    metadata = tomllib.loads("\n".join(line.removeprefix("# ") for line in metadata_block.splitlines()[1:]))
    ngc_sdk_pin = next(
        dependency.removeprefix("ngcsdk==")
        for dependency in metadata["dependencies"]
        if dependency.startswith("ngcsdk==")
    )

    assert version("ngcsdk") == ngc_sdk_pin


def test_ngc_sdk_supports_required_metadata_parameters() -> None:
    client = Client()

    container_parameters = {
        "image",
        "desc",
        "overview",
        "logo",
        "publisher",
        "display_name",
    }
    assert container_parameters | {"label"} <= signature(client.registry.image.create).parameters.keys()
    assert container_parameters | {"labels"} <= signature(client.registry.image.update).parameters.keys()

    chart_parameters = {
        "target",
        "short_description",
        "overview_filepath",
        "display_name",
        "labels",
        "logo",
        "publisher",
    }
    assert chart_parameters <= signature(client.registry.chart.create).parameters.keys()
    assert chart_parameters <= signature(client.registry.chart.update).parameters.keys()


def test_default_display_name_preserves_known_names() -> None:
    assert default_display_name("nhx-tasks") == "NeMo Helix Tasks"
    assert default_display_name("nhx-safe-synthesizer-tasks") == "Safe Synthesizer Tasks"


def test_load_asset_uses_defaults(tmp_path: Path) -> None:
    path = tmp_path / "nhx-auditor-tasks.md"
    path.write_text("# Overview\n", encoding="utf-8")

    asset = load_asset(path, "container")

    assert asset.name == "nhx-auditor-tasks"
    assert asset.display_name == "Auditor Tasks"
    assert asset.description == "Auditor Tasks is part of the NeMo Helix"
    assert asset.labels == DEFAULT_LABELS
    assert asset.logo == DEFAULT_LOGO
    assert asset.overview == "# Overview\n"


def test_load_chart_uses_deployment_description(tmp_path: Path) -> None:
    path = tmp_path / "nemo-helix.md"
    path.write_text("# Overview\n", encoding="utf-8")

    asset = load_asset(path, "chart")

    assert asset.description == "Deploy NeMo Helix to Kubernetes"


def test_load_asset_applies_front_matter_overrides(tmp_path: Path) -> None:
    path = tmp_path / "nhx-auditor-tasks.md"
    path.write_text(
        "---\n"
        "display_name: NeMo Auditor\n"
        "description: Runs auditor jobs\n"
        "labels: [NeMo, Security]\n"
        "logo: https://example.com/logo.png\n"
        "---\n"
        "# Overview\n",
        encoding="utf-8",
    )

    asset = load_asset(path, "container")

    assert asset.display_name == "NeMo Auditor"
    assert asset.description == "Runs auditor jobs"
    assert asset.labels == ["NeMo", "Security"]
    assert asset.logo == "https://example.com/logo.png"
    assert asset.overview == "# Overview\n"


def test_discover_assets_infers_type_and_name(tmp_path: Path) -> None:
    (tmp_path / "charts").mkdir()
    (tmp_path / "containers").mkdir()
    (tmp_path / "charts" / "nemo-helix.md").write_text("chart", encoding="utf-8")
    (tmp_path / "containers" / "nhx-api.md").write_text("container", encoding="utf-8")

    assets = discover_assets(tmp_path)

    assert [(asset.asset_type, asset.name) for asset in assets] == [
        ("container", "nhx-api"),
        ("chart", "nemo-helix"),
    ]


def test_repository_assets_are_valid() -> None:
    assets_dir = Path(__file__).parents[2] / "assets" / "ngc"

    assert discover_assets(assets_dir)


def test_sync_container_updates_existing_asset(tmp_path: Path) -> None:
    asset = load_asset(_write_overview(tmp_path / "nhx-api.md"), "container")
    client = MagicMock()

    action = sync_container(client, asset, "org/team/nhx-api")

    assert action == "updated"
    client.registry.image.update.assert_called_once_with(
        image="org/team/nhx-api",
        desc=asset.description,
        overview=asset.overview,
        labels=DEFAULT_LABELS,
        logo=DEFAULT_LOGO,
        publisher=PUBLISHER,
        display_name=asset.display_name,
    )
    client.registry.image.create.assert_not_called()


def test_sync_container_creates_missing_asset(tmp_path: Path) -> None:
    asset = load_asset(_write_overview(tmp_path / "nhx-api.md"), "container")
    client = MagicMock()
    client.registry.image.info.side_effect = ResourceNotFoundException("missing")

    action = sync_container(client, asset, "org/team/nhx-api")

    assert action == "created"
    client.registry.image.create.assert_called_once_with(
        image="org/team/nhx-api",
        desc=asset.description,
        overview=asset.overview,
        label=DEFAULT_LABELS,
        logo=DEFAULT_LOGO,
        publisher=PUBLISHER,
        display_name=asset.display_name,
    )
    client.registry.image.update.assert_not_called()


def test_sync_chart_creates_missing_asset(tmp_path: Path) -> None:
    asset = load_asset(_write_overview(tmp_path / "nemo-helix.md"), "chart")
    client = MagicMock()
    client.registry.chart.info.side_effect = ResourceNotFoundException("missing")
    uploaded_overview = ""

    def capture_overview(**kwargs: object) -> None:
        nonlocal uploaded_overview
        uploaded_overview = Path(str(kwargs["overview_filepath"])).read_text(encoding="utf-8")

    client.registry.chart.create.side_effect = capture_overview

    action = sync_chart(client, asset, "org/team/nemo-helix")

    assert action == "created"
    kwargs = client.registry.chart.create.call_args.kwargs
    assert kwargs | {"overview_filepath": None} == {
        "target": "org/team/nemo-helix",
        "overview_filepath": None,
        "display_name": asset.display_name,
        "labels": DEFAULT_LABELS,
        "logo": DEFAULT_LOGO,
        "publisher": PUBLISHER,
        "short_description": asset.description,
    }
    assert uploaded_overview == asset.overview
    client.registry.chart.update.assert_not_called()


def test_cli_dry_run_lists_assets(tmp_path: Path) -> None:
    (tmp_path / "containers").mkdir()
    _write_overview(tmp_path / "containers" / "nhx-api.md")

    result = CliRunner().invoke(
        app,
        ["--org", "org", "--team", "team", "--assets-dir", str(tmp_path), "--dry-run"],
    )

    assert result.exit_code == 0
    assert result.stdout == "Would sync container org/team/nhx-api\n"


def test_cli_configures_org_auth_for_team_target(tmp_path: Path) -> None:
    (tmp_path / "containers").mkdir()
    _write_overview(tmp_path / "containers" / "nhx-api.md")
    client = MagicMock()

    with patch("ngc_metadata.Client", return_value=client):
        result = CliRunner().invoke(
            app,
            [
                "--org",
                "org",
                "--team",
                "team",
                "--assets-dir",
                str(tmp_path),
                "--api-key",
                "service-key",
            ],
        )

    assert result.exit_code == 0
    client.configure.assert_called_once_with(api_key="service-key", org_name="org", team_name="no-team")
    client.registry.image.info.assert_called_once_with("org/team/nhx-api")


def test_cli_can_match_authentication_team(tmp_path: Path) -> None:
    (tmp_path / "containers").mkdir()
    _write_overview(tmp_path / "containers" / "nhx-api.md")
    client = MagicMock()

    with patch("ngc_metadata.Client", return_value=client):
        result = CliRunner().invoke(
            app,
            [
                "--org",
                "org",
                "--team",
                "team",
                "--assets-dir",
                str(tmp_path),
                "--api-key",
                "personal-key",
                "--auth-match-team",
            ],
        )

    assert result.exit_code == 0
    client.configure.assert_called_once_with(api_key="personal-key", org_name="org", team_name="team")


def _write_overview(path: Path) -> Path:
    path.write_text("# Overview\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                __file__,
                "-q",
                "-c",
                os.devnull,
                "-p",
                "no:cacheprovider",
                "-W",
                "ignore::SyntaxWarning",
                "--confcutdir",
                str(Path(__file__).parent),
            ]
        )
    )
