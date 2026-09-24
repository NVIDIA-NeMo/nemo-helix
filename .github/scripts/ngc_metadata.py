# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ngcsdk==4.20.1",
#   "pyyaml>=6.0.2",
#   "typer>=0.24.1,<0.25",
# ]
# ///

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create or update NGC metadata from Markdown files."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import typer
import yaml
from ngcbase.errors import ResourceNotFoundException
from ngcsdk import Client
from registry.errors import ChartNotFoundException

DEFAULT_ASSETS_DIR = Path(".github/assets/ngc")
DEFAULT_LABELS = ["NeMo"]
DEFAULT_LOGO = "https://images.ngc.nvidia.com/public-images/llm-social-data-flywheel-gtc25-1200x675.png"
PUBLISHER = "NVIDIA"
SUPPORTED_DIRECTORIES: dict[str, Literal["container", "chart"]] = {
    "containers": "container",
    "charts": "chart",
}
WORD_OVERRIDES = {
    "api": "API",
    "cpu": "CPU",
    "gpu": "GPU",
    "nemo": "NeMo",
    "nhx": "NeMo Helix",
    "sdk": "SDK",
}
COMPONENT_DISPLAY_NAMES = {
    "nhx-auditor-tasks": "Auditor Tasks",
    "nhx-safe-synthesizer-tasks": "Safe Synthesizer Tasks",
}

app = typer.Typer(add_completion=False, no_args_is_help=True)


@dataclass(frozen=True)
class Asset:
    asset_type: Literal["container", "chart"]
    name: str
    overview: str
    display_name: str
    description: str
    labels: list[str]
    logo: str


def default_display_name(name: str) -> str:
    """Convert an NGC asset name into a human-readable display name."""
    if name in COMPONENT_DISPLAY_NAMES:
        return COMPONENT_DISPLAY_NAMES[name]
    return " ".join(WORD_OVERRIDES.get(word.lower(), word.capitalize()) for word in name.split("-"))


def split_front_matter(content: str) -> tuple[dict[str, object], str]:
    """Split required YAML front matter from Markdown content."""
    lines = content.split("\n")
    if lines[0] != "---":
        raise ValueError("Markdown front matter must start with '---' on the first line")

    try:
        end = lines.index("---", 1)
    except ValueError as error:
        raise ValueError("Markdown front matter must end with '---'") from error

    front_matter = "\n".join(lines[1:end])
    overview = "\n".join(lines[end + 1 :])
    if all(not line.strip() or line.lstrip().startswith("#") for line in lines[1:end]):
        return {}, overview

    metadata = yaml.safe_load(front_matter)
    if not isinstance(metadata, dict):
        raise ValueError("Markdown front matter must be a mapping")
    return metadata, overview


def load_asset(path: Path, asset_type: Literal["container", "chart"]) -> Asset:
    """Load one asset and apply its metadata defaults."""
    try:
        metadata, overview = split_front_matter(path.read_text(encoding="utf-8"))
    except (ValueError, yaml.YAMLError) as error:
        raise ValueError(f"{path}: {error}") from error

    for field in ("display_name", "description", "logo"):
        if field in metadata:
            value = metadata[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{path}: {field} must be a nonblank string")

    name = path.stem
    display_name = str(metadata.get("display_name", default_display_name(name)))
    default_description = (
        f"Deploy {display_name} to Kubernetes" if asset_type == "chart" else f"{display_name} is part of the NeMo Helix"
    )
    description = str(metadata.get("description", default_description))
    labels = metadata.get("labels", DEFAULT_LABELS)
    if (
        not isinstance(labels, list)
        or not labels
        or not all(isinstance(label, str) and label.strip() for label in labels)
    ):
        raise ValueError(f"{path}: labels must be a nonempty list of nonblank strings")
    labels = [str(label) for label in labels]

    return Asset(
        asset_type=asset_type,
        name=name,
        overview=overview,
        display_name=display_name,
        description=description,
        labels=labels,
        logo=str(metadata.get("logo", DEFAULT_LOGO)),
    )


def discover_assets(assets_dir: Path) -> list[Asset]:
    """Discover assets from the directory name and Markdown filename."""
    assets = []
    for directory, asset_type in SUPPORTED_DIRECTORIES.items():
        assets.extend(load_asset(path, asset_type) for path in sorted((assets_dir / directory).glob("*.md")))
    return assets


def _target(org: str, team: str, name: str) -> str:
    return f"{org}/{team}/{name}"


def sync_container(client: Client, asset: Asset, target: str) -> Literal["created", "updated"]:
    """Create or update one container repository."""
    try:
        client.registry.image.info(target)
    except ResourceNotFoundException:
        client.registry.image.create(
            image=target,
            desc=asset.description,
            overview=asset.overview,
            label=asset.labels,
            logo=asset.logo,
            publisher=PUBLISHER,
            display_name=asset.display_name,
        )
        return "created"

    client.registry.image.update(
        image=target,
        desc=asset.description,
        overview=asset.overview,
        labels=asset.labels,
        logo=asset.logo,
        publisher=PUBLISHER,
        display_name=asset.display_name,
    )
    return "updated"


def sync_chart(client: Client, asset: Asset, target: str) -> Literal["created", "updated"]:
    """Create or update one Helm chart."""
    try:
        client.registry.chart.info(target)
        exists = True
    except (ChartNotFoundException, ResourceNotFoundException):
        exists = False

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", encoding="utf-8") as overview_file:
        overview_file.write(asset.overview)
        overview_file.flush()
        metadata = {
            "target": target,
            "overview_filepath": overview_file.name,
            "display_name": asset.display_name,
            "labels": asset.labels,
            "logo": asset.logo,
            "publisher": PUBLISHER,
            "short_description": asset.description,
        }
        if exists:
            client.registry.chart.update(**metadata)
            return "updated"

        client.registry.chart.create(**metadata)
        return "created"


def sync_asset(client: Client, asset: Asset, org: str, team: str) -> Literal["created", "updated"]:
    target = _target(org, team, asset.name)
    if asset.asset_type == "container":
        return sync_container(client, asset, target)
    return sync_chart(client, asset, target)


@app.command()
def main(
    org: Annotated[str, typer.Option(help="NGC organization.", envvar="NGC_ORG_NAME")],
    team: Annotated[str, typer.Option(help="NGC team.", envvar="NGC_TEAM_NAME")],
    assets_dir: Annotated[
        Path, typer.Option(help="Directory containing charts/ and containers/.")
    ] = DEFAULT_ASSETS_DIR,
    api_key: Annotated[str | None, typer.Option(help="NGC API key.", envvar="NGC_API_KEY")] = None,
    auth_match_team: Annotated[
        bool,
        typer.Option(help="Configure SDK authentication for the target team."),
    ] = False,
    dry_run: Annotated[bool, typer.Option(help="List assets without changing NGC.")] = False,
) -> None:
    """Synchronize Markdown metadata with NGC.

    The parent directory selects the asset type and the filename selects its
    NGC name. Every file must start with a YAML front matter block. Fields
    display_name, description, labels, and logo are optional overrides.
    """
    try:
        assets = discover_assets(assets_dir)
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    if not assets:
        raise typer.BadParameter(f"no Markdown assets found in {assets_dir}")

    if dry_run:
        for asset in assets:
            typer.echo(f"Would sync {asset.asset_type} {_target(org, team, asset.name)}")
        return

    if not api_key:
        raise typer.BadParameter("NGC API key is required")

    client = Client()
    client.configure(api_key=api_key, org_name=org, team_name=team if auth_match_team else "no-team")
    for asset in assets:
        action = sync_asset(client, asset, org, team)
        typer.echo(f"{action.capitalize()} {asset.asset_type} {_target(org, team, asset.name)}")


if __name__ == "__main__":
    app()
