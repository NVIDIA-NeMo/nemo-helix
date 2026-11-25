# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compass Fabric adapter descriptor and the config ``compass.analyst`` derives.

These exercise the real Fabric planner rather than a hand-rolled schema copy,
so an invalid ``config.accepts`` key or an untranslatable harness kind fails
here instead of inside a running job.
"""

from __future__ import annotations

import json
from pathlib import Path

import nemo_fabric as fabric
from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.fabric.translator import translate_agent_config
from nemo_agents_plugin.plugin_managed_agents import load_plugin_managed_agent_type
from nemo_compass_plugin.analyst import COMPASS_ADAPTER_ID, CompassAnalyst

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR = _PLUGIN_ROOT / "analyst.fabric-adapter.json"


_REINSTALL_HINT = "Refresh it with: uv sync --reinstall-package nemo-compass-plugin"


def _installed_descriptor() -> Path:
    """Where Fabric's default InstalledPackage discovery looks for the descriptor.

    The wheel ships it as shared-data (see ``pyproject.toml``), so this is the
    path Fabric names in its own ``FabricConfigError`` when a setting is missing
    from the schema.
    """
    import sys

    return Path(sys.prefix) / "share" / "nemo-fabric" / "adapters" / "helix" / "compass" / DESCRIPTOR.name


def _require_installed_descriptor() -> Path:
    """Fail rather than skip: a skipped plan check is a green CI that proved nothing.

    A fresh venv always has the descriptor, so absence means a stale local
    install, not an unsupported environment.
    """
    installed = _installed_descriptor()
    assert installed.exists(), f"Adapter descriptor is not installed at {installed}. {_REINSTALL_HINT}"
    return installed


def _compass_config() -> AgentConfig:
    return AgentConfig.model_validate(CompassAnalyst.config(name="compass-analyst"))


def test_descriptor_declares_the_expected_adapter_id() -> None:
    descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))

    assert descriptor["adapter_id"] == COMPASS_ADAPTER_ID


def test_installed_descriptor_matches_the_source() -> None:
    """Shared-data is copied at install time, not symlinked into the venv.

    ``uv sync`` alone will not refresh it for an unchanged editable package;
    the descriptor only reaches Fabric after a reinstall.
    """
    installed = _require_installed_descriptor()

    assert json.loads(installed.read_text(encoding="utf-8")) == json.loads(DESCRIPTOR.read_text(encoding="utf-8")), (
        f"Stale installed descriptor. {_REINSTALL_HINT}"
    )


def test_the_type_passes_the_platforms_load_checks() -> None:
    """The check the Agents service runs on every load: entry point, context models, config against the adapter."""
    assert load_plugin_managed_agent_type("compass.analyst") is CompassAnalyst


def test_the_derived_config_is_a_valid_agent_config() -> None:
    config = _compass_config()

    assert config.name == "compass-analyst"
    assert config.environment.provider == "local"


def test_the_derived_config_translates_to_the_compass_adapter_with_no_settings() -> None:
    fabric_config = translate_agent_config(_compass_config())

    assert fabric_config.harness is not None
    assert fabric_config.harness.adapter_id == COMPASS_ADAPTER_ID
    assert not fabric_config.harness.settings
    assert not fabric_config.models


def test_descriptor_takes_no_settings_and_no_models() -> None:
    """Everything a run varies, models included, arrives in its request context.

    A stray setting is rejected, not ignored; and because the adapter accepts no
    ``models``, the derived config may name none.
    """
    descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))

    assert descriptor["settings_schema"]["properties"] == {}
    assert descriptor["settings_schema"]["additionalProperties"] is False
    assert "models" not in descriptor["config"]["accepts"]


def test_descriptor_declares_relay_telemetry_support() -> None:
    """Without this, Fabric rejects the relay config agents auto-wires for every job.

    Compass then cannot run at all, rather than merely running untraced --
    the export is configured by the agents layer, which asks the descriptor
    first.
    """
    telemetry = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))["telemetry"]

    assert "relay" in telemetry["providers"]
    assert "atif" in telemetry["providers"]["relay"]["outputs"]


def test_fabric_plans_the_derived_config(tmp_path: Path) -> None:
    """Fabric validates the adapter descriptor while planning; this is the real check."""
    _require_installed_descriptor()

    plan = fabric.Fabric().plan(translate_agent_config(_compass_config()), base_dir=tmp_path)

    assert plan is not None
