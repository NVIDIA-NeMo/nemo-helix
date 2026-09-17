# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Insights analyst Fabric adapter descriptor and the config built for it.

These exercise the real Fabric planner rather than a hand-rolled schema copy,
so an invalid ``config.accepts`` key or an untranslatable harness kind fails
here instead of inside a running job.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nemo_fabric as fabric
from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.fabric.translator import translate_agent_config
from nemo_fabric_adapter_contract import models as contract
from nemo_insights_plugin import fabric_adapter
from nemo_insights_plugin.analyst.agent_config import ANALYST_ADAPTER_ID, build_analyst_agent_config

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR = _PLUGIN_ROOT / "insights-analyst.fabric-adapter.json"


_REINSTALL_HINT = "Refresh it with: uv sync --reinstall-package nemo-insights-plugin"


def _installed_descriptor() -> Path:
    """Where Fabric's default InstalledPackage discovery looks for the descriptor.

    The wheel ships it as shared-data (see ``pyproject.toml``), so this is the
    path Fabric names in its own ``FabricConfigError`` when a setting is missing
    from the schema.
    """
    import sys

    return Path(sys.prefix) / "share" / "nemo-fabric" / "adapters" / "insights-analyst" / DESCRIPTOR.name


def _require_installed_descriptor() -> Path:
    """Fail rather than skip: a skipped plan check is a green CI that proved nothing.

    A fresh venv always has the descriptor, so absence means a stale local
    install, not an unsupported environment.
    """
    installed = _installed_descriptor()
    assert installed.exists(), f"Adapter descriptor is not installed at {installed}. {_REINSTALL_HINT}"
    return installed


def _built_config(**overrides: Any) -> dict[str, Any]:
    return build_analyst_agent_config(
        agent="demo-agent",
        workspace="default",
        default_model="default/test-default",
        fast_model="default/test-fast",
        **overrides,
    )


def _analyst_config(**overrides: Any) -> AgentConfig:
    return AgentConfig.model_validate(_built_config(**overrides))


def test_descriptor_declares_the_expected_adapter_id() -> None:
    descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))

    assert descriptor["adapter_id"] == ANALYST_ADAPTER_ID


def test_installed_descriptor_matches_the_source() -> None:
    """Shared-data is copied at install time, not symlinked into the venv.

    ``uv sync`` alone will not refresh it for an unchanged editable package;
    the descriptor only reaches Fabric after a reinstall.
    """
    installed = _require_installed_descriptor()

    assert json.loads(installed.read_text(encoding="utf-8")) == json.loads(DESCRIPTOR.read_text(encoding="utf-8")), (
        f"Stale installed descriptor. {_REINSTALL_HINT}"
    )


def test_built_config_is_a_valid_agent_config() -> None:
    config = _analyst_config()

    assert config.name == "insights-analyst"
    assert config.default_harness == "insights"
    assert config.environment.provider == "local"


def test_built_config_translates_to_the_analyst_adapter() -> None:
    fabric_config = translate_agent_config(_analyst_config())

    assert fabric_config.harness is not None
    assert fabric_config.harness.adapter_id == ANALYST_ADAPTER_ID
    assert fabric_config.harness.settings["agent"] == "demo-agent"


def test_built_config_forwards_both_model_refs() -> None:
    """The analyst runs analysis on ``default`` and context summarization on ``fast``."""
    fabric_config = translate_agent_config(_analyst_config())

    assert set(fabric_config.models) == {"default", "fast"}
    assert fabric_config.models["default"].model == "default/test-default"
    assert fabric_config.models["fast"].model == "default/test-fast"


def test_analyst_adapter_resolves_the_forwarded_pair() -> None:
    """Guards the silent collapse: a missing ``fast`` falls back to ``default``."""
    fabric_config = translate_agent_config(_analyst_config())
    models = {
        key: contract.AgentModelConfig.from_mapping({"provider": "platform", "model": value.model})
        for key, value in fabric_config.models.items()
    }

    assert fabric_adapter._default_model_ref({}, models) == "default/test-default"
    assert fabric_adapter._fast_model_ref({}, models) == "default/test-fast"


def test_optional_read_settings_are_omitted_when_unset() -> None:
    settings = _built_config()["harnesses"]["insights"]["settings"]

    assert set(settings) == {"agent", "workspace"}


def test_optional_read_settings_are_carried_when_set() -> None:
    settings = _built_config(
        since=datetime(2026, 8, 1, tzinfo=timezone.utc),
        evaluation_id="eval-123",
        base_url="http://platform:8080",
        enable_observability=False,
    )["harnesses"]["insights"]["settings"]

    assert settings["since"] == "2026-08-01T00:00:00+00:00"
    assert settings["evaluation_id"] == "eval-123"
    assert settings["base_url"] == "http://platform:8080"
    assert settings["enable_observability"] is False


def test_descriptor_declares_the_ethos_setting() -> None:
    """The adapter reads ``ethos``; an undeclared key is rejected by the closed schema."""
    settings_schema = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))["settings_schema"]

    assert "ethos" in settings_schema["properties"]
    assert settings_schema["additionalProperties"] is False
    # The pre-rename name of the same field; nothing reads it any more.
    assert "agent_spec" not in settings_schema["properties"]


def test_descriptor_declares_relay_telemetry_support() -> None:
    """Without this, Fabric rejects the relay config agents auto-wires for every job.

    The Analyst then cannot run at all, rather than merely running untraced --
    the export is configured by the agents layer, which asks the descriptor
    first.
    """
    telemetry = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))["telemetry"]

    assert "relay" in telemetry["providers"]
    assert "atif" in telemetry["providers"]["relay"]["outputs"]


def test_fabric_plans_a_config_carrying_an_ethos(tmp_path: Path) -> None:
    """Planning is where a setting the descriptor omits actually blows up."""
    _require_installed_descriptor()

    config = _analyst_config(ethos="# Ethos\n\nBe careful.")
    assert config.harnesses["insights"].settings["ethos"] == "# Ethos\n\nBe careful."

    assert fabric.Fabric().plan(translate_agent_config(config), base_dir=tmp_path) is not None


def test_fabric_plans_the_built_config(tmp_path: Path) -> None:
    """Fabric validates the adapter descriptor while planning; this is the real check."""
    _require_installed_descriptor()

    plan = fabric.Fabric().plan(translate_agent_config(_analyst_config()), base_dir=tmp_path)

    assert plan is not None
