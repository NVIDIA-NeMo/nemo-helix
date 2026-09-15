# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The "safety" model choice must reach the guardrails defender, which powers the generated guardrail.

agent-hardener's defenders manager builds each defender's context as ``{**enriched_context, **defender.config}``,
so the model rides in on the guardrails entry's ``config``. Unset, agent-hardener copies the victim's own LLM —
which is why leaving ``config`` absent (rather than writing a null) is part of the contract.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from nemo_agent_hardener_plugin.jobs.manifest import DEFENDER_ENTRIES, _apply_manifest_overrides

GUARDRAILS = DEFENDER_ENTRIES["guardrails"]["name"]


def _manifest() -> dict[str, Any]:
    return {"agent": {"name": "finance", "workflow": "agents/finance/workflow.yaml"}}


def _defenders(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return manifest.get("overrides", {}).get("defenders", [])


def _entry(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    return next(entry for entry in _defenders(manifest) if entry["name"] == name)


def test_safety_model_lands_on_the_guardrails_entry() -> None:
    manifest = _manifest()

    _apply_manifest_overrides(manifest, {"defenders": ["guardrails"], "models": {"safety": {"model": "guard-1"}}})

    assert _entry(manifest, GUARDRAILS)["config"]["safety_llm"] == "guard-1"


def test_only_the_guardrails_entry_carries_it() -> None:
    """openshell has no use for a safety LLM; giving it one would be noise in its context."""
    manifest = _manifest()

    _apply_manifest_overrides(
        manifest, {"defenders": ["guardrails", "openshell"], "models": {"safety": {"model": "guard-1"}}}
    )

    assert "config" not in _entry(manifest, DEFENDER_ENTRIES["openshell"]["name"])


def test_unset_safety_leaves_config_absent() -> None:
    """No `config` key means agent-hardener's own fallback (copy the victim's LLM) still applies."""
    manifest = _manifest()

    _apply_manifest_overrides(manifest, {"defenders": ["guardrails"]})

    assert "config" not in _entry(manifest, GUARDRAILS)


def test_module_level_entries_are_not_mutated() -> None:
    """DEFENDER_ENTRIES is module-level: an in-place edit would leak one run's model into the next."""
    _apply_manifest_overrides(_manifest(), {"defenders": ["guardrails"], "models": {"safety": {"model": "guard-1"}}})

    assert "config" not in DEFENDER_ENTRIES["guardrails"]

    second = _manifest()
    _apply_manifest_overrides(second, {"defenders": ["guardrails"]})
    assert "config" not in _entry(second, GUARDRAILS)


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        ({"defenders": [], "models": {"safety": {"model": "guard-1"}}}, "no defender selection"),
        ({"defenders": ["openshell"], "models": {"safety": {"model": "guard-1"}}}, "not enabled"),
    ],
)
def test_warns_when_it_cannot_apply(caplog: pytest.LogCaptureFixture, data: dict[str, Any], reason: str) -> None:
    """A dropped choice must be visible — silently ignoring it is the bug this feature exists to fix."""
    with caplog.at_level(logging.WARNING):
        _apply_manifest_overrides(_manifest(), data)

    assert reason in caplog.text


def test_safety_model_reaches_guardrails_without_a_workflow(caplog: pytest.LogCaptureFixture) -> None:
    """A workflow-less victim keeps the guardrails defender, so the safety model has a home."""
    manifest = {"agent": {"name": "finance"}}

    with caplog.at_level(logging.WARNING):
        _apply_manifest_overrides(manifest, {"defenders": ["guardrails"], "models": {"safety": {"model": "guard-1"}}})

    entry = manifest["overrides"]["defenders"][0]
    assert entry["name"] == "defender-guardrails"
    assert "guard-1" in json.dumps(entry["config"])
    assert "not applied" not in caplog.text
