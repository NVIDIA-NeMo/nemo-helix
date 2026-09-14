# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the sanity-check selection helpers, the SDK sanity_check method, and the CLI command."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _doubles import make_sdk
from nemo_agent_hardener_plugin.cli import _shared
from nemo_agent_hardener_plugin.jobs.defenses import defense_ids, select_defense_ids
from typer.testing import CliRunner


def _mitigations() -> dict:
    guardrails = (
        "version = 1\n"
        "[[components]]\n"
        'kind = "agent_hardener.pre_tool_verifier"\n'
        "[[components.config.guardrails]]\n"
        'name = "custom_guardrail_1"\n'
        "[[components.config.guardrails]]\n"
        'name = "custom_guardrail_2"\n'
    )
    return {
        "guardrails": {"before": "version = 1\n", "after": guardrails},
        "policy": {"before": "v: 1\n", "after": "v: 1\nhardened: true\n"},
        "defenses": [
            {"id": "custom_guardrail_1", "kind": "guardrail"},
            {"id": "custom_guardrail_2", "kind": "guardrail"},
            {"id": "openshell_policy", "kind": "policy"},
        ],
    }


def test_defense_ids_reads_ids() -> None:
    assert defense_ids(_mitigations()) == ["custom_guardrail_1", "custom_guardrail_2", "openshell_policy"]


def test_select_defense_ids_keep_exclude_default() -> None:
    all_ids = ["custom_guardrail_1", "custom_guardrail_2", "openshell_policy"]
    assert select_defense_ids(all_ids) == all_ids
    assert select_defense_ids(all_ids, keep=["custom_guardrail_1", "openshell_policy"]) == [
        "custom_guardrail_1",
        "openshell_policy",
    ]
    assert select_defense_ids(all_ids, exclude=["custom_guardrail_2"]) == ["custom_guardrail_1", "openshell_policy"]


def test_sdk_sanity_check_composes_and_builds_validate_only_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin import sdk as sdk_module

    captured: dict[str, Any] = {}

    class _Scheduler:
        def submit_remote(self, _job: Any, spec: dict, **kwargs: Any) -> dict:
            captured["spec"] = spec
            captured["kwargs"] = kwargs
            return {"name": "job-x"}

    monkeypatch.setattr(sdk_module, "NemoJobScheduler", _Scheduler)
    monkeypatch.setattr(sdk_module, "base_url", lambda: "http://localhost:8080")

    resource = sdk_module.AgentHardenerPluginResource(make_sdk())
    resource.sanity_check(
        manifest_id="clockbot-hardening",
        mitigations=_mitigations(),
        selected_defense_ids=["custom_guardrail_1"],  # drop guardrail_2 and the policy
        replay_hitlog_fileset="default/hits",
    )

    spec = captured["spec"]
    assert spec["validate_only"] is True
    assert spec["driver"] == "service"
    assert spec["replay_hitlog_fileset"] == "default/hits"
    # The composed guardrail set keeps only guardrail_1; policy not selected → baseline policy.
    composed = tomllib.loads(spec["defense_guardrails"])
    rails = [rail["name"] for entry in composed["components"] for rail in entry["config"]["guardrails"]]
    assert rails == ["custom_guardrail_1"]
    assert spec["defense_policy"] == "v: 1\n"


def test_cli_sanity_check_selects_and_submits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    mitigations_file = tmp_path / "mitigations.json"
    mitigations_file.write_text(json.dumps(_mitigations()), encoding="utf-8")

    captured: dict[str, Any] = {}
    fake_sdk = SimpleNamespace(
        agent_hardener=SimpleNamespace(sanity_check=lambda **kwargs: captured.update(kwargs) or {"name": "job-x"})
    )
    monkeypatch.setattr(_shared.checks, "require_preflight", lambda _c: None)
    monkeypatch.setattr(_shared, "make_sdk", lambda _u: fake_sdk)
    monkeypatch.setattr(_shared, "base_url", lambda: "http://localhost:8080")
    monkeypatch.setattr(
        _shared.AgentHardenerConfig, "get", classmethod(lambda _cls: SimpleNamespace(default_workspace="default"))
    )

    app = cli_main.AgentHardenerCLI().get_cli()
    result = CliRunner().invoke(
        app,
        [
            "sanity-check",
            "--manifest-id",
            "clockbot-hardening",
            "--mitigations",
            str(mitigations_file),
            "--replay-hitlog",
            "default/hits",
            "--exclude",
            "custom_guardrail_2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["manifest_id"] == "clockbot-hardening"
    assert captured["replay_hitlog_fileset"] == "default/hits"
    # --exclude drops guardrail_2; the rest are kept.
    assert captured["selected_defense_ids"] == ["custom_guardrail_1", "openshell_policy"]


def test_cli_sanity_check_rejects_keep_and_exclude_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    mitigations_file = tmp_path / "mitigations.json"
    mitigations_file.write_text(json.dumps(_mitigations()), encoding="utf-8")
    monkeypatch.setattr(_shared.checks, "require_preflight", lambda _c: None)
    monkeypatch.setattr(
        _shared.AgentHardenerConfig, "get", classmethod(lambda _cls: SimpleNamespace(default_workspace="default"))
    )

    app = cli_main.AgentHardenerCLI().get_cli()
    result = CliRunner().invoke(
        app,
        [
            "sanity-check",
            "--manifest-id",
            "m1",
            "--mitigations",
            str(mitigations_file),
            "--replay-hitlog",
            "default/hits",
            "--keep",
            "custom_guardrail_1",
            "--exclude",
            "custom_guardrail_2",
        ],
    )

    assert result.exit_code == 1
    assert "either --keep or --exclude" in result.output
