# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `nemo agent-hardener manifest show|set` commands (stored defaults, not per-run overrides)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nemo_agent_hardener_plugin.cli import _shared
from typer.testing import CliRunner


def _patch_cli(cli_main: Any, monkeypatch: pytest.MonkeyPatch, manifests: Any) -> Any:
    fake_sdk = SimpleNamespace(agent_hardener=SimpleNamespace(manifests=manifests))
    monkeypatch.setattr(_shared.checks, "require_preflight", lambda _c: None)
    monkeypatch.setattr(_shared, "make_sdk", lambda _u: fake_sdk)
    monkeypatch.setattr(_shared, "base_url", lambda: "http://localhost:8080")
    monkeypatch.setattr(
        _shared.AgentHardenerConfig,
        "get",
        classmethod(lambda _cls: SimpleNamespace(default_workspace="default", operator_env_file=Path(".env"))),
    )
    return cli_main.AgentHardenerCLI().get_cli()


def test_manifest_show_prints_the_record(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    record = {"name": "finance", "rounds": 3, "attack_intensity": "thorough"}
    manifests = SimpleNamespace(get=lambda name, *, workspace: record)
    app = _patch_cli(cli_main, monkeypatch, manifests)

    result = CliRunner().invoke(app, ["manifest", "show", "finance"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output[result.output.index("{") :]) == record


def test_manifest_set_builds_patch_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the flags given are sent, so an unset field keeps its stored value."""
    from nemo_agent_hardener_plugin.cli import main as cli_main

    captured: dict[str, Any] = {}

    def _update(name: str, *, workspace: str, **body: Any) -> dict:
        captured["name"] = name
        captured["workspace"] = workspace
        captured["body"] = body
        return {"name": name}

    app = _patch_cli(cli_main, monkeypatch, SimpleNamespace(update=_update))
    result = CliRunner().invoke(
        app,
        [
            "manifest",
            "set",
            "finance",
            "--rounds",
            "3",
            "--defender",
            "guardrails",
            "--egress",
            "example.com",
            "--env",
            "FINANCE_BACKEND_URL=http://host.docker.internal:8086",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["name"] == "finance"
    assert captured["body"] == {
        "rounds": 3,
        "defenders": ["guardrails"],
        "egress": ["example.com"],
        "env": {"FINANCE_BACKEND_URL": "http://host.docker.internal:8086"},
    }
    assert "attack_intensity" not in captured["body"]
    assert "port" not in captured["body"]


def test_manifest_set_requires_a_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """A no-op PATCH would report success while changing nothing."""
    from nemo_agent_hardener_plugin.cli import main as cli_main

    def _update(*_args: Any, **_kwargs: Any) -> dict:  # pragma: no cover - must not run
        raise AssertionError("update should not be called with an empty body")

    app = _patch_cli(cli_main, monkeypatch, SimpleNamespace(update=_update))
    result = CliRunner().invoke(app, ["manifest", "set", "finance"])

    assert result.exit_code == 1
    assert "at least one field" in result.output


def test_manifest_set_rejects_unknown_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same validation as `run`: a typo must not be absorbed into a silent default."""
    from nemo_agent_hardener_plugin.cli import main as cli_main

    def _update(*_args: Any, **_kwargs: Any) -> dict:  # pragma: no cover - must not run
        raise AssertionError("update should not be called with an invalid value")

    app = _patch_cli(cli_main, monkeypatch, SimpleNamespace(update=_update))

    intensity = CliRunner().invoke(app, ["manifest", "set", "finance", "--attack-intensity", "heavy"])
    assert intensity.exit_code == 1
    assert "light, standard, thorough" in intensity.output

    defender = CliRunner().invoke(app, ["manifest", "set", "finance", "--defender", "bogus"])
    assert defender.exit_code == 1
    assert "bogus" in defender.output


def test_manifest_set_merges_models_over_stored(monkeypatch: pytest.MonkeyPatch) -> None:
    """PATCH replaces `models` wholesale, so setting one group must not clear the others."""
    from nemo_agent_hardener_plugin.cli import main as cli_main

    stored = {
        "models": {
            "attack": {"model": "stored/atk", "base_url": "https://stored/v1"},
            "analysis": {"model": "stored/ana"},
        }
    }
    captured: dict[str, Any] = {}

    def _update(name: str, *, workspace: str, **body: Any) -> dict:
        captured["body"] = body
        return {"name": name}

    manifests = _manifests(get=lambda name, *, workspace: stored, update=_update)
    app = _patch_cli(cli_main, monkeypatch, manifests)
    result = CliRunner().invoke(app, ["manifest", "set", "finance", "--safety-model", "guard-1"])

    assert result.exit_code == 0, result.output
    assert captured["body"]["models"] == {
        "attack": {"model": "stored/atk", "base_url": "https://stored/v1"},  # untouched group survives
        "analysis": {"model": "stored/ana"},
        "safety": {"model": "guard-1"},
    }


def test_manifest_set_merges_within_a_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """Overriding a group's model keeps that group's other stored fields."""
    from nemo_agent_hardener_plugin.cli import main as cli_main

    stored = {"models": {"attack": {"model": "old", "base_url": "https://stored/v1"}}}
    captured: dict[str, Any] = {}

    manifests = _manifests(
        get=lambda name, *, workspace: stored,
        update=lambda name, *, workspace, **body: captured.update(body) or {"name": name},
    )
    app = _patch_cli(cli_main, monkeypatch, manifests)
    result = CliRunner().invoke(app, ["manifest", "set", "finance", "--attack-model", "new"])

    assert result.exit_code == 0, result.output
    assert captured["models"]["attack"] == {"model": "new", "base_url": "https://stored/v1"}


def test_manifest_set_without_model_flags_skips_the_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """No model flags means no merge, so the extra GET is not paid for."""
    from nemo_agent_hardener_plugin.cli import main as cli_main

    def _get(*_a: Any, **_k: Any) -> dict:  # pragma: no cover - must not run
        raise AssertionError("manifest should not be read when no model flag was passed")

    captured: dict[str, Any] = {}
    manifests = _manifests(get=_get, update=lambda name, *, workspace, **body: captured.update(body) or {"name": name})
    app = _patch_cli(cli_main, monkeypatch, manifests)
    result = CliRunner().invoke(app, ["manifest", "set", "finance", "--rounds", "2"])

    assert result.exit_code == 0, result.output
    assert "models" not in captured


def test_cli_init_sends_model_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """`init` accepts the same model flags, so a target can be fully configured in one call."""
    from nemo_agent_hardener_plugin.cli import main as cli_main

    captured: dict[str, Any] = {}
    manifests = _manifests(
        create=lambda *, workspace, **body: captured.update(body) or {"name": body["name"]},
    )
    app = _patch_cli(cli_main, monkeypatch, manifests)
    result = CliRunner().invoke(
        app,
        ["init", "--agent", "react-agent", "--attack-model", "atk/m", "--safety-model", "guard-1"],
    )

    assert result.exit_code == 0, result.output
    assert captured["models"] == {"attack": {"model": "atk/m"}, "safety": {"model": "guard-1"}}


def test_cli_init_omits_models_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    captured: dict[str, Any] = {}
    manifests = _manifests(
        create=lambda *, workspace, **body: captured.update(body) or {"name": body["name"]},
    )
    app = _patch_cli(cli_main, monkeypatch, manifests)
    result = CliRunner().invoke(app, ["init", "--agent", "react-agent"])

    assert result.exit_code == 0, result.output
    assert "models" not in captured


def _manifests(*, ok: bool = True, probes: list | None = None, **extra: Any) -> Any:
    """Manifest resource double; `validate_model` stands in for `POST /model-config/validate`."""
    verdict = (
        {"ok": True}
        if ok
        else {"ok": False, "reason": "auth", "detail": "key not allowed", "available": ["good/model"]}
    )

    def _validate_model(*, workspace: str, **body: Any) -> dict:
        if probes is not None:
            probes.append(body)
        return verdict

    extra.setdefault("get", lambda name, *, workspace: {"models": {}})
    return SimpleNamespace(validate_model=_validate_model, **extra)


def test_manifest_set_rejects_an_unusable_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stored default is only exercised at run time, so an unreachable one must not be written."""
    from nemo_agent_hardener_plugin.cli import main as cli_main

    def _update(*_a: Any, **_k: Any) -> dict:  # pragma: no cover - must not run
        raise AssertionError("an unusable model must not be stored")

    app = _patch_cli(cli_main, monkeypatch, _manifests(ok=False, update=_update))
    result = CliRunner().invoke(app, ["manifest", "set", "finance", "--safety-model", "bad/model"])

    assert result.exit_code == 1
    assert "not usable" in result.output
    assert "good/model" in result.output  # the reachable list is what makes the error actionable


def test_manifest_set_stores_a_usable_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    captured: dict[str, Any] = {}
    app = _patch_cli(
        cli_main,
        monkeypatch,
        _manifests(update=lambda name, *, workspace, **body: captured.update(body) or {"name": name}),
    )
    result = CliRunner().invoke(app, ["manifest", "set", "finance", "--safety-model", "good/model"])

    assert result.exit_code == 0, result.output
    assert captured["models"]["safety"] == {"model": "good/model"}


def test_manifest_set_probes_each_group_against_its_own_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe must target the endpoint the run will use, or it verifies the wrong thing."""
    from nemo_agent_hardener_plugin.cli import main as cli_main
    from nemo_agent_hardener_plugin.model_config import ANALYSIS_DEFAULT_BASE_URL, ATTACK_DEFAULT_BASE_URL

    probes: list[dict[str, Any]] = []
    app = _patch_cli(
        cli_main, monkeypatch, _manifests(probes=probes, update=lambda name, *, workspace, **body: {"name": name})
    )
    result = CliRunner().invoke(
        app,
        ["manifest", "set", "finance", "--analysis-model", "a", "--safety-model", "s"],
    )

    assert result.exit_code == 0, result.output
    by_url = {p["model"]: p["base_url"] for p in probes}
    assert by_url["a"] == ANALYSIS_DEFAULT_BASE_URL
    # agent-hardener pins the guardrail LLM to the attack endpoint, not the analysis one.
    assert by_url["s"] == ATTACK_DEFAULT_BASE_URL


def test_cli_init_rejects_an_unusable_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    def _create(*_a: Any, **_k: Any) -> dict:  # pragma: no cover - must not run
        raise AssertionError("an unusable model must not be stored")

    app = _patch_cli(cli_main, monkeypatch, _manifests(ok=False, create=_create))
    result = CliRunner().invoke(app, ["init", "--agent", "react-agent", "--attack-model", "bad/model"])

    assert result.exit_code == 1
    assert "not usable" in result.output
