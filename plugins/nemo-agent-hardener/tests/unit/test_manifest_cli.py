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
    monkeypatch.setattr(_shared.checks, "require_preflight", lambda _c, **_k: None)
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


def test_manifest_set_replaces_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    captured: dict[str, Any] = {}

    def _update(name: str, *, workspace: str, **body: Any) -> dict:
        captured["body"] = body
        return {"name": name}

    app = _patch_cli(cli_main, monkeypatch, SimpleNamespace(update=_update))
    result = CliRunner().invoke(
        app,
        [
            "manifest",
            "set",
            "finance",
            "--backend",
            "finance:8086",
            "--backend",
            "ledger:8087",
            "--backend-host",
            "ledger=fd00::12",
            "--backend-allowed-ip",
            "ledger=fd00::/8",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["body"] == {
        "backends": [
            "finance:8086",
            {"name": "ledger", "ports": ["8087"], "host": "fd00::12", "allowed_ips": ["fd00::/8"]},
        ]
    }


def test_manifest_set_rejects_a_backend_host_without_its_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    def _update(*_args: Any, **_kwargs: Any) -> dict:  # pragma: no cover - must not run
        raise AssertionError("update should not be called")

    app = _patch_cli(cli_main, monkeypatch, SimpleNamespace(update=_update))
    result = CliRunner().invoke(app, ["manifest", "set", "finance", "--backend-host", "ledger=ledger.ns.svc"])

    assert result.exit_code == 1
    assert "no --backend named ledger" in result.output


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
    from nemo_agent_hardener_plugin.model_config import ANALYSIS_DEFAULT_BASE_URL, SAFETY_DEFAULT_BASE_URL

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
    assert by_url["s"] == SAFETY_DEFAULT_BASE_URL


def test_manifest_set_probes_safety_on_its_own_endpoint_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    probes: list[dict[str, Any]] = []
    captured: dict[str, Any] = {}
    app = _patch_cli(
        cli_main,
        monkeypatch,
        _manifests(probes=probes, update=lambda name, *, workspace, **body: captured.update(body) or {"name": name}),
    )
    result = CliRunner().invoke(
        app,
        [
            "manifest",
            "set",
            "finance",
            "--safety-model",
            "s",
            "--safety-base-url",
            "http://10.0.0.5:8000/v1",
            "--safety-key-secret",
            "safety-key",
        ],
    )

    assert result.exit_code == 0, result.output
    assert probes == [{"model": "s", "base_url": "http://10.0.0.5:8000/v1", "api_key_secret": "safety-key"}]
    assert captured["models"]["safety"] == {
        "model": "s",
        "base_url": "http://10.0.0.5:8000/v1",
        "api_key_secret": "safety-key",
    }


def test_manifest_set_probes_safety_on_the_stored_analysis_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting only the safety model must probe where the judge will go: the stored analysis endpoint and key."""
    from nemo_agent_hardener_plugin.cli import main as cli_main

    stored = {"analysis": {"base_url": "http://10.0.0.9/v1", "api_key_secret": "analysis-key"}}
    probes: list[dict[str, Any]] = []
    app = _patch_cli(
        cli_main,
        monkeypatch,
        _manifests(
            probes=probes,
            get=lambda name, *, workspace: {"models": stored},
            update=lambda name, *, workspace, **body: {"name": name},
        ),
    )
    result = CliRunner().invoke(app, ["manifest", "set", "finance", "--safety-model", "s"])

    assert result.exit_code == 0, result.output
    assert probes == [{"model": "s", "base_url": "http://10.0.0.9/v1", "api_key_secret": "analysis-key"}]


def test_manifest_set_reprobes_a_stored_safety_model_when_analysis_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stored safety model inherits the analysis endpoint and key, so moving analysis moves the judge too."""
    from nemo_agent_hardener_plugin.cli import main as cli_main

    stored = {"safety": {"model": "s"}}
    probes: list[dict[str, Any]] = []
    app = _patch_cli(
        cli_main,
        monkeypatch,
        _manifests(
            probes=probes,
            get=lambda name, *, workspace: {"models": stored},
            update=lambda name, *, workspace, **body: {"name": name},
        ),
    )
    result = CliRunner().invoke(
        app, ["manifest", "set", "finance", "--analysis-base-url", "http://10.0.0.9/v1", "--analysis-key-secret", "k"]
    )

    assert result.exit_code == 0, result.output
    assert {"model": "s", "base_url": "http://10.0.0.9/v1", "api_key_secret": "k"} in probes


def test_manifest_set_probes_a_safety_key_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key with no model or endpoint still reaches the judge, so it is checked against the built-in judge model."""
    from nemo_agent_hardener_plugin.cli import main as cli_main
    from nemo_agent_hardener_plugin.model_config import SAFETY_DEFAULT_BASE_URL, SAFETY_DEFAULT_MODEL

    probes: list[dict[str, Any]] = []
    app = _patch_cli(
        cli_main,
        monkeypatch,
        _manifests(probes=probes, update=lambda name, *, workspace, **body: {"name": name}),
    )
    result = CliRunner().invoke(app, ["manifest", "set", "finance", "--safety-key-secret", "guard-key"])

    assert result.exit_code == 0, result.output
    assert probes == [
        {"model": SAFETY_DEFAULT_MODEL, "base_url": SAFETY_DEFAULT_BASE_URL, "api_key_secret": "guard-key"}
    ]


def test_cli_init_rejects_an_unusable_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    def _create(*_a: Any, **_k: Any) -> dict:  # pragma: no cover - must not run
        raise AssertionError("an unusable model must not be stored")

    app = _patch_cli(cli_main, monkeypatch, _manifests(ok=False, create=_create))
    result = CliRunner().invoke(app, ["init", "--agent", "react-agent", "--attack-model", "bad/model"])

    assert result.exit_code == 1
    assert "not usable" in result.output


@pytest.mark.parametrize("image", ["reg.example/victim:1", ""])
def test_manifest_set_records_or_clears_the_victim_image(monkeypatch: pytest.MonkeyPatch, image: str) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    captured: dict[str, Any] = {}
    app = _patch_cli(cli_main, monkeypatch, SimpleNamespace(update=lambda _name, **kw: captured.update(kw) or {}))

    result = CliRunner().invoke(app, ["manifest", "set", "finance", "--victim-image", image])

    assert result.exit_code == 0, result.output
    assert captured == {"workspace": "default", "victim_image": image}


def _patch_stage(cli_main: Any, monkeypatch: pytest.MonkeyPatch, result: dict[str, Any]) -> Any:
    app = _patch_cli(cli_main, monkeypatch, SimpleNamespace())
    fake = SimpleNamespace(stage_victim_context=lambda **_kw: result)
    monkeypatch.setattr(_shared, "make_sdk", lambda _u: SimpleNamespace(agent_hardener=fake))
    return app


def test_stage_prints_the_build_push_and_record_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    app = _patch_stage(
        cli_main, monkeypatch, {"status": "completed", "context_archive": "/tmp/job dir/finance-victim-context.tar.gz"}
    )

    result = CliRunner().invoke(app, ["stage", "--manifest-id", "finance", "--workspace", "team"])

    assert result.exit_code == 0, result.output
    assert "docker build -t <registry>/<image>:<tag> - < '/tmp/job dir/finance-victim-context.tar.gz'" in result.output
    assert "nemo agent-hardener manifest set finance --workspace team --victim-image" in result.output


def test_stage_failure_shows_the_cause_and_its_remedy(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    error = {"message": "agent-hardener is not provisioned", "remediation": "Run `nemo agent-hardener setup`."}
    app = _patch_stage(cli_main, monkeypatch, {"status": "failed", "error": error})

    result = CliRunner().invoke(app, ["stage", "--manifest-id", "finance"])

    assert result.exit_code == 1
    assert "agent-hardener is not provisioned" in result.output
    assert "Run `nemo agent-hardener setup`." in result.output


def test_stage_a_local_manifest_points_at_victim_image_instead_of_manifest_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    seen: dict[str, Any] = {}
    app = _patch_cli(cli_main, monkeypatch, SimpleNamespace())
    fake = SimpleNamespace(
        stage_victim_context=lambda **kw: seen.update(kw) or {"status": "completed", "context_archive": "/c.tar.gz"}
    )
    monkeypatch.setattr(_shared, "make_sdk", lambda _u: SimpleNamespace(agent_hardener=fake))
    manifest = tmp_path / "agent-hardener.yaml"
    manifest.write_text("agent: {}\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["stage", "--config", str(manifest)])

    assert result.exit_code == 0, result.output
    assert seen["config"] == str(manifest)
    assert seen["manifest_id"] is None
    assert "--victim-image" in result.output
    assert "manifest set" not in result.output


@pytest.mark.parametrize("args", [[], ["--manifest-id", "m1", "--config", "a.yaml"]])
def test_stage_needs_exactly_one_manifest(args: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_agent_hardener_plugin.cli import main as cli_main

    app = _patch_stage(cli_main, monkeypatch, {"status": "completed", "context_archive": "/c.tar.gz"})

    result = CliRunner().invoke(app, ["stage", *args])

    assert result.exit_code == 1
    assert "exactly one of --manifest-id or --config" in result.output
