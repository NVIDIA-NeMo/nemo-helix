# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for user-selectable model configuration: env mapping, merge, and victim-model rewrite."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _doubles import make_entity, make_job_context, make_sdk
from nemo_agent_hardener_plugin.agent_resolver import inject_gateway_url
from nemo_agent_hardener_plugin.jobs import _common
from nemo_agent_hardener_plugin.jobs import run as run_module
from nemo_agent_hardener_plugin.model_config import ModelChoice, WarGameModels


class _FakeSecrets:
    """Stands in for the typed ``SecretsClient``, not the old ``sdk.secrets`` resource."""

    def __init__(self, values: dict[str, str]) -> None:
        self._values = values

    def access_secret(self, *, name: str, workspace: str) -> Any:
        return SimpleNamespace(data=lambda: SimpleNamespace(value=self._values.get(name)))


@pytest.fixture(autouse=True)
def _route_secrets_client(monkeypatch: Any) -> None:
    """``build_model_env`` resolves keys through ``client_from_platform(sdk, SecretsClient)``.

    The sdk doubles here carry their typed client on ``.mock_secrets``, so hand that back directly.
    """
    monkeypatch.setattr(_common, "client_from_platform", lambda sdk, _cls: sdk.mock_secrets)


def _sdk(secrets: dict[str, str]) -> Any:
    return SimpleNamespace(mock_secrets=_FakeSecrets(secrets))


def test_build_model_env_maps_attack_and_analysis_groups() -> None:
    models = WarGameModels(
        attack=ModelChoice(model="atk/model", base_url="https://atk/v1", api_key_secret="atk-key"),
        analysis=ModelChoice(model="ana/model", base_url="https://ana/v1", api_key_secret="ana-key"),
    )
    env = _common.build_model_env(models, sdk=_sdk({"atk-key": "AK", "ana-key": "NK"}), workspace="default")
    assert env["GARAK_RED_TEAM_MODEL_NAME"] == "atk/model"
    assert env["GARAK_DETECTOR_MODEL_NAME"] == "atk/model"
    assert env["GARAK_RED_TEAM_MODEL_URI"] == "https://atk/v1"
    assert env["GARAK_DETECTOR_MODEL_URI"] == "https://atk/v1"
    assert env["NIM_API_KEY"] == "AK"
    assert env["AGENT_HARDENER_MODEL"] == "ana/model"
    assert env["AGENT_HARDENER_BASE_URL"] == "https://ana/v1"
    assert env["INFERENCE_API_KEY"] == "NK"


def test_build_model_env_skips_unset_fields_and_safety_group() -> None:
    # Only a model name for analysis; no base_url/secret. Safety is never an env knob — it travels in
    # the manifest, as the guardrails defender entry's `config`.
    models = WarGameModels(analysis=ModelChoice(model="ana/model"), safety=ModelChoice(model="guard/model"))
    env = _common.build_model_env(models, sdk=_sdk({}), workspace="default")
    assert env == {"AGENT_HARDENER_MODEL": "ana/model"}


def test_build_model_env_none_is_empty() -> None:
    assert _common.build_model_env(None, sdk=_sdk({}), workspace="default") == {}


def test_effective_models_merges_override_over_stored_default(tmp_path: Path) -> None:
    stored = {"attack": {"model": "stored/atk", "base_url": "https://stored/v1"}}
    sdk = make_sdk(SimpleNamespace(get_entity_by_name=lambda **_k: make_entity(models=stored)))
    config = {"manifest_id": "m1", "models": {"attack": {"model": "override/atk"}, "analysis": {"model": "ana"}}}
    merged = run_module._effective_models(sdk, config, ctx=make_job_context(tmp_path))
    assert merged is not None
    # override wins per field; the stored base_url is preserved where the override left it unset.
    assert merged.attack is not None and merged.attack.model == "override/atk"
    assert merged.attack.base_url == "https://stored/v1"
    assert merged.analysis is not None and merged.analysis.model == "ana"


def test_effective_models_none_when_nothing_selected(tmp_path: Path) -> None:
    sdk = make_sdk(SimpleNamespace(get_entity_by_name=lambda **_k: make_entity()))
    config = {"manifest_id": "m1"}
    assert run_module._effective_models(sdk, config, ctx=make_job_context(tmp_path)) is None


def test_inject_gateway_url_never_rewrites_the_victims_model() -> None:
    """The victim's LLM is the target under test; the war-game binds its endpoint, never its model."""
    config = {"models": {"main": {"provider": "nvidia", "model": "orig"}, "other": {"model": "orig2"}}}
    injected = inject_gateway_url(config, "default", "https://gw")
    assert injected["models"]["main"]["model"] == "orig"
    assert injected["models"]["other"]["model"] == "orig2"
    # The endpoint is bound; the credential is not written into the config at all (the NAT path
    # used to leave a literal "not-used" api_key here — Fabric binds it separately).
    assert "/apis/inference-gateway/" in injected["models"]["main"]["base_url"]
    assert "api_key" not in injected["models"]["main"]


def test_preflight_probes_the_safety_group(monkeypatch: Any) -> None:
    """An unreachable safety model hangs the guardrails defender to its timeout, so fail fast instead."""
    import pytest
    from nemo_agent_hardener_plugin.jobs.errors import AgentHardenerRunError
    from nemo_agent_hardener_plugin.model_config import ATTACK_DEFAULT_BASE_URL

    probed: list[tuple[str | None, str | None]] = []

    def _validate(model: str | None, base_url: str | None, key: str | None) -> Any:
        probed.append((model, base_url))
        return SimpleNamespace(ok=False, reason="auth", detail="key not allowed", available=["good/model"])

    monkeypatch.setattr(run_module, "validate_choice", _validate)
    models = WarGameModels(safety=ModelChoice(model="bad/model"))

    with pytest.raises(AgentHardenerRunError, match="safety model"):
        run_module._preflight_models(models, sdk=None, workspace="default", default_key="k")

    # Probed against the endpoint agent-hardener pins for the guardrail LLM, not a made-up one.
    assert probed == [("bad/model", ATTACK_DEFAULT_BASE_URL)]


class _Secrets:
    """Secrets double: `access` raises for an unknown name, as the real client does."""

    def __init__(self, values: dict[str, str], *, reachable: bool = True) -> None:
        self._values, self._reachable = values, reachable

    def access_secret(self, *, name: str, workspace: str) -> Any:
        if not self._reachable:
            raise RuntimeError("secrets store unreachable")
        if name not in self._values:
            raise KeyError(name)
        return SimpleNamespace(data=lambda: SimpleNamespace(value=self._values[name]))


def _sdk_with(secrets: _Secrets, monkeypatch: Any) -> Any:
    """An sdk double carrying its typed client on ``.mock_secrets``, per ``_route_secrets_client``."""
    del monkeypatch  # routing is handled by the autouse fixture
    return SimpleNamespace(mock_secrets=secrets)


def _config_double(monkeypatch: Any, dotenv_key: str | None) -> None:
    from nemo_agent_hardener_plugin.jobs import _common as common

    monkeypatch.setattr(
        common.AgentHardenerConfig,
        "get",
        classmethod(lambda _cls: SimpleNamespace(inference_secret_name="agent-hardener-inference-key")),
    )
    monkeypatch.setattr(
        common, "build_subprocess_env", lambda _c, _e=None: {"INFERENCE_API_KEY": dotenv_key} if dotenv_key else {}
    )


def test_resolve_model_key_prefers_the_named_secret(monkeypatch: Any) -> None:
    from nemo_agent_hardener_plugin.jobs import _common as common

    _config_double(monkeypatch, "from-dotenv")
    sdk = _sdk_with(_Secrets({"my-key": "chosen", "agent-hardener-inference-key": "provisioned"}), monkeypatch)

    assert common.resolve_model_key(sdk, "my-key", workspace="default") == "chosen"


def test_resolve_model_key_falls_back_to_the_provisioned_secret(monkeypatch: Any) -> None:
    """A null api_key_secret means "the platform's provisioned agent-hardener key" — the documented default."""
    from nemo_agent_hardener_plugin.jobs import _common as common

    _config_double(monkeypatch, "from-dotenv")
    sdk = _sdk_with(_Secrets({"agent-hardener-inference-key": "provisioned"}), monkeypatch)

    assert common.resolve_model_key(sdk, None, workspace="default") == "provisioned"


def test_resolve_model_key_falls_back_to_the_dotenv(monkeypatch: Any) -> None:
    """The offline path: no such secret (or no store) still resolves the key `setup` wrote."""
    from nemo_agent_hardener_plugin.jobs import _common as common

    _config_double(monkeypatch, "from-dotenv")

    absent = _sdk_with(_Secrets({}), monkeypatch)
    assert common.resolve_model_key(absent, None, workspace="default") == "from-dotenv"

    unreachable = _sdk_with(_Secrets({}, reachable=False), monkeypatch)
    assert common.resolve_model_key(unreachable, None, workspace="default") == "from-dotenv"


def test_resolve_model_key_none_when_nothing_resolves(monkeypatch: Any) -> None:
    from nemo_agent_hardener_plugin.jobs import _common as common

    _config_double(monkeypatch, None)
    assert common.resolve_model_key(_sdk_with(_Secrets({}), monkeypatch), None, workspace="default") is None
