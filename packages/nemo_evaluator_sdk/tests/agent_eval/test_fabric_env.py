# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fabric configuration preparation needs neither native Fabric nor a sandbox."""

import copy
from typing import Any

import pytest
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.env import (
    host_fabric_config,
    model_fabric_config,
    validate_fabric_env,
)
from nemo_evaluator_sdk.resolver_protocols import MissingSecretError
from nemo_evaluator_sdk.values.common import SecretRef


class _Source:
    def __init__(self, **names: str) -> None:
        self.names = names

    def env_var_for(self, secret_ref: SecretRef, env_name: str) -> str:
        if env_name not in self.names:
            raise MissingSecretError(f"secret {secret_ref.root!r} is missing; export SOURCE_KEY")
        return self.names[env_name]


_REFS = {"KEY": SecretRef("ws/key")}
_ROLE = {
    "provider": "nvidia",
    "model": "nvidia/old",
    "api_key_env": "KEY",
    "base_url": "https://example.test/v1",
    "temperature": 0.3,
    "settings": {"nested": [1]},
}


@pytest.mark.parametrize("config", [{}, {"environment": None}, {"environment": {}}, {"environment": {"env": None}}])
def test_optional_environment_sections(config: dict[str, Any]) -> None:
    validate_fabric_env(config, _REFS)


@pytest.mark.parametrize(
    "config,field",
    [({"environment": "local"}, "config.environment"), ({"environment": {"env": []}}, "config.environment.env")],
)
def test_environment_shape_errors(config: dict[str, Any], field: str) -> None:
    with pytest.raises(ValueError, match=field):
        validate_fabric_env(config, _REFS)


def test_declared_secret_cannot_be_overridden_by_config_env() -> None:
    with pytest.raises(ValueError, match="KEY"):
        validate_fabric_env({"environment": {"env": {"KEY": "literal"}}}, _REFS)


@pytest.mark.parametrize("model", [None, ""])
def test_no_override_returns_a_detached_unchanged_config(model: str | None) -> None:
    config = {"models": {"default": copy.deepcopy(_ROLE)}}
    prepared = model_fabric_config(config, model=model)
    assert prepared == config
    prepared["models"]["default"]["settings"]["nested"].append(2)
    assert config["models"]["default"] == _ROLE


@pytest.mark.parametrize("role", ["default", "primary"])
def test_same_provider_override_preserves_selected_model_settings(role: str) -> None:
    config = {"models": {role: copy.deepcopy(_ROLE)}}
    before = copy.deepcopy(config)
    prepared = model_fabric_config(config, model="nvidia/new")
    assert prepared["models"]["default"] == {**_ROLE, "model": "nvidia/new"}
    prepared["models"]["default"]["settings"]["nested"].append(2)
    assert config == before
    if role == "primary":
        assert prepared["models"][role] == _ROLE


@pytest.mark.parametrize(
    "connection", [{"api_key_env": "KEY"}, {"base_url": "https://example.test"}, {"settings": {"x": 1}}]
)
def test_provider_switch_cannot_inherit_connection_settings(connection: dict[str, Any]) -> None:
    config = {"models": {"primary": {"provider": "nvidia", "model": "old", **connection}}}
    before = copy.deepcopy(config)
    with pytest.raises(ValueError, match="changes provider.*nvidia.*openai.*models.primary"):
        model_fabric_config(config, model="openai/new")
    assert config == before


@pytest.mark.parametrize(
    "config,model,provider",
    [
        ({}, "new", "openai"),
        ({}, "nvidia/new", "nvidia"),
        ({"models": {"default": {"provider": "nvidia", "model": "old"}}}, "openai/new", "openai"),
        ({"models": {"a": _ROLE, "b": _ROLE}}, "openai/new", "openai"),
    ],
)
def test_override_without_a_selected_connection_creates_a_fresh_default(
    config: dict[str, Any], model: str, provider: str
) -> None:
    assert model_fabric_config(config, model=model)["models"]["default"] == {"provider": provider, "model": model}


@pytest.mark.parametrize(
    "models,field",
    [
        (None, "config.models"),
        ([], "config.models"),
        ({"default": "bad"}, "models.default"),
        ({"default": {"api_key_env": []}}, "api_key_env"),
    ],
)
def test_model_shape_errors(models: Any, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        host_fabric_config({"models": models}, {}, _Source(), model=None)


@pytest.mark.parametrize("field,value", [("provider", None), ("provider", ""), ("base_url", []), ("settings", [])])
def test_override_checks_fields_before_reading_them(field: str, value: Any) -> None:
    config = {"models": {"default": {**_ROLE, field: value}}}
    with pytest.raises(ValueError, match=field):
        model_fabric_config(config, model="nvidia/new")


@pytest.mark.parametrize("provider", ["nvidia", "openai"])
@pytest.mark.parametrize("role", ["default", "primary"])
def test_host_rewrites_all_matching_roles_after_override(provider: str, role: str) -> None:
    model_role = {**_ROLE, "provider": provider}
    config = {"models": {role: model_role}, "environment": {"env": {"MODE": "test"}}}
    before = copy.deepcopy(config)
    prepared = host_fabric_config(config, _REFS, _Source(KEY="WS_KEY"), model=f"{provider}/new")
    assert prepared["models"]["default"] == {**model_role, "model": f"{provider}/new", "api_key_env": "WS_KEY"}
    if role == "primary":
        assert prepared["models"][role]["api_key_env"] == "WS_KEY"
    assert prepared["environment"] == config["environment"]
    assert config == before


def test_host_keeps_bare_bindings_and_returns_an_independent_config() -> None:
    config = {"models": {"default": copy.deepcopy(_ROLE)}}
    prepared = host_fabric_config(config, _REFS, _Source(KEY="KEY"), model=None)
    assert prepared == config
    prepared["models"]["default"]["settings"]["nested"].append(2)
    assert config["models"]["default"] == _ROLE


@pytest.mark.parametrize("models", [{}, {"default": {"provider": "nvidia", "model": "m"}}])
def test_unusable_prefixed_secret_has_conditional_model_and_tool_advice(models: dict[str, Any]) -> None:
    config = {"models": models}
    before = copy.deepcopy(config)
    with pytest.raises(
        ValueError, match="If KEY is a model credential.*sandbox=.*unset WS_KEY.*bare env name.*adapter"
    ):
        host_fabric_config(config, _REFS, _Source(KEY="WS_KEY"), model=None)
    assert config == before


def test_missing_secret_preserves_the_source_hint() -> None:
    with pytest.raises(MissingSecretError, match=r"env_secrets\['KEY'\].*export SOURCE_KEY"):
        host_fabric_config({}, _REFS, _Source(), model=None)


def test_config_env_cannot_override_the_selected_source_variable() -> None:
    config = {"models": {"default": _ROLE}, "environment": {"env": {"WS_KEY": "literal"}}}
    before = copy.deepcopy(config)
    with pytest.raises(ValueError, match="WS_KEY"):
        host_fabric_config(config, _REFS, _Source(KEY="WS_KEY"), model=None)
    assert config == before


def test_host_results_are_independent_across_sources_and_preserve_other_roles() -> None:
    config = {
        "models": {
            "default": copy.deepcopy(_ROLE),
            "judge": {"provider": "openai", "model": "j", "api_key_env": "JUDGE_KEY"},
        }
    }
    before = copy.deepcopy(config)
    first = host_fabric_config(config, _REFS, _Source(KEY="FIRST_KEY"), model=None)
    second = host_fabric_config(config, _REFS, _Source(KEY="SECOND_KEY"), model=None)
    first["models"]["default"]["settings"]["nested"].append(2)
    assert first["models"]["default"]["api_key_env"] == "FIRST_KEY"
    assert second["models"]["default"] == {**_ROLE, "api_key_env": "SECOND_KEY"}
    assert second["models"]["judge"] == before["models"]["judge"]
    assert config == before


def test_source_aliases_do_not_cascade_through_other_secret_keys() -> None:
    config = {"models": {"default": {**_ROLE, "api_key_env": "A"}, "judge": {**_ROLE, "api_key_env": "B"}}}
    refs = {"A": SecretRef("ws/a"), "B": SecretRef("ws/b")}
    prepared = host_fabric_config(config, refs, _Source(A="B", B="C"), model=None)
    assert prepared["models"]["default"]["api_key_env"] == "B"
    assert prepared["models"]["judge"]["api_key_env"] == "C"
