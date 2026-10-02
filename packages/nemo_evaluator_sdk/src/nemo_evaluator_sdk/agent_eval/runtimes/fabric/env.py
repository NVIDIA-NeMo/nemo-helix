# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Fabric model and secret bindings without importing the execution stack.

Job submission validates raw config mappings through this module. Check the shapes of fields used
by binding preparation explicitly so API validation needs no native Fabric or execution dependencies.
"""

import copy
from collections.abc import Mapping
from typing import Any

from nemo_evaluator_sdk.agent_eval.runtimes.secrets import env_secret_vars
from nemo_evaluator_sdk.resolver_protocols import EnvSecretSource
from nemo_evaluator_sdk.values.common import SecretRef


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    """Require a mapping, naming the config field in errors instead of failing on dictionary access."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _environment_vars(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read optional environment values for collision checks; reject malformed non-mapping sections."""
    environment = config.get("environment")
    if environment is None:
        return {}
    env = _mapping(environment, "config.environment").get("env")
    return {} if env is None else _mapping(env, "config.environment.env")


def validate_fabric_env(config: Mapping[str, Any], env_secrets: Mapping[str, SecretRef]) -> None:
    """Reject config environment values that would override declared secret bindings."""
    collisions = _environment_vars(config).keys() & env_secrets.keys()
    if collisions:
        raise ValueError(
            f"config.environment.env declares {', '.join(sorted(collisions))}, also provided by env_secrets"
        )


def _model_roles(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Copy entries keyed by model role, checking credential names before selection or rewriting.

    A role is a caller-chosen key in ``config['models']``, such as ``default`` or ``judge``.
    Each entry must be a mapping whose optional ``api_key_env`` is a string or ``None``.
    """
    roles = {}
    for name, value in _mapping(config.get("models", {}), "config.models").items():
        role = dict(_mapping(value, f"config.models.{name}"))
        if role.get("api_key_env") is not None and not isinstance(role["api_key_env"], str):
            raise ValueError(f"config.models.{name}.api_key_env must be a string or None")
        roles[name] = role
    return roles


def _selected_role(models: Mapping[str, Any]) -> str | None:
    """Select the default or sole role so an override preserves the connection the adapter would use.

    Mirrors Codex ``_selected_model_config``, Hermes ``_selected_model``, and Deepagents
    ``selected_model_config``. Return ``None`` for no models or multiple roles without a default;
    choosing an arbitrary entry could reuse an unrelated model's credentials or endpoint.
    """
    if "default" in models:
        return "default"
    return next(iter(models)) if len(models) == 1 else None


def model_fabric_config(config: Mapping[str, Any], *, model: str | None) -> dict[str, Any]:
    """Change the evaluation model without losing its configured connection and settings.

    Use this in both host and sandbox setup when evaluating another model through an existing
    provider. Replacing the entire model entry would discard ``api_key_env``, a custom ``base_url``,
    temperature, and provider settings, causing authentication failures or use of the wrong endpoint.

    Select ``models.default`` or the sole role (a key in ``config['models']``). For example,
    ``model='nvidia/new'`` preserves that entry's NVIDIA credential binding and custom endpoint.
    For the same provider, copy the selected entry into ``models.default`` and change only the
    provider/model fields. A provider switch with an explicit credential, endpoint, or non-empty
    provider settings raises ``ValueError``: the caller must configure the new connection explicitly.
    Without a selected connection to preserve, create a fresh default entry.

    Return an independent config even when ``model`` is absent or empty, so later task composition
    cannot mutate the caller's config or another evaluation's preparation.
    """
    prepared = copy.deepcopy(dict(config))
    models = _model_roles(prepared)
    if "models" in prepared:
        prepared["models"] = models
    if not model:
        return prepared
    provider = model.split("/", maxsplit=1)[0] if "/" in model else "openai"
    name = _selected_role(models)
    selected = models[name] if name is not None else {}
    if name is not None:
        original_provider = selected.get("provider")
        if not isinstance(original_provider, str) or not original_provider.strip():
            raise ValueError(f"config.models.{name}.provider must be a non-empty string")
        if selected.get("base_url") is not None and not isinstance(selected["base_url"], str):
            raise ValueError(f"config.models.{name}.base_url must be a string or None")
        if "settings" in selected:
            _mapping(selected["settings"], f"config.models.{name}.settings")
        if original_provider != provider:
            if (
                selected.get("api_key_env") is not None
                or selected.get("base_url") is not None
                or selected.get("settings")
            ):
                raise ValueError(
                    f"model= changes provider from {original_provider!r} to {provider!r} for models.{name}; "
                    "configure the new provider's credentials and endpoint in models.default and omit model="
                )
            selected = {}
    models["default"] = {**copy.deepcopy(selected), "provider": provider, "model": model}
    prepared["models"] = models
    return prepared


def host_fabric_config(
    config: Mapping[str, Any],
    env_secrets: Mapping[str, SecretRef],
    source: EnvSecretSource,
    *,
    model: str | None,
) -> dict[str, Any]:
    """Prepare model overrides and credential names before all tasks in one ``run_tasks`` call.

    A local secret may live in ``WS_KEY`` while the model's ``api_key_env`` names ``KEY``. Repoint
    matching model entries at the selected source variable so the host adapter can read it without
    copying credentials into Fabric config or changing the process environment. Reject conflicting
    config values and prefixed secrets without an explicit model binding.

    Resolve names afresh for each evaluation and raise configuration errors before any task starts,
    rather than recording the same setup failure on individual trials. The prepared config contains
    credential variable names only; adapters read their values later. Each task copies this independent
    config before applying its workspace and trajectory settings.
    """
    prepared = model_fabric_config(config, model=model)
    validate_fabric_env(prepared, env_secrets)
    sources = env_secret_vars(env_secrets, source)
    models = _model_roles(prepared)
    env = _environment_vars(prepared)
    for key, variable in sources.items():
        if variable == key:
            continue
        if variable in env:
            raise ValueError(f"config.environment.env declares {variable}, which holds env_secrets[{key!r}]")
        if not any(role.get("api_key_env") == key for role in models.values()):
            raise ValueError(
                f"env_secrets[{key!r}] -> secret {env_secrets[key].root!r} was found in {variable}, "
                f"but a Fabric harness on the host reads {key} itself and no models.*.api_key_env names it explicitly. "
                f"If {key} is a model credential, set models.<role>.api_key_env: {key} in the config. "
                f"Otherwise run with sandbox=, or unset {variable} so the lookup can fall back to {key} "
                f"(only when {key} is the secret's bare env name). Either way the variable is set as {key}; "
                "whether the harness forwards it to its tools depends on the adapter."
            )
    # Each role is rewritten from its pre-rewrite key once: A -> B and B -> C must not turn A into C.
    for role in models.values():
        key = role.get("api_key_env")
        if key in sources:
            role["api_key_env"] = sources[key]
    if "models" in prepared:
        prepared["models"] = models
    return prepared
