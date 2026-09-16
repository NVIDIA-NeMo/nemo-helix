# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate Platform-owned agent config into typed in-memory FabricConfig."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import nemo_fabric as fabric
from nemo_agents_plugin.agent_config import AgentConfig, HarnessConfig, ModelConfig
from nemo_agents_plugin.fabric.gateway_credentials import (
    bind_platform_gateway_model_credential,
    platform_gateway_credential_env,
)

HARNESS_ADAPTER_IDS = {
    "claude": "nvidia.fabric.claude",
    "codex": "nvidia.fabric.codex",
    "deepagents": "nvidia.fabric.langchain.deepagents",
    "hermes": "nvidia.fabric.hermes",
}

# A harness kind carrying one of these prefixes is already a fully-qualified
# Fabric adapter id and is passed through verbatim. Plugin-owned adapters ship
# their own descriptor to <sys.prefix>/share/nemo-fabric/adapters/ and are
# resolved there by Fabric at runtime, so nemo-agents does not need a short-name
# entry for each one. A bad id therefore surfaces as a Fabric resolution error
# rather than a translation error here.
#
# `nvidia.fabric.` is Fabric's own namespace, which upstream keeps adding to;
# platform-owned adapters take `nvidia.nemo-platform.` instead. A duplicate
# adapter_id is not resolved by precedence -- Fabric refuses the id outright,
# environment-wide -- and nemo-agents-plugin depends on nemo-fabric, so both
# descriptors would land in the same environment. Minting platform adapters
# under `nvidia.fabric.` would make us a hard break away from an upstream
# naming choice.
#
# AIRCORE-1169 Workstream C promotes this to an AgentsConfig setting so an
# operator can sanction their own prefix for in-house adapters. It is a naming
# convention, not a security boundary: the adapter lives inside an image the
# caller already chose.
FABRIC_ADAPTER_ID_PREFIXES = ("nvidia.fabric.", "nvidia.nemo-platform.")

PLATFORM_RUNTIME_ENV_VARS = ("NEMO_BASE_URL", "NMP_BASE_URL", "NMP_WORKSPACE")


class FabricTranslationError(ValueError):
    """Raised when Platform agent config cannot be translated to Fabric config."""


def translate_agent_config(config: AgentConfig, harness_name: str | None = None) -> fabric.FabricConfig:
    """Translate Platform-owned agent config into a typed in-memory FabricConfig."""
    selected_harness_name, harness = _select_harness(config, harness_name)
    model = _resolve_model(config, selected_harness_name, harness)
    model_payloads = _model_payloads(config, model)
    runtime_env = {
        **_platform_runtime_env(),
        **_gateway_credential_env(model_payloads),
    }
    _validate_untranslated_shared_fields(config)

    fabric_config = fabric.FabricConfig(
        metadata=fabric.MetadataConfig(name=config.name, description=config.description or None),
        harness=fabric.HarnessConfig(
            adapter_id=_adapter_id_for_harness(harness),
            resolution="preinstalled",
            settings=harness.settings,
        ),
        models={key: fabric.ModelConfig(**payload) for key, payload in model_payloads.items()},
        instructions=_instructions_config(config),
        runtime=fabric.RuntimeConfig(**config.runtime.model_dump(exclude_none=True)),
        environment=_environment_config(config, runtime_env),
        skills=_skills_config(config),
        mcp=_mcp_config(config),
        tools=_tools_config(config),
    )

    _apply_telemetry(fabric_config, config, model)
    return fabric_config


def _platform_runtime_env() -> dict[str, str]:
    """Forward the Platform location needed by SDK-backed child tools."""
    return {name: value for name in PLATFORM_RUNTIME_ENV_VARS if (value := os.environ.get(name))}


def _environment_config(config: AgentConfig, runtime_env: dict[str, str]) -> Any:
    """Build FabricConfig.environment, merging spec env + mirror fields.

    ``runtime_env`` carries platform-injected values (base URLs, gateway
    credential). The EnvironmentSpec's plaintext ``env`` (merged onto
    ``config.environment.env`` at deploy time) is layered underneath so
    platform-injected values win on key collision.
    """
    environment = config.environment
    env = {**environment.env, **runtime_env}
    kwargs: dict[str, Any] = {
        "provider": environment.provider,
        "workspace": environment.workspace,
        "artifacts": environment.artifacts,
        "env": env,
        "settings": environment.settings,
    }
    # Forward optional Fabric mirror fields only when set, so defaults stay with
    # Fabric rather than being pinned by the platform config.
    if environment.control_location is not None:
        kwargs["control_location"] = environment.control_location
    if environment.ownership is not None:
        kwargs["ownership"] = environment.ownership
    if environment.connection:
        kwargs["connection"] = environment.connection
    if environment.metadata:
        kwargs["metadata"] = environment.metadata
    return fabric.EnvironmentConfig(**kwargs)


def _select_harness(config: AgentConfig, harness_name: str | None) -> tuple[str, HarnessConfig]:
    selected_harness_name = harness_name or config.default_harness
    harness = config.harnesses.get(selected_harness_name)
    if harness is None:
        available = ", ".join(sorted(config.harnesses))
        raise FabricTranslationError(
            f"Unknown configured harness {selected_harness_name!r}. Configured harnesses: {available}"
        )
    return selected_harness_name, harness


def _adapter_id_for_harness(harness: HarnessConfig) -> str:
    if harness.kind.startswith(FABRIC_ADAPTER_ID_PREFIXES):
        return harness.kind
    adapter_id = HARNESS_ADAPTER_IDS.get(harness.kind)
    if adapter_id is None:
        available = ", ".join(sorted(HARNESS_ADAPTER_IDS))
        prefixes = ", ".join(repr(prefix) for prefix in FABRIC_ADAPTER_ID_PREFIXES)
        raise FabricTranslationError(
            f"Unsupported harness kind {harness.kind!r}. Supported harness kinds: {available}; "
            f"or a fully-qualified Fabric adapter id starting with one of: {prefixes}."
        )
    return adapter_id


def _resolve_model(config: AgentConfig, harness_name: str, harness: HarnessConfig) -> ModelConfig:
    if harness.model is not None:
        return harness.model

    model = config.models.get("default")
    if model is None:
        raise FabricTranslationError(
            f"Harness {harness_name!r} does not define a model and no models.default is configured."
        )
    return model


def _model_payloads(config: AgentConfig, selected: ModelConfig) -> dict[str, dict[str, Any]]:
    """Translate every configured model, keyed as the Platform config keys them.

    ``FabricConfig.models`` and the adapter contract's ``AgentConfig.models`` are
    both keyed maps, so the whole map is forwarded rather than just the harness's
    primary model. An adapter that reads a secondary key (the Insights analyst
    reads ``fast`` for context summarization) would otherwise see it silently
    missing and fall back to ``default``.

    ``selected`` is always published as ``default`` — that is the key adapters
    treat as the primary model, and ``harness.model`` outranks ``models.default``
    when both are set.
    """
    payloads = {
        key: bind_platform_gateway_model_credential(_model_payload(model)) for key, model in config.models.items()
    }
    payloads["default"] = bind_platform_gateway_model_credential(_model_payload(selected))
    return payloads


def _gateway_credential_env(model_payloads: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Union the runtime-only IGW credential env over every forwarded model.

    Each Platform-gateway-routed model needs its ``api_key_env`` present in the
    child process, not just the primary one.
    """
    env: dict[str, str] = {}
    for payload in model_payloads.values():
        env.update(platform_gateway_credential_env({"models": {"default": payload}}))
    return env


def _model_payload(model: ModelConfig) -> dict[str, Any]:
    payload = model.model_dump(exclude_none=True)
    settings = payload.get("settings")
    if "base_url" not in payload and isinstance(settings, dict) and isinstance(settings.get("base_url"), str):
        payload["base_url"] = settings.pop("base_url")
    return payload


def _validate_untranslated_shared_fields(config: AgentConfig) -> None:
    if config.prompts:
        raise FabricTranslationError(
            "Top-level prompts are not translated yet. Configure prompt settings under the selected harness instead."
        )


def _instructions_config(config: AgentConfig) -> Any:
    if config.instructions is None or config.instructions.system is None:
        return None
    return fabric.InstructionsConfig(
        system=fabric.InstructionConfig(
            content=config.instructions.system.content,
            mode=config.instructions.system.mode,
        )
    )


def _skills_config(config: AgentConfig) -> Any:
    if config.skills is None:
        return None
    paths: list[str | Path] = []
    paths.extend(config.skills.paths)
    return fabric.SkillConfig(paths=paths)


def _mcp_config(config: AgentConfig) -> Any:
    if config.mcp is None:
        return None
    return fabric.McpConfig(
        servers={name: fabric.McpServerConfig(**server.model_dump()) for name, server in config.mcp.servers.items()}
    )


def _tools_config(config: AgentConfig) -> Any:
    if config.tools is None:
        return None
    return fabric.ToolsConfig(blocked=config.tools.blocked)


def _apply_telemetry(fabric_config: Any, config: AgentConfig, model: ModelConfig) -> None:
    telemetry = config.telemetry
    if not telemetry.enabled:
        return

    provider = telemetry.provider or "relay"
    if provider != "relay":
        raise FabricTranslationError(f"Unsupported telemetry provider {provider!r}. Only 'relay' is supported.")

    fabric_config.enable_relay(
        project=telemetry.project,
        output_dir=telemetry.output_dir,
        observability=_relay_observability_config(config, model),
    )


def _telemetry_agent_name(config: AgentConfig) -> str:
    """Return the name traces should be tagged with for this agent."""
    return config.telemetry.agent_name or config.name


def _relay_observability_config(config: AgentConfig, model: ModelConfig) -> dict[str, Any]:
    telemetry = config.telemetry
    agent_name = _telemetry_agent_name(config)
    observability: dict[str, Any] = {"version": 3}

    if telemetry.atif is not None:
        atif = dict(telemetry.atif)
        if telemetry.output_dir is not None:
            atif.setdefault("output_directory", telemetry.output_dir)
        atif.setdefault("agent_name", agent_name)
        atif.setdefault("model_name", model.model)
        observability["atif"] = atif

    if telemetry.atof is not None:
        observability["atof"] = _relay_atof_config(telemetry.atof, output_dir=telemetry.output_dir)

    if telemetry.opentelemetry is not None:
        observability["opentelemetry"] = _relay_opentelemetry_config(
            telemetry.opentelemetry,
            agent_name=agent_name,
        )

    return observability


def _relay_opentelemetry_config(opentelemetry_config: dict[str, Any], agent_name: str) -> dict[str, Any]:
    opentelemetry = dict(opentelemetry_config)
    endpoints = opentelemetry.get("endpoints")
    if not isinstance(endpoints, list):
        return opentelemetry

    enriched_endpoints: list[Any] = []
    for endpoint_config in endpoints:
        if isinstance(endpoint_config, dict):
            endpoint_config = dict(endpoint_config)
            endpoint_config.setdefault("service_name", agent_name)
        enriched_endpoints.append(endpoint_config)
    opentelemetry["endpoints"] = enriched_endpoints
    return opentelemetry


def _relay_atof_config(atof_config: dict[str, Any], output_dir: str | None) -> dict[str, Any]:
    atof = dict(atof_config)
    sinks = list(atof.pop("sinks", []) or [])

    file_sink = {"type": "file"}
    if output_dir is not None:
        file_sink["output_directory"] = output_dir
    for key in ("output_directory", "filename", "mode"):
        if key in atof:
            file_sink[key] = atof.pop(key)
    if len(file_sink) > 1:
        sinks.insert(0, file_sink)

    for endpoint in atof.pop("endpoints", []) or []:
        stream_sink = dict(endpoint)
        endpoint_url = stream_sink.pop("endpoint", None)
        if endpoint_url is not None and "url" not in stream_sink:
            stream_sink["url"] = endpoint_url
        stream_sink["type"] = "stream"
        sinks.append(stream_sink)

    translated = {"enabled": atof.pop("enabled", False)}
    if sinks:
        translated["sinks"] = sinks
    translated.update(atof)
    return translated
