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
    "remote-agent": "nvidia.fabric.remote-agent",
    "pi": "nvidia.fabric.pi",
    "nooa-bench-agent": "nvidia.fabric.nooa.bench-agent",
}

# A harness kind carrying this prefix is already a fully-qualified Fabric
# adapter id and is passed through verbatim. Plugin-owned adapters ship their
# own descriptor to <sys.prefix>/share/nemo-fabric/adapters/ and are resolved
# there by Fabric at runtime, so nemo-agents does not need a short-name entry
# for each one. A bad id therefore surfaces as a Fabric resolution error rather
# than a translation error here.
FABRIC_ADAPTER_ID_PREFIX = "nvidia.fabric."

PLATFORM_RUNTIME_ENV_VARS = ("NEMO_BASE_URL", "NHX_BASE_URL", "NHX_WORKSPACE")

# In-image path where the Fabric Dockerfile stages the NeMo Relay Pi extension
# (see container/template.py PI_RELAY_EXTENSION_PATH). The Pi adapter hard-requires
# harness.settings["relay_extension_path"] when Relay telemetry is enabled, so a
# Relay-enabled Pi agent defaults to the staged copy unless the config set a path.
PI_RELAY_EXTENSION_PATH = "/opt/pi-relay-extension"


class FabricTranslationError(ValueError):
    """Raised when Platform agent config cannot be translated to Fabric config."""


def translate_agent_config(config: AgentConfig, harness_name: str | None = None) -> fabric.FabricConfig:
    """Translate Platform-owned agent config into a typed in-memory FabricConfig."""
    harness_config = None
    workflow_config = None
    runtime_payload = config.runtime.model_dump(exclude_none=True)
    if config.workflow is not None:
        if harness_name is not None:
            raise FabricTranslationError("A harness override cannot be used with a workflow config.")
        workflow_config = fabric.WorkflowConfig(**config.workflow.model_dump())
        model = config.models.get("default")
        if model is None:
            raise FabricTranslationError("Workflow requires models.default to be configured.")
    else:
        selected_harness_name, harness = _select_harness(config, harness_name)
        model = _resolve_model(config, selected_harness_name, harness)
        _default_relay_runtime_artifacts(runtime_payload, config, harness)
        harness_config = fabric.HarnessConfig(
            adapter_id=_adapter_id_for_harness(harness),
            resolution="preinstalled",
            settings=harness.settings,
        )
    model_payloads = _model_payloads(config, model)
    runtime_env = {
        **_platform_runtime_env(),
        **_gateway_credential_env(model_payloads),
    }
    _validate_untranslated_shared_fields(config)

    fabric_config = fabric.FabricConfig(
        metadata=fabric.MetadataConfig(name=config.name, description=config.description or None),
        harness=harness_config,
        workflow=workflow_config,
        models={key: fabric.ModelConfig(**payload) for key, payload in model_payloads.items()},
        instructions=_instructions_config(config),
        runtime=fabric.RuntimeConfig(**runtime_payload),
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
    if selected_harness_name is None:
        raise FabricTranslationError("No default_harness is configured.")
    harness = config.harnesses.get(selected_harness_name)
    if harness is None:
        available = ", ".join(sorted(config.harnesses))
        raise FabricTranslationError(
            f"Unknown configured harness {selected_harness_name!r}. Configured harnesses: {available}"
        )
    return selected_harness_name, harness


def _adapter_id_for_harness(harness: HarnessConfig) -> str:
    if harness.kind.startswith(FABRIC_ADAPTER_ID_PREFIX):
        return harness.kind
    adapter_id = HARNESS_ADAPTER_IDS.get(harness.kind)
    if adapter_id is None:
        available = ", ".join(sorted(HARNESS_ADAPTER_IDS))
        raise FabricTranslationError(
            f"Unsupported harness kind {harness.kind!r}. Supported harness kinds: {available}; "
            f"or a fully-qualified Fabric adapter id starting with {FABRIC_ADAPTER_ID_PREFIX!r}."
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

    _default_relay_extension_path(fabric_config)

    fabric_config.enable_relay(
        project=telemetry.project,
        output_dir=telemetry.output_dir,
        observability=_relay_observability_config(config, model),
    )


def _default_relay_runtime_artifacts(
    runtime_payload: dict[str, Any], config: AgentConfig, harness: HarnessConfig
) -> None:
    """Default ``runtime.artifacts`` for a Relay-enabled Pi harness when unset.

    Fabric creates the runtime-owned Relay config under ``runtime.artifacts``, so a
    Relay-enabled Pi agent needs it set or Pi startup fails. Default it to
    ``./artifacts/pi`` while preserving an explicitly configured value; a no-op for
    other harnesses and for telemetry-disabled Pi agents.
    """
    telemetry = config.telemetry
    if not telemetry.enabled or (telemetry.provider or "relay") != "relay":
        return
    if _adapter_id_for_harness(harness) != HARNESS_ADAPTER_IDS["pi"]:
        return
    runtime_payload.setdefault("artifacts", "./artifacts/pi")


def _default_relay_extension_path(fabric_config: Any) -> None:
    """Point a Pi harness at the image-staged Relay extension when none is set.

    The Pi adapter raises ``pi_relay_extension_not_found`` if Relay telemetry is
    active without ``harness.settings["relay_extension_path"]``. The Fabric image
    stages the extension at ``PI_RELAY_EXTENSION_PATH``, so default to it while
    preserving an explicitly configured path. Only the Pi adapter consumes this
    setting, so defaulting it is a no-op for the other harnesses.
    """
    harness = fabric_config.harness
    if harness is None or harness.adapter_id != HARNESS_ADAPTER_IDS["pi"]:
        return
    harness.settings.setdefault("relay_extension_path", PI_RELAY_EXTENSION_PATH)


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
