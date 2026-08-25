# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve an AgentEnvironment and merge its EnvironmentSpec into agent config.

An AgentDeployment references an :class:`AgentEnvironment` (by name or inline).
At create time the deployment route:

1. resolves the environment - dereferencing any ``"workspace/name"`` refs for
   the environment and its two specs into concrete inline specs;
2. merges the resolved EnvironmentSpec into the Platform-owned agent config
   (``nemo-agents-spec-v1``) so the Fabric translator sees a single config, and
   collects the spec's secret-env references (top-level + per-MCP) for the
   container backend to inject as secret-backed env vars; and
3. snapshots the resolved ComputeSpec and the collected secrets onto the
   deployment for the container backend.

Secret references are never written into the config as plaintext. Both the
spec's top-level ``secrets`` and each declared MCP server's ``secrets`` are
returned as an ENV_VAR_NAME -> ref map; the substrate injects the resolved value
into the process env under that name, and MCP servers read their credentials
from the process env by name (env-var-name indirection) rather than the config
embedding a raw reference string.

Merge precedence is **EnvironmentSpec wins over the Agent config**: the Agent
config supplies defaults, and the EnvironmentSpec overrides them where it sets a
value. Fields the EnvironmentSpec leaves unset fall back to the Agent's
defaults, so an agent authored without an environment behaves identically.

This ordering is the middle tier of an intended precedence chain
``deployment overrides > EnvironmentSpec > Agent defaults``. A future
deployment-specific override layer (e.g. an inline ``environment_overrides`` on
the deployment) slots in by applying this same merge a second time with the
higher-priority spec last, so each successive layer overrides the previous.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from nemo_agents_plugin.entities import (
    NEMO_AGENTS_SPEC_CONFIG_FORMAT,
    AgentComputeSpec,
    AgentEnvironment,
    AgentEnvironmentInline,
    AgentEnvironmentSpec,
    AgentSandboxSpec,
    ComputeSpecInline,
    EnvironmentSpecInline,
    SandboxSpecInline,
)
from nemo_platform_plugin.entities.base import parse_qualified_name
from nemo_platform_plugin.entity_client import NemoEntitiesClient, NemoEntityNotFoundError


class EnvironmentResolutionError(ValueError):
    """Raised when an AgentEnvironment or one of its specs cannot be resolved."""


@dataclass(frozen=True)
class ResolvedEnvironment:
    """Concrete environment/sandbox/compute specs resolved from an AgentEnvironment."""

    environment_spec: EnvironmentSpecInline | None = None
    sandbox_spec: SandboxSpecInline | None = None
    compute_spec: ComputeSpecInline | None = None


@dataclass(frozen=True)
class MergedEnvironment:
    """Result of merging an EnvironmentSpec into an agent config.

    ``config`` is the merged ``nemo-agents-spec-v1`` config. ``secrets`` maps
    ENV_VAR_NAME -> Secrets-service ref, collected from both the EnvironmentSpec's
    top-level ``secrets`` and every declared MCP server's ``secrets``. The caller
    snapshots ``secrets`` onto the deployment so the substrate injects the
    resolved value into the process env (never plaintext); MCP servers read their
    credentials from that process env by name (env-var-name indirection), rather
    than the config embedding a raw secret reference as a literal value.
    """

    config: dict[str, Any]
    secrets: dict[str, str]


async def resolve_environment(
    environment: str | AgentEnvironmentInline | None,
    *,
    workspace: str,
    entity_client: NemoEntitiesClient,
) -> ResolvedEnvironment:
    """Resolve an AgentEnvironment (ref | inline | None) into concrete specs.

    Dereferences the environment and its ``environment_spec`` / ``compute_spec``
    refs. Missing refs raise :class:`EnvironmentResolutionError`. ``None`` yields
    an empty :class:`ResolvedEnvironment` (no environment configured).
    """
    if environment is None:
        return ResolvedEnvironment()

    resolved_env = await _resolve_agent_environment(environment, workspace=workspace, entity_client=entity_client)
    environment_spec = await _resolve_environment_spec(
        resolved_env.environment_spec, workspace=workspace, entity_client=entity_client
    )
    sandbox_spec = await _resolve_sandbox_spec(
        resolved_env.sandbox_spec, workspace=workspace, entity_client=entity_client
    )
    compute_spec = await _resolve_compute_spec(
        resolved_env.compute_spec, workspace=workspace, entity_client=entity_client
    )
    return ResolvedEnvironment(
        environment_spec=environment_spec,
        sandbox_spec=sandbox_spec,
        compute_spec=compute_spec,
    )


async def _resolve_agent_environment(
    environment: str | AgentEnvironmentInline,
    *,
    workspace: str,
    entity_client: NemoEntitiesClient,
) -> AgentEnvironmentInline:
    if isinstance(environment, str):
        ref_workspace, name = parse_qualified_name(environment, default_workspace=workspace)
        try:
            return await entity_client.get(AgentEnvironment, name=name, workspace=ref_workspace)
        except NemoEntityNotFoundError as exc:
            raise EnvironmentResolutionError(
                f"AgentEnvironment '{name}' not found in workspace '{ref_workspace}'."
            ) from exc
    return environment


async def _resolve_environment_spec(
    spec: str | EnvironmentSpecInline | None,
    *,
    workspace: str,
    entity_client: NemoEntitiesClient,
) -> EnvironmentSpecInline | None:
    if spec is None:
        return None
    if isinstance(spec, str):
        ref_workspace, name = parse_qualified_name(spec, default_workspace=workspace)
        try:
            return await entity_client.get(AgentEnvironmentSpec, name=name, workspace=ref_workspace)
        except NemoEntityNotFoundError as exc:
            raise EnvironmentResolutionError(
                f"AgentEnvironmentSpec '{name}' not found in workspace '{ref_workspace}'."
            ) from exc
    return spec


async def _resolve_compute_spec(
    spec: str | ComputeSpecInline | None,
    *,
    workspace: str,
    entity_client: NemoEntitiesClient,
) -> ComputeSpecInline | None:
    if spec is None:
        return None
    if isinstance(spec, str):
        ref_workspace, name = parse_qualified_name(spec, default_workspace=workspace)
        try:
            return await entity_client.get(AgentComputeSpec, name=name, workspace=ref_workspace)
        except NemoEntityNotFoundError as exc:
            raise EnvironmentResolutionError(
                f"AgentComputeSpec '{name}' not found in workspace '{ref_workspace}'."
            ) from exc
    return spec


async def _resolve_sandbox_spec(
    spec: str | SandboxSpecInline | None,
    *,
    workspace: str,
    entity_client: NemoEntitiesClient,
) -> SandboxSpecInline | None:
    if spec is None:
        return None
    if isinstance(spec, str):
        ref_workspace, name = parse_qualified_name(spec, default_workspace=workspace)
        try:
            return await entity_client.get(AgentSandboxSpec, name=name, workspace=ref_workspace)
        except NemoEntityNotFoundError as exc:
            raise EnvironmentResolutionError(
                f"AgentSandboxSpec '{name}' not found in workspace '{ref_workspace}'."
            ) from exc
    return spec


def merge_environment_spec_into_agent_config(
    config: dict[str, Any],
    env_spec: EnvironmentSpecInline | None,
) -> MergedEnvironment:
    """Merge a resolved EnvironmentSpec into a ``nemo-agents-spec-v1`` config.

    Returns a :class:`MergedEnvironment` holding a deep-copied merged config and
    the collected secret-env references (top-level + per-MCP). Merge precedence is
    EnvironmentSpec-wins: the Agent config supplies defaults and the spec overrides
    them where it sets a value; fields the spec leaves unset keep the Agent default.

    Only ``nemo-agents-spec-v1`` (Fabric) configs are merged; other formats are
    returned unchanged (they have no environment concept to fulfill).
    """
    if env_spec is None:
        return MergedEnvironment(config=config, secrets={})
    if config.get("config_format") != NEMO_AGENTS_SPEC_CONFIG_FORMAT:
        return MergedEnvironment(config=config, secrets=dict(env_spec.secrets))

    merged = copy.deepcopy(config)
    _merge_environment_block(merged, env_spec)
    _merge_process_env(merged, env_spec)
    _merge_model_provider_override(merged, env_spec)
    secrets = _collect_secrets(merged, env_spec)
    return MergedEnvironment(config=merged, secrets=secrets)


def _merge_environment_block(config: dict[str, Any], env_spec: EnvironmentSpecInline) -> None:
    """Override FabricConfig.environment mirror fields with the EnvironmentSpec's values."""
    environment = config.setdefault("environment", {})
    if not isinstance(environment, dict):
        return

    # Scalar mirror fields: the EnvironmentSpec overrides the Agent's value only
    # when it EXPLICITLY set the field. ``model_fields_set`` distinguishes an
    # explicit value from a schema default - important for ``provider``, whose
    # default is "local" (not None); without this guard a spec that never set
    # ``provider`` would clobber an Agent's "docker"/"k8s" with "local". The
    # spec's ``workspace_path``/``artifacts_path`` map onto the config's
    # ``workspace``/``artifacts`` (the harness paths); the entity/tenant
    # ``workspace`` is unrelated and never merged here.
    scalar_fields = {
        "provider": "provider",
        "workspace_path": "workspace",
        "artifacts_path": "artifacts",
        "control_location": "control_location",
        "ownership": "ownership",
    }
    for spec_attr, config_key in scalar_fields.items():
        if spec_attr not in env_spec.model_fields_set:
            continue
        value = getattr(env_spec, spec_attr)
        if value is not None:
            environment[config_key] = value

    # Dict mirror fields: Agent supplies the base, EnvironmentSpec keys win on collision.
    for field in ("connection", "metadata", "settings"):
        spec_value = getattr(env_spec, field)
        if spec_value:
            environment[field] = {**environment.get(field, {}), **spec_value}


def _merge_process_env(config: dict[str, Any], env_spec: EnvironmentSpecInline) -> None:
    """Merge plaintext env vars into environment.env (EnvironmentSpec keys win)."""
    if not env_spec.env:
        return
    environment = config.setdefault("environment", {})
    if not isinstance(environment, dict):
        return
    existing = environment.get("env")
    existing = existing if isinstance(existing, dict) else {}
    environment["env"] = {**existing, **env_spec.env}


def _merge_model_provider_override(config: dict[str, Any], env_spec: EnvironmentSpecInline) -> None:
    """Apply an external model-provider override to models.default (overrides the Agent)."""
    override = env_spec.model_provider_override
    if override is None:
        return
    models = config.setdefault("models", {})
    if not isinstance(models, dict):
        return
    model = models.setdefault("default", {})
    if not isinstance(model, dict):
        return
    model["base_url"] = override.base_url
    if override.provider is not None:
        model["provider"] = override.provider
    if override.api_key is not None:
        model["api_key_env"] = override.api_key


def _collect_secrets(config: dict[str, Any], env_spec: EnvironmentSpecInline) -> dict[str, str]:
    """Collect secret-env references and fulfill Agent-declared MCP servers.

    Returns ENV_VAR_NAME -> Secrets-service ref for both the spec's top-level
    ``secrets`` and every declared MCP server's ``secrets``. Secret refs are never
    written into the config as plaintext: the caller injects the resolved value
    into the process env under ENV_VAR_NAME, and MCP servers read that value by
    name (env-var-name indirection). Non-secret MCP ``env`` is still merged into
    the server config; the fulfillment overrides the Agent's url/env on collision.

    Raises :class:`EnvironmentResolutionError` when the same env var name maps to
    two different secret references (an ambiguous, unresolvable collision), or
    when a name is bound both as a secret and as a plaintext value (in the merged
    ``environment.env`` or a declared MCP server's ``env``) - the effective
    credential would otherwise depend on runtime env construction order.
    """
    secrets: dict[str, str] = {}

    def _record(env_name: str, ref: str) -> None:
        existing = secrets.get(env_name)
        if existing is not None and existing != ref:
            raise EnvironmentResolutionError(
                f"Secret env var {env_name!r} is bound to conflicting references "
                f"{existing!r} and {ref!r} in the environment spec."
            )
        secrets[env_name] = ref

    for env_name, ref in env_spec.secrets.items():
        _record(env_name, ref)

    if env_spec.mcp:
        mcp = config.setdefault("mcp", {})
        servers = mcp.setdefault("servers", {}) if isinstance(mcp, dict) else None
        if isinstance(servers, dict):
            for name, fulfillment in env_spec.mcp.items():
                server = servers.get(name)
                # Only fulfill servers the Agent declared; skip undeclared names.
                if not isinstance(server, dict):
                    continue
                # url: the fulfillment's endpoint overrides the Agent's.
                server["url"] = fulfillment.url
                # Non-secret env merges into the server config (fulfillment wins).
                if fulfillment.env:
                    existing_env = server.get("env")
                    existing_env = existing_env if isinstance(existing_env, dict) else {}
                    server["env"] = {**existing_env, **fulfillment.env}
                # Secret refs are collected for injection into the process env,
                # NOT stored in the server config as a literal reference string.
                for env_name, ref in fulfillment.secrets.items():
                    _record(env_name, ref)

    _reject_plaintext_secret_collisions(config, secrets)
    return secrets


def _reject_plaintext_secret_collisions(config: dict[str, Any], secrets: dict[str, str]) -> None:
    """Reject any secret-bound env name that is also bound as a plaintext value.

    A name present both in ``secrets`` (injected as a secret-backed env var) and
    as a plaintext value in the merged ``environment.env`` or a declared MCP
    server's ``env`` would double-bind: the effective value would depend on
    runtime env construction order. Reject it up front.
    """
    if not secrets:
        return

    def _check(env_map: Any, where: str) -> None:
        if not isinstance(env_map, dict):
            return
        clash = sorted(name for name in secrets if name in env_map)
        if clash:
            raise EnvironmentResolutionError(
                f"Env var(s) {', '.join(clash)} are bound both as a secret and as a "
                f"plaintext value ({where}). Use one or the other."
            )

    environment = config.get("environment")
    if isinstance(environment, dict):
        _check(environment.get("env"), "environment.env")

    mcp = config.get("mcp")
    servers = mcp.get("servers") if isinstance(mcp, dict) else None
    if isinstance(servers, dict):
        for name, server in servers.items():
            if isinstance(server, dict):
                _check(server.get("env"), f"mcp.servers.{name}.env")
