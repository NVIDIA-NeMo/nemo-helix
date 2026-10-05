# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a registered platform agent into what a runner target needs, at submission.

``resolve_registered_agent`` is the one entry point: it looks the agent up, resolves its config as a
deployment would, snapshots the files it refers to, and writes the result into the Fabric or Harbor
target. The helpers around it (gateway placeholders, stdio MCP secret templates, secret-ref merging)
are the details of that translation. The job module calls the entry point and nothing else here, apart
from ``expand_mcp_secret_env``, which the host Fabric path applies at run time.
"""

from __future__ import annotations

import copy
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.agent_config_formats import AgentConfigFormatError, resolve_agent_config_for_deployment
from nemo_agents_plugin.entities import ethos_fileset_name
from nemo_agents_plugin.environment_resolution import (
    EnvironmentResolutionError,
    merge_environment_spec_into_agent_config,
)
from nemo_agents_plugin.fabric.gateway_credentials import platform_gateway_credential_env
from nemo_agents_plugin.fabric.translator import FabricTranslationError, translate_agent_config
from nemo_evaluator.api.schemas import AgentRef
from nemo_evaluator.filesets import FilesetRef
from nemo_evaluator.jobs.agent_files_snapshot import discard_registered_agent_files, snapshot_registered_agent_files
from nemo_evaluator.jobs.agent_spec import (
    FabricRunnerTarget,
    HarborRunnerTarget,
    RegisteredAgentSource,
    Target,
    registered_agent_config,
    registered_agent_config_needs_files,
    registered_agent_source,
)
from nemo_evaluator.jobs.fabric_harness_packages import fabric_harness_package
from nemo_evaluator_sdk.values import SecretRef
from nemo_helix_plugin.agents.client import AsyncAgentsClient
from nemo_helix_plugin.agents.types import EnvironmentSpecInline
from nemo_helix_plugin.client.adapter import AsyncHelixClient, client_from_platform
from nemo_helix_plugin.client.errors import NotFoundError, PermissionDeniedError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.refs import parse_entity_ref

logger = logging.getLogger(__name__)


#: The only registered-agent config format a runner exists for; ``nat-workflow-v1`` describes a NAT workflow.
FABRIC_AGENT_CONFIG_FORMAT = "nemo-agents-spec-v1"


def _without_gateway_placeholders(config: dict[str, Any]) -> dict[str, Any]:
    """The translated config minus the gateway placeholder credentials the Harbor agent adds itself.

    Harbor persists ``agent_kwargs`` unredacted and refuses a credential-named value in them, placeholder
    included; ``NemoFabricAgent`` restores the placeholder per trial from the model's ``api_key_env``.
    """
    env = dict(config.get("environment", {}).get("env", {}))
    for payload in config.get("models", {}).values():
        for name, placeholder in platform_gateway_credential_env({"models": {"default": payload}}).items():
            if env.get(name) == placeholder:
                del env[name]
    if env == config.get("environment", {}).get("env", {}):
        return config
    return {**config, "environment": {**config["environment"], "env": env}}


@dataclass(frozen=True)
class _ResolvedRegisteredAgent:
    """A registered agent turned into what a runner needs: where it lives, its Fabric config, its secret refs."""

    ref: AgentRef
    config: dict[str, Any]
    env_secrets: dict[str, SecretRef]
    files: FilesetRef | None


async def _load_registered_agent(
    agent_ref: str,
    environment: EnvironmentSpecInline | None,
    *,
    workspace: str,
    async_sdk: AsyncHelixClient | None,
) -> _ResolvedRegisteredAgent:
    """Look a registered agent up and resolve it as ``nemo agents deploy`` would.

    Inference Gateway binding first, then the environment spec merged on top, then translation of the
    platform ``agent.yaml`` into a Fabric config. A config that refers to files by relative path must have
    the agent's Ethos FileSet to stage them from; that is checked here, once, rather than in every job.
    """
    parsed = parse_entity_ref(agent_ref, workspace)
    agent_workspace, agent_name = parsed.workspace, parsed.name

    agents = client_from_platform(async_sdk, AsyncAgentsClient)
    try:
        agent = (await agents.get_agent(workspace=agent_workspace, name=agent_name)).data()
    except NotFoundError as exc:
        raise ValueError(f"registered agent {agent_workspace}/{agent_name} does not exist") from exc
    except PermissionDeniedError as exc:
        raise PermissionError(f"access denied to registered agent {agent_workspace}/{agent_name}") from exc

    if agent.config_format != FABRIC_AGENT_CONFIG_FORMAT:
        raise ValueError(
            f"registered agent {agent_workspace}/{agent_name} has config_format {agent.config_format!r}; only "
            f"{FABRIC_AGENT_CONFIG_FORMAT!r} agents can be evaluated as a runner"
        )

    try:
        resolved = resolve_agent_config_for_deployment(
            agent.config_format,
            dict(agent.config),
            workspace=agent_workspace,
            agent_name=agent_name,
        )
        merged = merge_environment_spec_into_agent_config(resolved, environment)
        fabric_config = translate_agent_config(AgentConfig.model_validate(merged.config))
    except (EnvironmentResolutionError, AgentConfigFormatError, FabricTranslationError) as exc:
        raise ValueError(
            f"registered agent {agent_workspace}/{agent_name} cannot be run through Fabric: {exc}"
        ) from exc

    config = _template_mcp_secret_env(fabric_config.model_dump(mode="json", exclude_none=True), environment)
    files: FilesetRef | None = None
    if registered_agent_config_needs_files(config):
        files_client = client_from_platform(async_sdk, AsyncFilesClient)
        try:
            await files_client.get_fileset(workspace=agent_workspace, name=ethos_fileset_name(agent_name))
        except NotFoundError:
            raise ValueError(
                f"registered agent {agent_workspace}/{agent_name} refers to files by relative path but has no "
                f"Ethos FileSet {agent_workspace}/{ethos_fileset_name(agent_name)} to stage them from"
            ) from None
        # The Ethos FileSet is rewritten on every re-registration; the job stages a copy taken now.
        files = await snapshot_registered_agent_files(files_client, workspace=agent_workspace, agent_name=agent_name)
    return _ResolvedRegisteredAgent(
        ref=AgentRef(root=f"{agent_workspace}/{agent_name}"),
        config=config,
        env_secrets={env_name: SecretRef(root=ref) for env_name, ref in merged.secrets.items()},
        files=files,
    )


def _merge_env_secrets(
    target_secrets: Mapping[str, SecretRef], agent_secrets: Mapping[str, SecretRef], *, agent: str
) -> dict[str, SecretRef]:
    """The agent's secret refs under the submitter's: a name both bind keeps the target's ref, with a warning.

    The warning carries a count, not the names: anything read out of a secrets mapping, keys included,
    reads as secret material to a scanner and to anyone grepping logs, and the submitter can diff the two
    specs themselves.
    """
    overridden = sum(
        1 for env_name, agent_ref in agent_secrets.items() if target_secrets.get(env_name) not in (None, agent_ref)
    )
    if overridden:
        logger.warning(
            "env_secrets on the target override registered agent %s's binding of %d environment variable(s)",
            agent,
            overridden,
        )
    return {**agent_secrets, **target_secrets}


_ENV_TEMPLATE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def _template_mcp_secret_env(config: dict[str, Any], environment: EnvironmentSpecInline | None) -> dict[str, Any]:
    """Name each stdio MCP server's bound secrets in its ``env`` as ``${NAME}`` templates.

    A stdio MCP server does not inherit the harness process's environment, only its own ``env``; the
    runtime holding the value expands the template at launch, so nothing persisted carries a value.
    """
    if environment is None or not environment.mcp:
        return config
    servers = ((config.get("mcp") or {}).get("servers")) or {}
    for name, fulfillment in environment.mcp.items():
        server = servers.get(name)
        if not isinstance(server, dict) or server.get("transport") != "stdio" or not fulfillment.secrets:
            continue
        server["env"] = {
            **(server.get("env") or {}),
            **{env_name: f"${{{env_name}}}" for env_name in fulfillment.secrets},
        }
    return config


def _without_mcp_env_templates(config: dict[str, Any]) -> dict[str, Any]:
    """The config minus stdio MCP server ``${NAME}`` env templates, for a runtime that cannot expand them."""
    servers = ((config.get("mcp") or {}).get("servers")) or {}
    templated = {
        name: server
        for name, server in servers.items()
        if isinstance(server, dict)
        and any(isinstance(v, str) and _ENV_TEMPLATE.match(v) for v in (server.get("env") or {}).values())
    }
    if not templated:
        return config
    stripped = copy.deepcopy(config)
    for name in templated:
        env = stripped["mcp"]["servers"][name]["env"]
        stripped["mcp"]["servers"][name]["env"] = {k: v for k, v in env.items() if not _ENV_TEMPLATE.match(v)}
    return stripped


def expand_mcp_secret_env(config: dict[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
    """A copy of ``config`` with stdio MCP server ``env`` templates filled from ``values``.

    Only the in-memory config handed to the host runtime is expanded; the persisted spec keeps the
    templates. Templates naming a variable ``values`` lacks are left as they are.
    """
    servers = ((config.get("mcp") or {}).get("servers")) or {}
    if not any(isinstance(s, dict) and s.get("transport") == "stdio" and s.get("env") for s in servers.values()):
        return config
    expanded = copy.deepcopy(config)
    for server in expanded["mcp"]["servers"].values():
        if not isinstance(server, dict) or server.get("transport") != "stdio":
            continue
        for key, value in list((server.get("env") or {}).items()):
            match = _ENV_TEMPLATE.match(value) if isinstance(value, str) else None
            if match and match.group(1) in values:
                server["env"][key] = values[match.group(1)]
    return expanded


async def resolve_registered_agent(
    target: Target | None,
    *,
    workspace: str,
    async_sdk: AsyncHelixClient | None,
) -> Target | None:
    """Fill a Fabric or Harbor runner target that names a registered agent with what that agent is.

    The source keeps its ``agent``, workspace-qualified. A Fabric target gets the config it resolved to as
    ``resolved_config``; a Harbor target gets it in its kwargs, for the installed Fabric agent its source
    implies. Either way the job runs the agent fresh per trial without ever looking it up itself.
    """
    if not isinstance(target, (FabricRunnerTarget, HarborRunnerTarget)):
        return target
    source = registered_agent_source(target)
    if source is None or registered_agent_config(target) is not None:
        return target

    agent = await _load_registered_agent(
        source.agent.root, source.environment, workspace=workspace, async_sdk=async_sdk
    )
    try:
        return _fill_target(target, source, agent)
    except Exception:
        # The load may have snapshotted the agent's files; nothing downstream owns that FileSet until a
        # resolved target exists, so a failure here deletes it rather than leaving one per attempt.
        if agent.files is not None:
            await discard_registered_agent_files(client_from_platform(async_sdk, AsyncFilesClient), agent.files)
        raise


def _fill_target(
    target: FabricRunnerTarget | HarborRunnerTarget, source: RegisteredAgentSource, agent: _ResolvedRegisteredAgent
) -> FabricRunnerTarget | HarborRunnerTarget:
    """The target with the loaded agent written into it, per runner kind. Pure: no I/O, so it can fail freely."""
    resolved_source = source.model_copy(update={"agent": agent.ref, "files": agent.files})
    env_secrets = _merge_env_secrets(target.env_secrets, agent.env_secrets, agent=agent.ref.root)
    if isinstance(target, FabricRunnerTarget):
        return target.model_copy(
            update={"source": resolved_source, "resolved_config": agent.config, "env_secrets": env_secrets}
        )

    adapter_id = agent.config["harness"]["adapter_id"]
    fabric_package = target.agent_kwargs.get("fabric_package")
    if fabric_package is not None and not isinstance(fabric_package, str):
        raise ValueError(
            f"`agent_kwargs.fabric_package` must be a requirement string or null, not {type(fabric_package).__name__}"
        )
    agent_kwargs: dict[str, Any] = {
        **target.agent_kwargs,
        # The runner inside the task container expands no templates; a literal "${NAME}" is worse than
        # no variable, so a Harbor agent's stdio servers go without the secret (see the Harbor docs).
        "fabric_config": _without_mcp_env_templates(_without_gateway_placeholders(agent.config)),
        "fabric_package": fabric_package if fabric_package is not None else fabric_harness_package(adapter_id),
    }
    return target.model_copy(
        update={"source": resolved_source, "agent_kwargs": agent_kwargs, "env_secrets": env_secrets}
    )
