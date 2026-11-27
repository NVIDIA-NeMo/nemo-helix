# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a registered platform agent into what a runner target needs, at submission.

``resolve_registered_agent`` is the one entry point: it looks the agent up, resolves its config as a
deployment would, snapshots the files it refers to, and writes the result into the Fabric or Harbor
target. Asked to, it only names the snapshot, and ``take_registered_agent_files_snapshot`` takes it once
the submission's checks have passed, so a dry run copies nothing. The helpers around it (gateway
placeholders, stdio MCP secret templates, secret-ref merging) are the details of that translation. The
job module calls those two and nothing else here, apart from ``expand_mcp_secret_env``, which the host
Fabric path applies at run time.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar
from urllib.parse import quote

from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.agent_config_formats import AgentConfigFormatError, resolve_agent_config_for_deployment
from nemo_agents_plugin.entities import ethos_fileset_name
from nemo_agents_plugin.environment_resolution import (
    EnvironmentResolutionError,
    merge_environment_spec_into_agent_config,
)
from nemo_agents_plugin.fabric.gateway_credentials import platform_gateway_credential_env
from nemo_agents_plugin.fabric.translator import (
    FabricTranslationError,
    translate_agent_config,
    upgrade_relay_observability_version,
)
from nemo_evals._fileset_receipt import FilesetReceipt
from nemo_evals.api.schemas import AgentRef
from nemo_evals.filesets import FilesetRef
from nemo_evals.jobs.agent_files_snapshot import (
    discard_registered_agent_files,
    snapshot_fileset_name,
    snapshot_registered_agent_files,
)
from nemo_evals.jobs.agent_spec import (
    FabricRunnerTarget,
    GymRunnerTarget,
    HarborRunnerTarget,
    RegisteredAgentSource,
    Target,
    registered_agent_config,
    registered_agent_config_needs_files,
    registered_agent_source,
)
from nemo_evals.jobs.fabric_harness_packages import fabric_harness_package
from nemo_evals.jobs.utils import require_platform_client
from nemo_evals.source_authorization import SourceReadAuthorizer
from nemo_helix_plugin.agents.client import AsyncAgentsClient
from nemo_helix_plugin.agents.types import EnvironmentSpecInline
from nemo_helix_plugin.client.adapter import AsyncHelixClient, client_from_platform
from nemo_helix_plugin.client.errors import NotFoundError, PermissionDeniedError
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.refs import parse_entity_ref
from nhx_evals_sdk.agent_eval.runtimes.harbor.mcp_env_runner import MCP_ENV_TEMPLATE, unbound_mcp_env_templates
from nhx_evals_sdk.values import SecretRef

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
    #: The files snapshot resolution took, if it took one.
    files: FilesetReceipt | None
    #: ``workspace/name`` of the snapshot still to take, when resolution named it and left the copy for later.
    pending_files: str | None = None


async def _authorize_files_snapshot(
    authorize: SourceReadAuthorizer | None, *, workspace: str, agent_name: str, snapshot: str
) -> None:
    """Check as the caller what taking the files snapshot does, before any of it.

    The snapshot is taken with the caller's client, so a caller who may not do this is refused when it's
    taken. Checking first refuses them before anything is written, and gives a dry run, which takes no
    snapshot, the same answer.
    """
    if authorize is None:
        return
    filesets = f"/apis/files/v2/workspaces/{quote(workspace, safe='')}/filesets"
    await authorize("GET", f"{filesets}/{quote(ethos_fileset_name(agent_name), safe='')}/files")
    await authorize("POST", filesets)
    # One representative file path: the platform authorizes file writes per workspace collection.
    await authorize("PUT", f"{filesets}/{quote(snapshot, safe='')}/-/file")


async def _load_registered_agent(
    agent_ref: str,
    environment: EnvironmentSpecInline | None,
    *,
    workspace: str,
    async_sdk: AsyncHelixClient | None,
    authorize: SourceReadAuthorizer | None,
    snapshot_files: bool,
) -> _ResolvedRegisteredAgent:
    """Look a registered agent up and resolve it as ``nemo agents deploy`` would.

    Inference Gateway binding first, then the environment spec merged on top, then translation of the
    platform ``agent.yaml`` into a Fabric config. A config that refers to files by relative path must have
    the agent's Ethos FileSet to stage them from; that is checked here, once, rather than in every job.
    Its files are snapshotted here too, unless ``snapshot_files`` is off: then the snapshot is only named.
    """
    parsed = parse_entity_ref(agent_ref, workspace)
    agent_workspace, agent_name = parsed.workspace, parsed.name

    platform = require_platform_client(async_sdk, "resolving a registered agent")
    agents = client_from_platform(platform, AsyncAgentsClient)
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
        merged.require_runtime_support()
        fabric_config = translate_agent_config(AgentConfig.model_validate(merged.config))
    except (EnvironmentResolutionError, AgentConfigFormatError, FabricTranslationError) as exc:
        raise ValueError(
            f"registered agent {agent_workspace}/{agent_name} cannot be run through Fabric: {exc}"
        ) from exc

    config = _template_mcp_secret_env(fabric_config.model_dump(mode="json", exclude_none=True), environment)
    files: FilesetReceipt | None = None
    pending_files: str | None = None
    if registered_agent_config_needs_files(config):
        snapshot = snapshot_fileset_name()
        await _authorize_files_snapshot(authorize, workspace=agent_workspace, agent_name=agent_name, snapshot=snapshot)
        files_client = client_from_platform(platform, AsyncFilesClient)
        try:
            await files_client.get_fileset(workspace=agent_workspace, name=ethos_fileset_name(agent_name))
        except NotFoundError:
            raise ValueError(
                f"registered agent {agent_workspace}/{agent_name} refers to files by relative path but has no "
                f"Ethos FileSet {agent_workspace}/{ethos_fileset_name(agent_name)} to stage them from"
            ) from None
        # The Ethos FileSet is rewritten on every re-registration; the job stages a copy taken at submit.
        if snapshot_files:
            files = await snapshot_registered_agent_files(
                files_client, workspace=agent_workspace, agent_name=agent_name, snapshot=snapshot
            )
        else:
            pending_files = f"{agent_workspace}/{snapshot}"
    return _ResolvedRegisteredAgent(
        ref=AgentRef(root=f"{agent_workspace}/{agent_name}"),
        config=config,
        env_secrets={env_name: SecretRef(root=ref) for env_name, ref in merged.secrets.items()},
        files=files,
        pending_files=pending_files,
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


def _require_mcp_env_templates_bound(config: dict[str, Any], bound: set[str], *, agent: str) -> None:
    """Refuse a stdio MCP server ``${NAME}`` env template that names no variable the job will set."""
    unbound = unbound_mcp_env_templates(config, bound)
    if unbound:
        name, variables = next(iter(unbound.items()))
        raise ValueError(
            f"registered agent {agent}'s stdio MCP server {name!r} reads {', '.join(variables)} from the "
            "environment, but the job does not set it; bind it in the target's `env_secrets` (or `env_vars` "
            f"for a non-secret value) or the source's `environment.mcp.{name}.secrets`"
        )


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
            match = MCP_ENV_TEMPLATE.match(value) if isinstance(value, str) else None
            if match and match.group(1) in values:
                server["env"][key] = values[match.group(1)]
    return expanded


async def resolve_registered_agent(
    target: Target | None,
    *,
    workspace: str,
    async_sdk: AsyncHelixClient | None,
    authorize: SourceReadAuthorizer | None = None,
    snapshot_files: bool = True,
) -> Target | None:
    """Fill a Fabric, Harbor or Gym runner target that names a registered agent with what that agent is.

    The source keeps its ``agent``, workspace-qualified. A Fabric or Gym target gets the config it resolved
    to as ``resolved_config``; a Harbor target gets it in its kwargs, for the installed Fabric agent its
    source implies. Either way the job runs the agent fresh per trial without ever looking it up itself.

    ``authorize``, when set, checks as the caller what snapshotting the agent's files takes, before taking
    it. With ``snapshot_files`` off, the source names the snapshot without an ID, and nothing is written
    until :func:`take_registered_agent_files_snapshot` takes it.
    """
    if not isinstance(target, (FabricRunnerTarget, HarborRunnerTarget, GymRunnerTarget)):
        return target
    source = registered_agent_source(target)
    if source is None:
        return target
    if registered_agent_config(target) is not None:
        return upgrade_registered_agent_config(target)

    agent = await _load_registered_agent(
        source.agent.root,
        source.environment,
        workspace=workspace,
        async_sdk=async_sdk,
        authorize=authorize,
        snapshot_files=snapshot_files,
    )
    try:
        return _fill_target(target, source, agent)
    except Exception:
        # The load may have snapshotted the agent's files; nothing downstream owns that FileSet until a
        # resolved target exists, so a failure here deletes it rather than leaving one per attempt.
        if agent.files is not None and async_sdk is not None:
            await discard_registered_agent_files(client_from_platform(async_sdk, AsyncFilesClient), agent.files)
        raise


def has_pending_files_snapshot(target: Target | None) -> bool:
    """Whether resolution named a files snapshot for ``target`` and left taking it for later."""
    source = registered_agent_source(target)
    return source is not None and source.files is not None and source.fileset_id is None


async def take_registered_agent_files_snapshot(
    target: Target | None, *, async_sdk: AsyncHelixClient | None
) -> Target | None:
    """Take the files snapshot :func:`resolve_registered_agent` named and left for later, and record its ID.

    Only for a target that resolution just produced: a stored legacy source also lacks an ID, but its snapshot
    exists. A target without a pending snapshot comes back as it is.
    """
    source = registered_agent_source(target)
    if target is None or source is None or not has_pending_files_snapshot(target):
        return target
    assert source.files is not None
    agent = parse_entity_ref(source.agent.root)
    snapshot_workspace, _, snapshot = source.files.root.partition("/")
    assert snapshot_workspace == agent.workspace, "the snapshot is taken in the agent's workspace"
    platform = require_platform_client(async_sdk, "snapshotting a registered agent's files")
    receipt = await snapshot_registered_agent_files(
        client_from_platform(platform, AsyncFilesClient),
        workspace=agent.workspace,
        agent_name=agent.name,
        snapshot=snapshot,
    )
    return target.model_copy(update={"source": source.model_copy(update={"fileset_id": receipt.fileset_id})})


_RunnerTargetT = TypeVar("_RunnerTargetT", bound=FabricRunnerTarget | HarborRunnerTarget | GymRunnerTarget)


def upgrade_registered_agent_config(target: _RunnerTargetT) -> _RunnerTargetT:
    """The target with a config resolved under an older Fabric brought up to what the running one accepts.

    A resolved config is persisted with the spec, so a job submitted before a Fabric upgrade re-parses it
    under the new release. Returns ``target`` itself when nothing needs raising.
    """
    config = registered_agent_config(target)
    if config is None:
        return target
    upgraded = upgrade_relay_observability_version(config)
    if upgraded is config:
        return target
    if isinstance(target, HarborRunnerTarget):
        return target.model_copy(update={"agent_kwargs": {**target.agent_kwargs, "fabric_config": upgraded}})
    return target.model_copy(update={"resolved_config": upgraded})


def _gym_policy_params(config: dict[str, Any]) -> dict[str, Any]:
    """Gym ``policy_*`` settings for a registered agent's default model: its gateway route and model name."""
    default = (config.get("models") or {}).get("default") or {}
    params: dict[str, Any] = {}
    if default.get("base_url"):
        params["policy_base_url"] = default["base_url"]
    if default.get("model"):
        params["policy_model_name"] = default["model"]
    if params:
        params.setdefault("policy_api_key", "not-used")
    return params


def _fill_target(
    target: FabricRunnerTarget | HarborRunnerTarget | GymRunnerTarget,
    source: RegisteredAgentSource,
    agent: _ResolvedRegisteredAgent,
) -> FabricRunnerTarget | HarborRunnerTarget | GymRunnerTarget:
    """The target with the loaded agent written into it, per runner kind. Pure: no I/O, so it can fail freely."""
    files = agent.files.ref if agent.files is not None else agent.pending_files
    resolved_source = source.model_copy(
        update={
            "agent": agent.ref,
            "files": FilesetRef(root=files) if files is not None else None,
            "fileset_id": agent.files.fileset_id if agent.files is not None else None,
        }
    )
    env_secrets = _merge_env_secrets(target.env_secrets, agent.env_secrets, agent=agent.ref.root)
    if isinstance(target, FabricRunnerTarget):
        return target.model_copy(
            update={"source": resolved_source, "resolved_config": agent.config, "env_secrets": env_secrets}
        )
    if isinstance(target, GymRunnerTarget):
        _require_mcp_env_templates_bound(agent.config, {*env_secrets, *target.env_vars}, agent=agent.ref.root)
        hydra_params = {**_gym_policy_params(agent.config), **target.hydra_params}
        return target.model_copy(
            update={
                "source": resolved_source,
                "resolved_config": agent.config,
                "hydra_params": hydra_params,
                "env_secrets": env_secrets,
            }
        )

    # The Fabric agent Harbor installs fills these templates in inside the task container, from the runner's
    # environment, where Harbor puts each `env_secrets` value; for Hermes, Hermes fills them in itself.
    _require_mcp_env_templates_bound(agent.config, {*env_secrets, *target.env_vars}, agent=agent.ref.root)
    adapter_id = agent.config["harness"]["adapter_id"]
    fabric_package = target.agent_kwargs.get("fabric_package")
    if fabric_package is not None and not isinstance(fabric_package, str):
        raise ValueError(
            f"`agent_kwargs.fabric_package` must be a requirement string or null, not {type(fabric_package).__name__}"
        )
    agent_kwargs: dict[str, Any] = {
        **target.agent_kwargs,
        "fabric_config": _without_gateway_placeholders(agent.config),
        "fabric_package": fabric_package if fabric_package is not None else fabric_harness_package(adapter_id),
    }
    return target.model_copy(
        update={"source": resolved_source, "agent_kwargs": agent_kwargs, "env_secrets": env_secrets}
    )
