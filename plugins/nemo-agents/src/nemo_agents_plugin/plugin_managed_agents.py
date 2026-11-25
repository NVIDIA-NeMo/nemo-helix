# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discovery, load-time checks and derivation for plugin-managed agent types.

The *contract* a type implements lives in
:mod:`nemo_helix_plugin.agents.plugin_managed_agents`, so a plugin can write one
without depending on this package. This module is the host half:

- **Discovery.** Types are entry points in
  ``nemo.agents.plugin-managed-agent-types``, named by type ID.
- **Load-time checks.** A type is checked whenever it is loaded, so a broken
  type fails the request that uses it with a message naming the problem,
  rather than producing an agent that cannot run.
- **Derivation.** A plugin-managed agent's stored row holds no config. Every
  read and run fills ``config``, ``config_format``, ``context_schema`` and
  ``post_run_actions`` in from the installed type, so upgrading the plugin
  upgrades every agent of the type.
"""

from __future__ import annotations

import logging
import re
from importlib.metadata import EntryPoint, entry_points

from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.context_schema import check_context_schema
from nemo_agents_plugin.entities import NEMO_AGENTS_SPEC_CONFIG_FORMAT, Agent
from nemo_agents_plugin.entry_points import group_entry_points, load_unique_entry_point
from nemo_agents_plugin.fabric.adapters import UnknownAdapterError, adapter_reads_request_context, resolve_adapter_id
from nemo_agents_plugin.fabric.translator import FabricTranslationError, translate_agent_config
from nemo_agents_plugin.jobs.post_run_actions import validate_post_run_action_kinds
from nemo_helix_plugin.agents.plugin_managed_agents import (
    NEMO_HELIX_CONTEXT_KEY,
    PLUGIN_MANAGED_AGENT_TYPE_ENTRY_POINT_GROUP,
    PluginManagedAgent,
)
from nemo_helix_plugin.agents.types import PLUGIN_MANAGED_AGENT_TYPE_PATTERN, PluginManagedAgentTypeInfo
from nemo_helix_plugin.entity_naming import NAME_PATTERN
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

PluginManagedAgentType = type[PluginManagedAgent]


class PluginManagedAgentTypeError(ValueError):
    """A plugin-managed agent type is not installed, or is installed but broken."""


def load_plugin_managed_agent_type(type_id: str) -> PluginManagedAgentType:
    """Load and check the installed type *type_id*.

    Raises:
        PluginManagedAgentTypeError: It is not installed, is registered twice
            with different implementations, fails to import, or fails a
            load-time check.
    """
    return _load_registered_type(type_id, _registered_entry_points())


def list_plugin_managed_agent_types() -> list[PluginManagedAgentType]:
    """Return every installed type that loads and passes its checks, sorted by type ID.

    A type that fails is logged and left out rather than failing the listing;
    creating or running an agent of that type still reports the real error.
    """
    registered = _registered_entry_points()
    types: list[PluginManagedAgentType] = []
    for type_id in sorted(registered):
        try:
            types.append(_load_registered_type(type_id, registered))
        except PluginManagedAgentTypeError:
            logger.exception("Skipping plugin-managed agent type %r: it could not be loaded.", type_id)
    return types


def type_info(type_cls: PluginManagedAgentType, *, agent: str | None = None) -> PluginManagedAgentTypeInfo:
    """Describe *type_cls* for the types route; *agent* is the workspace's agent of the type, if any."""
    return PluginManagedAgentTypeInfo(
        type=type_cls.type_id,
        title=type_cls.title,
        description=type_cls.description,
        default_name=type_cls.default_name,
        adapter_id=type_cls.adapter_id,
        context_schema=type_cls.context_schema(),
        post_run_actions=sorted(type_cls.post_run_actions),
        subject_label=type_cls.subject_label,
        agent=agent,
    )


def derive_agent(agent: Agent) -> Agent:
    """Return *agent* with what its type supplies filled in.

    An agent that is not plugin-managed is returned unchanged. So is one whose
    type is not installed (or is broken): reading or listing it should not
    fail because of a plugin, and running it reports the problem through
    :func:`plugin_managed_agent_type_for`.
    """
    if agent.plugin_managed_agent is None:
        return agent
    try:
        type_cls = load_plugin_managed_agent_type(agent.plugin_managed_agent.type)
    except PluginManagedAgentTypeError:
        logger.warning(
            "Agent %s/%s is of plugin-managed type %r, which could not be loaded; returning it as stored.",
            agent.workspace,
            agent.name,
            agent.plugin_managed_agent.type,
            exc_info=True,
        )
        return agent
    return derive_agent_of_type(agent, type_cls)


def derive_agent_of_type(agent: Agent, type_cls: PluginManagedAgentType) -> Agent:
    """Return *agent* with *type_cls*'s config, context_schema, default_input and post_run_actions filled in."""
    return agent.model_copy(
        update={
            "config": type_cls.config(name=agent.name),
            "config_format": NEMO_AGENTS_SPEC_CONFIG_FORMAT,
            "context_schema": type_cls.context_schema(),
            "default_input": type_cls.default_input,
            "post_run_actions": set(type_cls.post_run_actions),
        }
    )


def plugin_managed_agent_type_for(agent: Agent) -> PluginManagedAgentType | None:
    """Return *agent*'s type for running it, or ``None`` when it is not plugin-managed.

    Raises:
        ValueError: The type is not installed or is broken, so the agent cannot run.
    """
    if agent.plugin_managed_agent is None:
        return None
    type_id = agent.plugin_managed_agent.type
    try:
        return load_plugin_managed_agent_type(type_id)
    except PluginManagedAgentTypeError as exc:
        raise ValueError(
            f"Agent '{agent.workspace}/{agent.name}' is of plugin-managed agent type {type_id!r}, "
            f"which cannot run on this platform: {exc}"
        ) from exc


def _registered_entry_points() -> dict[str, list[EntryPoint]]:
    return group_entry_points(entry_points(group=PLUGIN_MANAGED_AGENT_TYPE_ENTRY_POINT_GROUP))


def _load_registered_type(type_id: str, registered: dict[str, list[EntryPoint]]) -> PluginManagedAgentType:
    try:
        loaded = load_unique_entry_point(
            type_id, registered, label="plugin-managed agent type", plural="plugin-managed agent types"
        )
    except ValueError as exc:
        raise PluginManagedAgentTypeError(str(exc)) from exc
    except Exception as exc:
        raise PluginManagedAgentTypeError(f"Plugin-managed agent type {type_id!r} failed to import: {exc}") from exc
    try:
        check_plugin_managed_agent_type(loaded, entry_point_name=type_id)
    except PluginManagedAgentTypeError:
        raise
    except Exception as exc:
        # The checks call the plugin's own code (config(), context_schema()); a bug there
        # makes the type unusable, like a failed import, rather than failing every read.
        raise PluginManagedAgentTypeError(
            f"Plugin-managed agent type {type_id!r} failed its load checks: {exc}"
        ) from exc
    return loaded


def check_plugin_managed_agent_type(type_cls: object, *, entry_point_name: str) -> None:
    """Check that *type_cls*, registered as *entry_point_name*, is a usable type.

    Raises:
        PluginManagedAgentTypeError: naming the first problem found.
    """

    def fail(problem: str) -> PluginManagedAgentTypeError:
        return PluginManagedAgentTypeError(f"Plugin-managed agent type {entry_point_name!r} is invalid: {problem}")

    if not (isinstance(type_cls, type) and issubclass(type_cls, PluginManagedAgent)):
        raise fail(f"its entry point must name a PluginManagedAgent subclass, not {type_cls!r}.")
    if not hasattr(type_cls, "context_model"):
        raise fail(
            "it must parameterize PluginManagedAgent with its context model, e.g. PluginManagedAgent[MyContext]."
        )
    for attribute in ("type_id", "title", "adapter_id", "default_name", "post_run_actions"):
        if not getattr(type_cls, attribute, None):
            raise fail(f"it must set {attribute}.")

    if type_cls.type_id != entry_point_name:
        raise fail(f"its type_id {type_cls.type_id!r} must match its entry point name.")
    if not re.fullmatch(PLUGIN_MANAGED_AGENT_TYPE_PATTERN, type_cls.type_id):
        raise fail("its type_id must be '<plugin>.<name>' in lowercase letters, digits, '-' and '_'.")
    if not re.fullmatch(NAME_PATTERN, type_cls.default_name):
        raise fail(f"its default_name {type_cls.default_name!r} is not a valid agent name.")
    if not isinstance(type_cls.post_run_actions, frozenset):
        raise fail("its post_run_actions must be a frozenset of post-run action kinds.")

    for label, model in (("context", type_cls.context_model), ("prepared context", type_cls.prepared_context_model)):
        problem = _context_model_problem(model, label=label)
        if problem is not None:
            raise fail(problem)
    try:
        check_context_schema(type_cls.context_schema())
    except ValueError as exc:
        raise fail(str(exc)) from exc

    try:
        expected_adapter = resolve_adapter_id(type_cls.adapter_id)
    except UnknownAdapterError as exc:
        raise fail(f"its adapter_id is not an installed Fabric adapter: {exc}") from exc

    try:
        config = AgentConfig.model_validate(type_cls.config(name=type_cls.default_name))
    except ValidationError as exc:
        raise fail(f"its config() is not a valid {NEMO_AGENTS_SPEC_CONFIG_FORMAT} config: {exc}") from exc
    default_harness = config.harnesses.get(config.default_harness) if config.default_harness else None
    if default_harness is None:
        raise fail("its config() must select a default harness.")
    try:
        harness_adapter = resolve_adapter_id(default_harness.kind)
    except UnknownAdapterError as exc:
        raise fail(f"its config()'s default harness names an unknown adapter: {exc}") from exc
    if harness_adapter != expected_adapter:
        raise fail(
            f"its config()'s default harness runs {harness_adapter!r}, but its adapter_id is {type_cls.adapter_id!r}."
        )

    context_fields = sorted({*type_cls.context_model.model_fields, *type_cls.prepared_context_model.model_fields})
    if context_fields and not adapter_reads_request_context(harness_adapter):
        raise fail(
            f"its context declares {', '.join(context_fields)}, but its adapter {harness_adapter!r} does not read the "
            "request context, so they would never reach the agent. Run the type on a Fabric adapter that reads "
            "AgentRunRequest.context, or give it an empty context model."
        )
    try:
        translate_agent_config(config)
    except FabricTranslationError as exc:
        raise fail(f"its config() cannot run on Fabric: {exc}") from exc

    try:
        validate_post_run_action_kinds(set(type_cls.post_run_actions))
    except ValueError as exc:
        raise fail(f"it declares a post-run action that cannot be used: {exc}") from exc


def _context_model_problem(model: type[BaseModel], *, label: str) -> str | None:
    if model.model_config.get("extra") != "forbid":
        return f'its {label} model {model.__name__} must set model_config = ConfigDict(extra="forbid").'
    names = set(model.model_fields)
    names.update(field.alias for field in model.model_fields.values() if field.alias)
    if NEMO_HELIX_CONTEXT_KEY in names:
        return f"its {label} model {model.__name__} declares {NEMO_HELIX_CONTEXT_KEY!r}, which the platform reserves."
    return None
