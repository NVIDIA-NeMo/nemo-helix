# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-author contract for plugin-managed agent types.

A plugin-managed agent type lets a plugin ship an agent that users create by
type instead of by config: ``nemo agents create <plugin-managed-agent-type>``.
The Agents service stores only ``{name, plugin_managed_agent: {type}}`` and derives the
agent's ``config``, ``context_schema`` and ``post_run_actions`` from the installed
type whenever the agent is read or run, so the agent never goes stale when the
plugin is upgraded.

A type is a :class:`PluginManagedAgent` subclass registered as an entry point in
:data:`PLUGIN_MANAGED_AGENT_TYPE_ENTRY_POINT_GROUP`, named by its
:attr:`~PluginManagedAgent.type_id`. Example:

    [project.entry-points."nemo.agents.plugin-managed-agent-types"]
    "my-plugin.reviewer" = "my_plugin.reviewer:Reviewer"

This module holds only what an *implementer* needs to import.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Protocol, cast, get_args, get_origin

from nemo_helix_plugin.agents.client import AsyncAgentsClient
from nemo_helix_plugin.agents.types import NEMO_AGENTS_SPEC_CONFIG_FORMAT, Agent
from nemo_helix_plugin.client.client import AsyncNemoClient
from pydantic import BaseModel
from typing_extensions import TypeVar

PLUGIN_MANAGED_AGENT_TYPE_ENTRY_POINT_GROUP = "nemo.agents.plugin-managed-agent-types"

# The Fabric request context key the platform reserves for its own fields.
# Neither context model may declare it.
NEMO_HELIX_CONTEXT_KEY = "nemo_helix"

# JSON Schema extension keyword naming where a client may find a default for a
# context property the caller left out. Hints are for clients only (the CLI
# fills them from its configured models, Studio from its settings): the server
# never invents a value, so a run's context is always what its caller sent.
HELIX_DEFAULT_SCHEMA_KEY = "x-helix-default"
#: The CLI context's configured default model, as a ``workspace/name`` Model Entity ref.
CONFIGURED_DEFAULT_MODEL = "configured_default_model"
#: The CLI context's configured fast model (falling back to the default model).
CONFIGURED_FAST_MODEL = "configured_fast_model"
HELIX_DEFAULT_HINTS = frozenset({CONFIGURED_DEFAULT_MODEL, CONFIGURED_FAST_MODEL})

# The harness name in a type's default config.
DEFAULT_HARNESS_NAME = "default"


def helix_default(hint: str) -> dict[str, Any]:
    """Return ``json_schema_extra`` marking a context field with a client-side default hint.

    Use it on a :class:`PluginManagedAgent` context model::

        default_model: str = Field(json_schema_extra=helix_default(CONFIGURED_DEFAULT_MODEL))
    """
    if hint not in HELIX_DEFAULT_HINTS:
        raise ValueError(
            f"Unknown {HELIX_DEFAULT_SCHEMA_KEY} hint {hint!r}; use one of: {sorted(HELIX_DEFAULT_HINTS)}."
        )
    return {HELIX_DEFAULT_SCHEMA_KEY: hint}


class PluginManagedAgentInputError(ValueError):
    """Raised by :meth:`PluginManagedAgent.prepare` when the caller's context cannot be run.

    The Agents service reports it to the caller as a 422 on the job create
    request, with this error's message. Raise it for problems the caller can
    fix (a model that does not exist, a target that cannot be found), not for
    plugin bugs.
    """


ContextT = TypeVar("ContextT", bound=BaseModel)
PreparedContextT = TypeVar("PreparedContextT", bound=BaseModel, default=ContextT)
_RunContextT = TypeVar("_RunContextT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class PreparedRun(Generic[_RunContextT]):
    """What :meth:`PluginManagedAgent.prepare` resolves one run into."""

    #: The context the Fabric adapter receives, as the type's prepared context model.
    context: _RunContextT
    #: What the run is about, stamped on the job as ``spec.subject`` so a subject's runs can be listed
    #: with ``filter[spec.subject]``. Its meaning is the type's own: a type that reviews another agent
    #: might use that agent's name. ``None`` when the run has no natural subject.
    subject: str | None = None


class _AgentLike(Protocol):
    @property
    def plugin_managed_agent(self) -> Any: ...


class PluginManagedAgent(Generic[ContextT, PreparedContextT]):
    """Base class for a plugin-managed agent type.

    Parameterize it with the type's context models::

        class Reviewer(PluginManagedAgent[ReviewerContext, ReviewerPreparedContext]): ...

    ``ContextT`` is the user-facing contract: what callers pass as an execute
    job's ``context``. Its JSON Schema is the agent's public ``context_schema``.
    ``PreparedContextT`` is the adapter-facing contract: what :meth:`prepare`
    produces, what the job snapshots, what the Fabric adapter parses with
    :meth:`parse_context`, and what post-run actions receive as ``run_context``.
    Most types need one model; pass a second only when the two differ in shape.
    Both must set ``model_config = ConfigDict(extra="forbid")``.

    Types are never instantiated: every member is a class attribute or classmethod.
    """

    #: Unique type ID, ``<plugin>.<name>``; also the entry point's name.
    type_id: ClassVar[str]
    #: Human-readable name.
    title: ClassVar[str]
    #: One-line summary of what agents of this type do.
    description: ClassVar[str] = ""
    #: The Fabric adapter that runs agents of this type.
    adapter_id: ClassVar[str]
    #: Post-run action kinds an execute job of this type may choose.
    post_run_actions: ClassVar[frozenset[str]]
    #: Agent name ``nemo agents create <type>`` uses when given no ``--name``.
    default_name: ClassVar[str]
    #: Short label for what :attr:`PreparedRun.subject` names, e.g. ``"Analyzed agent"``; ``None`` if runs have none.
    subject_label: ClassVar[str | None] = None
    #: The agent's ``default_input``: the prompt a job that omits ``input`` hands the adapter. The
    #: adapter owns the prompt and may ignore it, building its own from the prepared context; a
    #: placeholder suits such a type. ``None`` makes every job pass ``input``.
    default_input: ClassVar[str | None] = None

    #: Set from the generic arguments.
    context_model: ClassVar[type[BaseModel]]
    prepared_context_model: ClassVar[type[BaseModel]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for base in getattr(cls, "__orig_bases__", ()):
            if get_origin(base) is not PluginManagedAgent:
                continue
            context_model, prepared_context_model = get_args(base)
            if not (isinstance(context_model, type) and issubclass(context_model, BaseModel)):
                raise TypeError(f"{cls.__name__}: the context model must be a pydantic BaseModel subclass.")
            if isinstance(prepared_context_model, TypeVar):
                # Only the context model was given: the adapter receives the same shape.
                prepared_context_model = context_model
            if not (isinstance(prepared_context_model, type) and issubclass(prepared_context_model, BaseModel)):
                raise TypeError(f"{cls.__name__}: the prepared context model must be a pydantic BaseModel subclass.")
            cls.context_model = context_model
            cls.prepared_context_model = prepared_context_model

    @classmethod
    def config(cls, *, name: str) -> dict[str, Any]:
        """Return the ``nemo-agents-spec-v1`` config an agent of this type named *name* runs.

        The default is one local harness of kind :attr:`adapter_id` with no
        settings and no models, which suits an adapter that reads everything it
        needs, models included, from the run context; its Fabric manifest then
        leaves ``models`` out of ``config.accepts``. Override it to add what the
        stock config lacks, such as the model an adapter that takes one needs,
        and start from ``super().config(name=name)``. The host checks the result
        when it loads the type: its default harness must still be of kind
        :attr:`adapter_id`, and it must translate to a Fabric config.
        """
        return {
            "config_format": NEMO_AGENTS_SPEC_CONFIG_FORMAT,
            "name": name,
            "description": cls.description or cls.title,
            "default_harness": DEFAULT_HARNESS_NAME,
            "harnesses": {DEFAULT_HARNESS_NAME: {"kind": cls.adapter_id}},
            "environment": {"provider": "local"},
        }

    @classmethod
    def context_schema(cls) -> dict[str, Any]:
        """Return the JSON Schema of the user-facing context: the agent's ``context_schema``."""
        return cls.context_model.model_json_schema()

    @classmethod
    def parse_context(cls, raw: Mapping[str, Any]) -> PreparedContextT:
        """Parse a Fabric request context into the prepared context model.

        For the type's Fabric adapter. Drops the platform's reserved
        ``nemo_helix`` key, which the job adds at run time, before validating.
        """
        fields = {key: value for key, value in raw.items() if key != NEMO_HELIX_CONTEXT_KEY}
        return cast(PreparedContextT, cls.prepared_context_model.model_validate(fields))

    @classmethod
    async def prepare(
        cls, context: ContextT, *, workspace: str, client: AsyncNemoClient
    ) -> PreparedRun[PreparedContextT]:
        """Resolve a run's context into what the adapter receives.

        Called by the Agents service when an execute job is created, after
        *context* has been validated against the context model, with a
        *client* that acts as the job's caller: a resource the caller cannot
        read fails the request, not the job. This is where per-run lookups go
        (qualifying model names, reading files), and where the user-facing
        context becomes the adapter-facing one.

        Raise :class:`PluginManagedAgentInputError` for a context the caller
        can fix. The default, for a type with one context model, passes the
        context through unchanged.
        """
        del workspace, client
        if cls.prepared_context_model is not cls.context_model:
            raise NotImplementedError(
                f"{cls.__name__} has a separate prepared context model, so it must implement prepare()."
            )
        # The models are one and the same here, so the context already is the prepared context.
        return cast(PreparedRun[PreparedContextT], PreparedRun(context=context))

    @classmethod
    def is_instance(cls, agent: _AgentLike) -> bool:
        """Return whether *agent* (an Agent entity or API model) is of this type."""
        ref = agent.plugin_managed_agent
        if ref is None:
            return False
        type_id = ref.get("type") if isinstance(ref, Mapping) else getattr(ref, "type", None)
        return type_id == cls.type_id

    @classmethod
    async def find_agent(cls, *, workspace: str, client: AsyncNemoClient) -> Agent | None:
        """Return the workspace's agent of this type, or ``None`` when it has none.

        Should the workspace hold more than one, this is the oldest, the same
        agent the plugin-managed agent types route reports.
        """
        agents_client = AsyncAgentsClient.from_client(client)
        response = await agents_client.list_agents(
            workspace=workspace,
            query_params={
                "filter": json.dumps({"plugin_managed_agent.type": cls.type_id}),
                "sort": "created_at",
                "page_size": 1,
            },
        )
        items = response.page().items
        return items[0] if items else None
