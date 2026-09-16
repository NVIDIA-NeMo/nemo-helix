# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted lifecycle extensions for ``agents.execute`` jobs."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, ClassVar, Protocol

from nemo_agents_plugin.fabric.runtime import FabricRuntimeResult
from nemo_helix_plugin.job_context import JobContext
from pydantic import BaseModel, ConfigDict, ValidationError

EXECUTE_AGENT_EXTENSION_ENTRY_POINT_GROUP = "nemo.agents.execute_extensions"
NOOP_EXECUTE_AGENT_EXTENSION_KIND = "noop"


class ExecuteAgentExtension(Protocol):
    """Plugin-owned deterministic work tied to the execute-agent lifecycle."""

    # Declared so ``extension.config`` can be rejected on the create request
    # instead of deep inside ``after_invoke``, after a full Fabric run. Point it
    # at the same model ``after_invoke`` parses, so create-time and run-time
    # validation cannot drift. ``validate_execute_agent_extension_config``
    # tolerates its absence for out-of-tree extensions.
    config_model: ClassVar[type[BaseModel]]

    def after_invoke(self, context: ExecuteAgentAfterInvokeContext) -> None:
        """Run after a successful Fabric invocation."""


class NoopExecuteAgentExtensionConfig(BaseModel):
    """The noop extension takes no configuration at all."""

    model_config = ConfigDict(extra="forbid")


class NoopExecuteAgentExtension:
    """Default extension used when no plugin extension is configured."""

    config_model: ClassVar[type[BaseModel]] = NoopExecuteAgentExtensionConfig

    def after_invoke(self, context: ExecuteAgentAfterInvokeContext) -> None:
        del context


@dataclass(frozen=True, slots=True)
class ExecuteAgentAfterInvokeContext:
    """Inputs available to a trusted execute-agent extension after Fabric succeeds."""

    ctx: JobContext
    config: dict[str, Any]
    agent_name: str
    fabric_result: FabricRuntimeResult


def resolve_execute_agent_extension(kind: str) -> type[ExecuteAgentExtension]:
    """Resolve an installed trusted execute-agent extension kind."""
    return _load_execute_agent_extension(kind)


def validate_execute_agent_extension_config(kind: str, config: dict[str, Any]) -> None:
    """Reject a malformed extension config body.

    Lenient by design: an extension that declares no ``config_model`` is
    accepted unvalidated rather than rejected.
    """
    config_model = getattr(_load_execute_agent_extension(kind), "config_model", None)
    if config_model is None:
        return
    try:
        config_model.model_validate(config)
    except ValidationError as exc:
        # The caller supplied a `kind`, not a model name; lead with the kind.
        raise ValueError(f"Invalid config for agents.execute extension {kind!r}: {exc}") from exc


def run_execute_agent_after_invoke_extension(kind: str, context: ExecuteAgentAfterInvokeContext) -> None:
    """Run the ``after_invoke`` lifecycle method for an installed extension kind."""
    extension_cls = _load_execute_agent_extension(kind)
    extension_cls().after_invoke(context)


def _load_execute_agent_extension(kind: str) -> type[ExecuteAgentExtension]:
    if kind == NOOP_EXECUTE_AGENT_EXTENSION_KIND:
        return NoopExecuteAgentExtension

    matches = [
        entry_point
        for entry_point in entry_points(group=EXECUTE_AGENT_EXTENSION_ENTRY_POINT_GROUP)
        if entry_point.name == kind
    ]
    if not matches:
        raise ValueError(f"Unknown agents.execute extension {kind!r}.")

    # A kind may legitimately be declared by more than one installed
    # distribution: the aggregate ``nemo-helix`` wheel re-declares every
    # bundled plugin's entry points, so a standard install sees each of them
    # twice — once from the plugin, once from the aggregate. What must be
    # unique is the *implementation*, not the number of declarations, so
    # collapse identical targets and reject only genuine conflicts.
    targets = {entry_point.value for entry_point in matches}
    if len(targets) > 1:
        conflicting = ", ".join(sorted(targets))
        raise ValueError(f"Conflicting agents.execute extensions are registered for {kind!r}: {conflicting}.")
    return matches[0].load()
