# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-author contract for ``agents.execute`` lifecycle extensions.

An execute-agent extension is deterministic, plugin-owned work that runs after a
successful Fabric invocation inside an ``agents.execute`` job — persisting a
result, emitting telemetry, and so on. A plugin registers one by declaring an
entry point in the :data:`EXECUTE_AGENT_EXTENSION_ENTRY_POINT_GROUP` group.

This module holds only what an *implementer* needs to import. The machinery that
discovers and dispatches extensions is the host's, and lives in
``nemo_agents_plugin.jobs.execute_extensions``. Keeping the contract here means a
plugin implementing an extension does not take a dependency on the agents plugin
and its harness stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

from nemo_platform_plugin.job_context import JobContext
from pydantic import BaseModel

EXECUTE_AGENT_EXTENSION_ENTRY_POINT_GROUP = "nemo.agents.execute_extensions"


@dataclass(frozen=True, slots=True)
class FabricRuntimeResult:
    """Platform-normalized result for one Fabric runtime invocation.

    This shape preserves Fabric's correlation IDs separately so it can later map
    cleanly into Platform's ``AgentRun.output`` / ``RunOutput``. It is
    deliberately free of Fabric types so the extension contract does not drag
    the Fabric runtime into a plugin that only reads a result.
    """

    status: str
    output: Any = None
    response: Any | None = None
    error: Any | None = None
    artifacts: Any | None = None
    telemetry: list[Any] = field(default_factory=list)
    events: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    runtime_id: str | None = None
    invocation_id: str | None = None
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class ExecuteAgentAfterInvokeContext:
    """Inputs available to a trusted execute-agent extension after Fabric succeeds."""

    ctx: JobContext
    config: dict[str, Any]
    agent_name: str
    fabric_result: FabricRuntimeResult


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
