# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-author contract for ``agents.execute`` post-run actions.

A post-run action is deterministic, plugin-owned work that runs after a
successful Fabric invocation, inside the execute step of an ``agents.execute``
job — persisting a result, emitting telemetry, and so on. Its failure fails the
job. It never runs for an ad hoc ``nemo agents invoke``. A plugin registers one
by declaring an entry point in the :data:`POST_RUN_ACTION_ENTRY_POINT_GROUP`
group; the entry point's name is the action's ``kind``.

This module holds only what an *implementer* needs to import. The machinery that
discovers and dispatches actions is the host's, and lives in
``nemo_agents_plugin.jobs.post_run_actions``. Keeping the contract here means a
plugin implementing an action does not take a dependency on the agents plugin
and its harness stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

from nemo_helix_plugin.job_context import JobContext
from pydantic import BaseModel

POST_RUN_ACTION_ENTRY_POINT_GROUP = "nemo.agents.post_run_actions"


@dataclass(frozen=True, slots=True)
class FabricRuntimeResult:
    """Platform-normalized result for one Fabric runtime invocation.

    This shape preserves Fabric's correlation IDs separately so it can later map
    cleanly into Platform's ``AgentRun.output`` / ``RunOutput``. It is
    deliberately free of Fabric types so the post-run action contract does not
    drag the Fabric runtime into a plugin that only reads a result.
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
class PostRunActionContext:
    """Inputs available to a post-run action after Fabric succeeds."""

    ctx: JobContext
    config: dict[str, Any]
    agent_name: str
    # The resolved agent config the job ran: from the Agent entity or inline
    # definition, merged with its AgentEnvironment's spec, as snapshotted when
    # the job was created. Excludes per-run wiring such as the telemetry proxy
    # address, and is the action's own copy.
    agent_config: dict[str, Any]
    fabric_result: FabricRuntimeResult


class PostRunAction(Protocol):
    """Plugin-owned deterministic work run after a successful agent invocation."""

    # Display metadata, surfaced by ``GET /apis/agents/v2/post-run-actions``.
    # ``description`` is one line.
    title: ClassVar[str]
    description: ClassVar[str]

    # Declared so ``post_run_action.config`` can be rejected on the create
    # request instead of deep inside ``after_invoke``, after a full Fabric run.
    # Point it at the same model ``after_invoke`` parses, so create-time and
    # run-time validation cannot drift. ``validate_post_run_action_config``
    # tolerates its absence for out-of-tree actions.
    config_model: ClassVar[type[BaseModel]]

    def after_invoke(self, context: PostRunActionContext) -> None:
        """Run after a successful Fabric invocation."""
