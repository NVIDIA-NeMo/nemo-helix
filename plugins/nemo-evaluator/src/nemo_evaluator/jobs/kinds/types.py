# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kind adapters bridge stored definitions, resolved snapshots, and runtime tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from nemo_evaluator.entities import TaskEntity, TaskRevisionEntity
from nemo_evaluator.jobs.agent_spec import AgentEvalTaskInput, ResolvedTask, ResolvedTaskDefinition, Target
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial
from nemo_helix_plugin.client.adapter import AsyncHelixClient
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.entities import EntityClient


@dataclass(frozen=True)
class LoadedTask:
    """A task after load and before metric resolution. Exactly one of ``inline`` or ``stored`` is set.

    - ``inline``: the resolved evaluator task body was sent on the submission, so nothing is read from the store.
    - ``stored``: the submission named a published task, either a ``TaskRef`` or a taskset member. Harbor
      tasks are always stored.

    Both fields exist so snapshot and kind adapters handle either source as one value. A submission
    cannot mix them.

    ``stored`` keeps the task record and the one revision the ref named.
      - the ``TaskEntity`` supplies ``workspace/name``
      - the ``TaskRevisionEntity`` supplies the definition and content hash.
    """

    inline: AgentEvalTaskInput | None = None
    stored: tuple[TaskEntity, TaskRevisionEntity] | None = None

    def __post_init__(self) -> None:
        if (self.inline is None) == (self.stored is None):
            raise ValueError("LoadedTask needs exactly one of inline or stored")

    @property
    def kind(self) -> str:
        """Return the discriminator for the inline or stored task definition.

        Raises:
            ValueError: The stored task payload is incomplete.
        """
        if self.inline is not None:
            return "evaluator"
        if self.stored is None or len(self.stored) != 2:
            raise ValueError("LoadedTask stored tasks need a task and a revision")
        return self.stored[1].spec.kind


@dataclass(frozen=True)
class SubmitContext:
    workspace: str
    entity_client: EntityClient | None
    async_sdk: AsyncHelixClient | None
    adapters: Mapping[str, TaskKindAdapter]


@dataclass(frozen=True)
class PrepareContext:
    storage_root: Path
    client: NemoClient | None
    async_client: AsyncNemoClient | None
    target: Target | None
    trials: Sequence[AgentEvalTrial] | None
    adapters: Mapping[str, TaskKindAdapter]


class TaskKindAdapter(Protocol):
    kind: str

    def runtime_id(self, item: LoadedTask) -> str:
        """Choose the evaluator runtime ID during submission-time loading and snapshotting."""
        ...

    async def resolve(self, item: LoadedTask, ctx: SubmitContext) -> ResolvedTaskDefinition:
        """Return a complete definition with resolved metrics and stored revision provenance, if any."""
        ...

    def accepts_target(self, target: Target | None, tasks: Sequence[ResolvedTask]) -> bool:
        """Check target compatibility at submission and again on the worker."""
        ...

    def validate_scoring(
        self, tasks: Sequence[ResolvedTask], *, target: Target | None, trials: Sequence[AgentEvalTrial] | None
    ) -> None:
        """Validate scoring configuration at submission and again on the worker."""
        ...

    def prepare(self, tasks: Sequence[ResolvedTask], ctx: PrepareContext) -> list[AgentEvalTask]:
        """Run on the worker to prepare resolved tasks for execution or offline scoring."""
        ...
