# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a component sees of the run, and the working copy a Builder builds in.

Protocols rather than the concrete context, so a role contract does not import the
implementation: naming ``ExperimentContext`` in :mod:`roles` closes an import cycle
through the backend, the config tree, and every component config slice. It also lets an
out-of-tree strategy type its own ``run`` without importing our internals.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from nemo_experimentalist_plugin.entities import (
    Candidate,
    Dataset,
    DataValue,
    EvaluationResult,
    MetricTarget,
    Proposal,
    ResourceRef,
    RewardRecord,
)
from nemo_experimentalist_plugin.experimentalist.components.evaluator import Evaluator
from nemo_experimentalist_plugin.experimentalist.components.trace_explorer import TraceExplorer
from pydantic import BaseModel, ConfigDict, Field

#: The split a candidate's headline reward is measured on, and the winner chosen by.
PRIMARY_SPLIT = "validation"


class Fork(BaseModel):
    """A working copy, and the thing it was forked from, as one value."""

    model_config = ConfigDict(frozen=True)

    workdir: Path = Field(description="Write here. Seeded from upstream, and this becomes the candidate's artifact.")
    # Still needed after seeding: once the Builder edits, the parent is gone from its
    # view, and architecture.md is excluded from the seeding by design.
    upstream: Path | None = Field(
        default=None, description="The pristine parent, read-only; None when forked from the agent under test."
    )


TraceLoader = Callable[[ResourceRef], Awaitable[TraceExplorer]]
"""Resolves a trace reference to a loaded trace, wherever it is stored.

The context supplies one to any component whose constructor names ``load_trace``, so a
component that reads traces never names a platform type and one shipped by another
package can read them too.
"""


@runtime_checkable
class BuilderContext(Protocol):
    """Everything a Builder may reach, which is two verbs: get a place to work, hand back
    the finished thing.

    Both take the ``Proposal`` and return a ``Fork`` or a ``Candidate``. A Builder never
    handles an id and never receives a path from outside. Resolving a proposal's ancestor
    to a location is the context's job, which is what lets candidate storage move without
    changing a single Builder signature.
    """

    async def fork(self, proposal: Proposal) -> Fork:
        """Reserve a working copy for *proposal*, seeded from what it branches off."""
        ...

    async def commit_candidate(self, *, proposal: Proposal, artifact: Path, generation: int = 0) -> Candidate:
        """Validate the finished artifact and create the Candidate addressing it."""
        ...


@runtime_checkable
class StrategyContext(BuilderContext, Protocol):
    """Everything the run offers, which is what a strategy orchestrating it needs.

    Wider than :class:`BuilderContext` on purpose. The point is not to restrict the
    strategy — it is the thing in charge — but to let ``roles`` and any out-of-tree
    strategy name this contract without importing the implementation behind it.
    """

    #: True when the runner re-opened an existing run, so the strategy should rebuild its
    #: state from :meth:`candidates` rather than starting over.
    resuming: bool

    #: NeMo Helix workspace this run belongs to.
    workspace: str

    @property
    def run_id(self) -> str:
        """Durable id of this run."""
        ...

    #: Evaluator-domain datasets keyed by split. ``validation`` is always present.
    datasets: Mapping[str, Dataset]

    #: Markdown description of the agent under test, when the run has one.
    ethos: Path | None

    @property
    def objective_metrics(self) -> list[MetricTarget]:
        """The run's effective objectives.

        Settled by the host before the strategy starts, because authoring an Insight
        suite can narrow them: the authored verifiers emit their own metric keys, and
        those become what the run is scored against. A strategy resolved before that
        happens must read the contract from here rather than from its own config.
        """
        ...

    @property
    def regression_metrics(self) -> list[MetricTarget]:
        """The run's effective guardrails, settled alongside :attr:`objective_metrics`."""
        ...

    @property
    def outcome_evaluator(self) -> Evaluator:
        """The run's configured evaluation component."""
        ...

    async def load_trace(self, ref: ResourceRef) -> TraceExplorer:
        """The trace *ref* names, wherever this run stores traces."""
        ...

    async def candidates(self) -> list[Candidate]:
        """Every Candidate committed to this run. What makes resume possible."""
        ...

    def candidate_dir(self, candidate: Candidate) -> Path:
        """Where *candidate*'s artifact lives, raising if the record outlived it."""
        ...

    async def update_candidate(self, candidate: Candidate, **fields: object) -> Candidate:
        """Persist changes to a Candidate that already exists."""
        ...

    async def discard_candidate(self, candidate: Candidate) -> None:
        """Remove a Candidate entirely: its artifact and the record addressing it."""
        ...

    async def archive_candidate(self, candidate: Candidate) -> None:
        """Persist a Candidate's code to durable storage, if this run archives at all."""
        ...

    async def record_reward(
        self,
        candidate: Candidate,
        *,
        channel: str,
        result: EvaluationResult | RewardRecord,
        metadata: dict[str, DataValue] | None = None,
    ) -> None:
        """Store one measurement of *candidate* on *channel*. Universal."""
        ...

    async def evaluate(
        self,
        candidate: Candidate,
        *,
        split: str = "validation",
        task_ids: Sequence[str] | None = None,
        minimum_attempts: int | None = None,
    ) -> EvaluationResult:
        """Run the configured evaluation component. Optional: a self-scoring strategy
        skips this and calls :meth:`record_reward` directly."""
        ...

    async def report_progress(
        self, *, completed: int, total: int | None = None, unit: str = "step", note: str | None = None
    ) -> None:
        """Report progress as a counter, not a fraction — opaque strategies cannot
        compute one."""
        ...

    def note(self, message: str) -> None:
        """Narrate, for a human watching. Touches no entity and never raises."""
        ...

    def component(self, role: str, name: str, **kwargs: object) -> object:
        """Resolve a component by name, so a strategy need not import the registry."""
        ...
