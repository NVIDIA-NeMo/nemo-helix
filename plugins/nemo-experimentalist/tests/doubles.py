# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared doubles for the runner, the context, and anything a strategy touches.

Before the runner existed every test that wanted to drive the loop had to assemble a
backend, an evaluator and a bag of ``deps`` by hand. These three build the same things
once: an in-memory backend that records what it was asked to persist, an evaluator that
returns a fixed result, and a context wired from both.

``conftest.py`` re-exports these as fixtures; import the classes directly when a test
needs to build more than one.
"""

import pathlib
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx
from nemo_experimentalist_plugin.config import CandidateStorageConfig
from nemo_experimentalist_plugin.entities import (
    Candidate,
    Dataset,
    EvaluationResult,
    ExperimentRun,
    MetricTarget,
    Proposal,
    ResourceRef,
    RewardMap,
    RewardRecord,
    TrialResult,
)
from nemo_experimentalist_plugin.experimentalist.components.evaluator.base import Evaluator, EvaluatorConfig
from nemo_experimentalist_plugin.experimentalist.components.repository import AgentSource
from nemo_experimentalist_plugin.experimentalist.context import ExperimentContext
from nemo_experimentalist_plugin.experimentalist.experimentalist_backend import ExperimentalistBackend
from nemo_experimentalist_plugin.experimentalist.reporting import RunReporter
from nemo_experimentalist_plugin.experimentalist.result import ExperimentalistResult
from nemo_insights_plugin.entities import Insight
from nemo_platform_plugin.client.client import AsyncNemoClient


def fake_client() -> AsyncNemoClient:
    """A stand-in platform client for paths that only check whether one is present."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(500, request=request, json={"detail": "fake client"})
    )
    return AsyncNemoClient(
        base_url="http://platform.test",
        workspace="default",
        http_client=httpx.AsyncClient(transport=transport),
    )


class RecordedEvaluation(dict):
    """One ``persist_evaluation`` call, as keyword arguments."""


class FakeBackend(ExperimentalistBackend):
    """In-memory backend that keeps everything it was asked to persist.

    Candidates are keyed by durable id, the way the local backend keys them — a double
    that equated id with display label would make any id/label confusion in the code
    under test pass, which is the defect class this contract exists to separate.
    ``by_label`` is there for assertions that want the readable handle.
    """

    def __init__(
        self,
        *,
        client: AsyncNemoClient | None = None,
        storage: CandidateStorageConfig | None = None,
        insight: Insight | None = None,
    ) -> None:
        super().__init__(client or fake_client(), None, storage)
        self.candidates: dict[str, Candidate] = {}
        self.runs: list[ExperimentRun] = []
        self.evaluations: list[RecordedEvaluation] = []
        self.results: list[ExperimentalistResult] = []
        self.archived: list[str] = []
        self.published: list[str] = []
        self.agent_code_calls: list[tuple[str | Path, Path]] = []
        self._insight = insight

    async def get_insight(self, *, workspace: str, insight_id: str) -> Insight:
        if self._insight is None:
            raise ValueError(f"FakeBackend has no insight to return for {insight_id!r}")
        return self._insight

    async def create_run(self, *, workspace: str, run: ExperimentRun) -> ExperimentRun:
        run._id = run.id or f"run-{len(self.runs) + 1}"  # type: ignore[attr-defined]
        self.runs.append(run)
        return run

    async def update_run(self, *, workspace: str, run: ExperimentRun) -> ExperimentRun:
        return run

    async def create_candidate(self, *, workspace: str, candidate: Candidate) -> Candidate:
        candidate._id = candidate.id or f"id-{uuid.uuid4()}"  # type: ignore[attr-defined]
        self.candidates[candidate.id] = candidate
        return candidate

    async def update_candidate(self, *, workspace: str, candidate: Candidate) -> Candidate:
        self.candidates[candidate.id] = candidate
        return candidate

    async def get_candidate(self, *, workspace: str, candidate_id: str) -> Candidate:
        return self.candidates[candidate_id]

    @property
    def by_label(self) -> dict[str, Candidate]:
        """Stored candidates keyed by display handle, for readable assertions."""
        return {c.label: c for c in self.candidates.values()}

    async def list_candidates(self, *, workspace: str, run_id: str, include_discarded: bool = False) -> list[Candidate]:
        return [c for c in self.candidates.values() if c.run_id == run_id and (include_discarded or not c.discarded)]

    async def persist_result(self, *, workspace: str, result: ExperimentalistResult) -> None:
        self.results.append(result)

    async def persist_evaluation(
        self, *, workspace: str, result: EvaluationResult, candidate: Candidate, split: str
    ) -> None:
        self.evaluations.append(
            RecordedEvaluation(workspace=workspace, result=result, candidate=candidate, split=split)
        )

    async def get_evaluation_name(self, *, workspace: str, candidate: Candidate, split: str) -> str:
        return f"exp-{candidate.label}-{split}"

    async def get_agent_code(
        self, *, workspace: str, agent: str | Path, dest: Path, clone_depth: int | None = None
    ) -> AgentSource | None:
        self.agent_code_calls.append((agent, dest))
        dest.mkdir(parents=True, exist_ok=True)
        return None

    async def archive_candidate(self, *, workspace: str, candidate: Candidate) -> str | None:
        self.archived.append(candidate.label)
        return f"archived://{candidate.label}"

    async def publish_candidate(self, *, workspace: str, candidate: Candidate) -> str | None:
        self.published.append(candidate.label)
        return f"https://example.invalid/pr/{candidate.label}"

    async def get_ethos(self, *, workspace: str, ethos: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"# Ethos from {ethos}\n")
        return dest


class FakeEvaluator(Evaluator):
    """Returns a fixed reward and records the options each run was given."""

    evaluator_type = "harbor"

    def __init__(self, *, reward: float = 0.5, options: EvaluatorConfig | None = None) -> None:
        super().__init__(options or EvaluatorConfig(), None)
        self.reward = reward
        self.runs: list[dict[str, object]] = []

    async def _run(self, agent: Path, dataset: Dataset, options: EvaluatorConfig) -> Sequence[TrialResult]:
        self.runs.append({"agent": agent, "dataset": dataset, "options": options})
        return [
            TrialResult(id=f"{agent.name}-{task.id}", task_id=task.id, status="completed")
            for task in dataset.list_tasks()
        ]

    async def aggregate_results(self, results: Sequence[TrialResult]) -> dict[str, float | int]:
        return {"reward": self.reward}


#: Artifacts must exist -- ``candidate_dir`` refuses a record whose resource has gone,
#: because resolving it to the shared root is what let a stale record copy every
#: candidate's code out as its own. Doubles therefore materialise a real directory.
_ARTIFACT_ROOT = pathlib.Path(tempfile.mkdtemp(prefix="experimentalist-doubles-"))


def make_candidate(
    *,
    label: str = "agent-0",
    run_id: str = "run-1",
    generation: int = 0,
    description: str | None = None,
    ancestor: str | None = None,
    optimization_type: str | None = None,
    artifact: str | None = None,
    candidate_id: str | None = None,
    rewards: Mapping[str, RewardRecord] | None = None,
    killed_generation: int | None = None,
    workspace: str = "default",
) -> Candidate:
    """A Candidate with a synthetic artifact, for tests that only care about metadata.

    Every real Candidate is created by ``ctx.commit_candidate()`` from a finished
    artifact; a test that is not exercising that path still needs one, so this supplies
    a plausible directory URI and derives the projections the same way the context does.
    """
    text = description if description is not None else ("baseline" if ancestor is None else f"change in {label}")
    # Every candidate has an origin, including the baseline: its Proposal asks for the
    # agent under test to be imported unchanged, and `ancestor is None` is what makes it
    # the baseline.
    proposal = Proposal(
        ancestor=ancestor,
        description=text,
        kind="code-change" if ancestor is not None else "import",
        payload={"optimization_type": optimization_type} if optimization_type else {},
    )
    candidate = Candidate(
        name=label,
        label=label,
        run_id=run_id,
        generation=generation,
        ancestor=ancestor,
        generated_from=proposal,
        description=text,
        artifact=ResourceRef(uri=artifact or _default_artifact(run_id, label)),
        rewards=RewardMap(rewards or {}),
        killed_generation=killed_generation,
        workspace=workspace,
    )
    # Distinct from the label by default: identity and the display handle are different
    # strings in a real run, and a double that conflates them hides that class of bug.
    candidate._id = candidate_id or f"id-{label}"  # type: ignore[attr-defined]
    return candidate


def _default_artifact(run_id: str, label: str) -> str:
    """A real, empty directory standing in for a built candidate."""
    path = _ARTIFACT_ROOT / run_id / "eval-and-optimize" / "agents" / label
    path.mkdir(parents=True, exist_ok=True)
    return path.as_uri()


def make_context(
    *,
    root: Path,
    backend: ExperimentalistBackend | None = None,
    evaluator: Evaluator | None = None,
    datasets: Mapping[str, Dataset] | None = None,
    run: ExperimentRun | None = None,
    workspace: str = "default",
    resuming: bool = False,
    reporter: RunReporter | None = None,
    objective_metrics: list[MetricTarget] | None = None,
    regression_metrics: list[MetricTarget] | None = None,
) -> ExperimentContext:
    """Build a context over *root*, defaulting every collaborator to a double."""
    run = run or ExperimentRun(workspace=workspace, agent="agent-under-test")
    if not run.id:
        run._id = "run-1"  # type: ignore[attr-defined]
    agent_dir = root / "eval-and-optimize" / "source-agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    return ExperimentContext(
        backend=backend or FakeBackend(),
        workspace=workspace,
        run=run,
        root=root,
        agent_dir=agent_dir,
        datasets=datasets or {"train": Dataset(id="train"), "validation": Dataset(id="validation")},
        evaluator=evaluator or FakeEvaluator(),
        resuming=resuming,
        reporter=reporter,
        objective_metrics=objective_metrics,
        regression_metrics=regression_metrics,
    )


def seed_reward(candidate: Candidate, channel: str, record: RewardRecord) -> None:
    """Put a measurement on a candidate for a fixture, without persisting it.

    Production writes go through ``ctx.record_reward``; this exists so a test can build
    a candidate that is already scored without standing up a store.
    """
    dict.__setitem__(candidate.rewards, channel, record)
