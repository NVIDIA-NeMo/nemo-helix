# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The plugin-facing seam: everything a strategy is allowed to reach.

``ExperimentContext`` is the implementation; ``StrategyContext`` in :mod:`seam` is the
Protocol it satisfies, and is what components are typed against so a role contract does
not depend on this module.

A strategy receives one context and nothing else. The context holds
the :class:`~nemo_experimentalist_plugin.experimentalist.experimentalist_backend.ExperimentalistBackend`
privately, so no component ever sees ``create_run``, ``publish_candidate``, or the
platform client; the runner that built the context is the only code that holds a
backend.

The two keyword arguments below are deliberately different words. ``evaluate(split=…)``
names a *dataset split* to run against; ``record_reward(channel=…)`` names a *reward
channel* to store under. Usually a run on the validation split lands in the
``validation`` channel, but trajectory scoring produces a second channel from the same
split, which is why one is not spelled with the other's name.
"""

from __future__ import annotations

import fnmatch
import inspect
import logging
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from nemo_experimentalist_plugin.entities import (
    Candidate,
    Dataset,
    DataValue,
    EvaluationResult,
    ExperimentRun,
    MetricTarget,
    Proposal,
    ResourceRef,
    RewardRecord,
    local_path_from_uri,
)
from nemo_experimentalist_plugin.experimentalist.components.evaluator import Evaluator
from nemo_experimentalist_plugin.experimentalist.components.models import missing_objective_reason
from nemo_experimentalist_plugin.experimentalist.components.trace_explorer import TraceExplorer
from nemo_experimentalist_plugin.experimentalist.experimentalist_backend import (
    ExperimentalistBackend,
)
from nemo_experimentalist_plugin.experimentalist.registry import resolve, validated_config
from nemo_experimentalist_plugin.experimentalist.reporting import RunReporter
from nemo_experimentalist_plugin.experimentalist.seam import PRIMARY_SPLIT, Fork

logger = logging.getLogger(__name__)


def _accepted(constructor: Any, supplied: dict[str, Any]) -> dict[str, Any]:
    """Narrow *supplied* to the arguments *constructor* names.

    Named parameters only. A ``**kwargs`` on a component almost always means "forward
    the rest to my base class" rather than "I tolerate anything", so treating it as
    permission hands an nooa ``Agent`` a ``workspace`` it will reject.
    """
    parameters = inspect.signature(constructor).parameters
    return {
        key: value
        for key, value in supplied.items()
        if parameters.get(key) is not None and parameters[key].kind is not inspect.Parameter.VAR_KEYWORD
    }


#: What a fork must not carry from its source, composed from the owners that actually
#: contribute names: this run's own layout, generic developer hygiene, the evaluator's
#: scratch, and the strategy's generated documentation. Hardcoding one flat list is how
#: a third-party strategy's real output gets stripped.
#: Directories the run owns. A fork must not inherit them, and the winner copy-out
#: must not overwrite them -- both directions of the same rule, so it lives here once.
RUN_LAYOUT = frozenset({"eval-and-optimize", "artifacts", "dataset", "scratch"})
_HYGIENE = frozenset({"__pycache__", ".git", ".runtime-cache", ".claude", ".uv", ".venv"})
_EVALUATOR_SCRATCH_GLOBS = ("*traces*", "*eval-and-optimize_*")
#: Generated *about* the ancestor rather than part of it, and the Proposer's only view of
#: an agent — it is told to reason from this file and not to read source. The Builder
#: re-seeds it from the ancestor after building and rewrites it against the finished
#: source, so the fork must leave it absent: absence is what tells the Proposer it has no
#: model of a candidate. Inherit it and a failed regeneration silently hands the Proposer
#: the *ancestor's* graph instead, which is worse than handing it nothing.
_GENERATED_DOCS = frozenset({"architecture.md"})


def _ignore_forked(source_root: Path) -> Callable[[str, list[str]], set[str]]:
    """What a forked candidate must not inherit from the directory it came from.

    Returns a callable bound to *source_root* because one of the three owners applies at
    the top level only: ``architecture.md`` is generated *about* the agent and sits beside
    it, so an agent whose own source contains ``docs/architecture.md`` must keep it. The
    others — run layout, hygiene, evaluator scratch — are stripped at every depth.
    """
    root = source_root.resolve()

    def _ignore(directory: str, contents: list[str]) -> set[str]:
        skip = RUN_LAYOUT | _HYGIENE
        if Path(directory).resolve() == root:
            skip = skip | _GENERATED_DOCS
        return {
            name
            for name in contents
            if name in skip or any(fnmatch.fnmatch(name, pattern) for pattern in _EVALUATOR_SCRATCH_GLOBS)
        }

    return _ignore


class ExperimentContext:
    """One run's view of the platform, handed to exactly one strategy.

    Args:
        backend: Data-access backend. Held privately; never handed to a component.
        workspace: NeMo Helix workspace this run belongs to.
        run: The ``ExperimentRun`` entity the runner created (or re-opened on resume).
        root: Working directory for the run's artifacts.
        agent_dir: The agent under test, materialized by the runner. A strategy forks
            candidates from here; it must not write into it.
        ethos: Optional markdown description of the agent under test.
        datasets: Evaluator-domain datasets keyed by split. ``validation`` is always
            present; ``train`` and ``insight`` are present when the run has them.
        evaluator: OutcomeEvaluator component the run was configured with.
        resuming: True when the runner re-opened an existing run, so the strategy
            should rebuild its state from :meth:`candidates` instead of starting over.
        reporter: Optional human narration sink. Best-effort and never load-bearing.
    """

    def __init__(
        self,
        *,
        backend: ExperimentalistBackend,
        workspace: str,
        run: ExperimentRun,
        root: Path,
        agent_dir: Path,
        ethos: Path | None = None,
        datasets: Mapping[str, Dataset],
        evaluator: Evaluator,
        resuming: bool = False,
        reporter: RunReporter | None = None,
        objective_metrics: list[MetricTarget] | None = None,
        regression_metrics: list[MetricTarget] | None = None,
    ) -> None:
        if PRIMARY_SPLIT not in datasets:
            raise ValueError(f"ExperimentContext requires a {PRIMARY_SPLIT!r} dataset; got {sorted(datasets)}")
        self._backend = backend
        self._run = run
        self._evaluator = evaluator
        self._reporter = reporter
        self._objective_metrics = objective_metrics or [MetricTarget(name="reward", direction="maximize")]
        self._regression_metrics = regression_metrics or []
        self.workspace = workspace
        self.root = root
        self.agent_dir = agent_dir
        self.ethos = ethos
        self.datasets: Mapping[str, Dataset] = dict(datasets)
        self.resuming = resuming

    # -- Inputs --------------------------------------------------------------

    @property
    def run_id(self) -> str:
        """Durable id of this run, and the key every Candidate is grouped under."""
        return self._run.id or ""

    @property
    def outcome_evaluator(self) -> Evaluator:
        """The evaluation component this run was configured with.

        Exposed so a composite strategy can hand it to a component it owns — the CodeEditBuilder
        runs smoke evals of its own, against work that is not yet a Candidate.
        """
        return self._evaluator

    async def load_trace(self, ref: ResourceRef) -> TraceExplorer:
        """The trace *ref* names, wherever it is stored.

        A ``file://`` ref is read from disk and an ``intake://`` one from the platform.
        Which of those a run produces is the host's business, so this is the seam: a
        component that reads traces asks for ``load_trace`` and never names a platform
        type, which is what lets one shipped by another package read traces at all.

        Raises:
            ValueError: if the reference names a scheme this run cannot resolve.
        """
        return await TraceExplorer.from_ref(ref, self._backend.client, self.workspace)

    # -- Candidates ----------------------------------------------------------

    async def candidates(self, *, include_discarded: bool = False) -> list[Candidate]:
        """The Candidates committed to this run, in store order.

        Persisted rather than checkpointed, which is what lets a strategy the host did
        not write rebuild its population on resume.
        """
        return await self._backend.list_candidates(
            workspace=self.workspace, run_id=self.run_id, include_discarded=include_discarded
        )

    def candidate_dir(self, candidate: Candidate) -> Path:
        """Local directory holding *candidate*'s artifact.

        A record whose artifact has gone is broken, not a candidate whose directory can
        be guessed. Falling back to the parent resolves it to the shared candidate root,
        which callers then copy out of and push wholesale.

        Raises:
            ValueError: if the artifact does not exist.
        """
        path = local_path_from_uri(candidate.artifact.uri, context="Candidate artifact")
        if not path.exists():
            raise ValueError(
                f"Candidate {candidate.label!r} ({candidate.id}) has no artifact at {path}; "
                "its record outlived the resource it addresses"
            )
        return path if path.is_dir() else path.parent

    async def fork(self, proposal: Proposal) -> Fork:
        """Reserve a working copy for *proposal*, seeded from what it branches off.

        The ignore policy is host knowledge composed from three owners — this run's
        layout, generic hygiene, and the evaluator's scratch — so it lives here rather
        than in a Builder.

        Returns:
            Fork: the reserved directory to write into, and the pristine parent it came
            from. Neither is a Candidate until :meth:`commit_candidate` validates the
            finished work and commits it.
        """
        upstream = await self._ancestor_dir(proposal) if proposal.ancestor is not None else None
        # No "already forked?" guard: _reserve returns max(existing) + 1, so the name is
        # always fresh. A stray *file* by that name would make copytree raise, which is
        # the right outcome — skipping the copy would hand the Builder a Fork whose
        # workdir is a file.
        destination = self._reserve()
        source = upstream or self.agent_dir
        shutil.copytree(source, destination, ignore=_ignore_forked(source))
        return Fork(workdir=destination, upstream=upstream)

    async def commit_candidate(self, *, proposal: Proposal, artifact: Path, generation: int = 0) -> Candidate:
        """Validate a finished artifact and create the Candidate that addresses it.

        This is the only way a Candidate is born from a build, which is what makes
        ``artifact`` safe to require: nothing durable ever points at partial work.

        ``ancestor`` and ``description`` are always derived from the Proposal — there is
        no override, so the two accounts of a candidate's origin cannot drift.

        Raises:
            ValueError: if the artifact is missing or lies outside the run's candidate root.
        """
        return await self._create(
            artifact=artifact,
            proposal=proposal,
            description=proposal.description,
            generation=generation,
        )

    async def _create(self, *, artifact: Path, proposal: Proposal, description: str, generation: int) -> Candidate:
        """Validate a finished artifact and persist the Candidate addressing it.

        Private, so a caller cannot assemble a Candidate and persist it around the
        validation that makes ``artifact`` safe to require.
        """
        if not artifact.exists():
            raise ValueError(f"Candidate artifact does not exist: {artifact}")
        artifact = artifact.resolve()
        root = self._candidate_root.resolve()
        if not artifact.is_relative_to(root):
            raise ValueError(f"Candidate artifact must live under {root}, got {artifact}")

        handle = (artifact.parent if artifact.is_file() else artifact).name
        candidate = Candidate(
            name=handle,
            label=handle,
            workspace=self.workspace,
            run_id=self.run_id,
            ancestor=proposal.ancestor,
            generation=generation,
            generated_from=proposal,
            description=description,
            artifact=ResourceRef(uri=artifact.as_uri(), description=description),
        )
        stored = await self._backend.create_candidate(workspace=self.workspace, candidate=candidate)
        candidate._id = stored._id  # type: ignore[attr-defined]
        return candidate

    @property
    def _candidate_root(self) -> Path:
        """Runner-owned root every candidate artifact must live under."""
        return self.root / "eval-and-optimize" / "agents"

    def _reserve(self) -> Path:
        """Pick the next free candidate directory under the run's candidate root.

        Every fork gets a fresh directory, including one built from the agent under test
        rather than from a parent: a run may hold many of those, and sharing a directory
        would have them overwrite each other while still reporting separate rewards. The
        baseline lands on ``agent-0`` simply by being the first.
        """
        root = self._candidate_root
        root.mkdir(parents=True, exist_ok=True)
        taken = [
            int(entry.name.split("-")[1])
            for entry in root.iterdir()
            if entry.is_dir() and entry.name.startswith("agent-") and entry.name.split("-")[1].isdigit()
        ]
        return root / f"agent-{max(taken, default=-1) + 1}"

    async def _ancestor_dir(self, proposal: Proposal) -> Path:
        """Where the candidate *proposal* branches from keeps its artifact.

        Resolved through the ancestor's stored ``artifact``, not by treating the id as
        a directory name — that identification is exactly what this contract removes.
        """
        assert proposal.ancestor is not None
        ancestor = await self._backend.get_candidate(workspace=self.workspace, candidate_id=proposal.ancestor)
        return self.candidate_dir(ancestor)

    async def update_candidate(self, candidate: Candidate, **fields: Any) -> Candidate:
        """Persist changes to a candidate that already exists.

        Update-only: the create path is private, so nothing can persist a Candidate that
        never went through :meth:`commit_candidate`.

        Raises:
            ValueError: if *candidate* was never committed, or is a lossy :meth:`~.slim`
                copy whose emptied trials would be written back over the real ones. The
                second check is best-effort: the marker does not survive serialization.
        """
        if not candidate.id:
            raise ValueError(
                f"Candidate {candidate.label!r} has no store id; a Candidate is created by "
                "commit_candidate, never by updating one into existence"
            )
        if candidate._slim:
            raise ValueError(
                f"Candidate {candidate.label!r} is a slim() copy; persisting one empties every "
                "channel's trials in the store. Look the real candidate up by id first."
            )
        for key, value in fields.items():
            setattr(candidate, key, value)
        return await self._backend.update_candidate(workspace=self.workspace, candidate=candidate)

    async def discard_candidate(self, candidate: Candidate) -> None:
        """Mark *candidate* as rolled back, keeping both its record and its artifact.

        Both halves survive together: deleting the directory while keeping the record
        would let :meth:`_reserve` hand out a label the record still claims, giving one
        run two candidates with the same handle. :meth:`candidates` excludes them.
        """
        await self.update_candidate(candidate, discarded=True)

    async def archive_candidate(self, candidate: Candidate) -> None:
        """Persist *candidate*'s code to durable storage, if the run archives at all.

        Best-effort in both directions: a backend that cannot archive returns nothing,
        and a failure is logged rather than raised — archival must never fail a run.
        """
        if not self._backend.storage.archive_candidates:
            return
        try:
            await self._backend.archive_candidate(workspace=self.workspace, candidate=candidate)
        except Exception as exc:  # noqa: BLE001 - archival must never fail the run
            logger.warning("[PERSISTENCE] archive failed for candidate %s; continuing: %s", candidate.label, exc)

    # -- Measurement ---------------------------------------------------------

    async def record_reward(
        self,
        candidate: Candidate,
        *,
        channel: str,
        result: EvaluationResult | RewardRecord,
        metadata: dict[str, DataValue] | None = None,
    ) -> None:
        """Store one measurement of *candidate* on *channel*, and persist the candidate.

        An ``EvaluationResult`` is the outcome of running the candidate, so its traces
        are persisted before the record is stored; a ``RewardRecord`` is a measurement
        the strategy computed itself (trajectory scoring, a self-scoring strategy) and
        is stored as given. Either way the channel is an open key — adding one costs no
        entity change.
        """
        if isinstance(result, EvaluationResult):
            await self._backend.persist_evaluation(
                workspace=self.workspace,
                result=result,
                candidate=candidate,
                split=channel,
            )
            record = RewardRecord(
                metrics={k: float(v) for k, v in result.aggregate_metrics.items()},
                trials=list(result.trials),
                metadata=dict(metadata or {}),
            )
        else:
            record = result if metadata is None else result.model_copy(update={"metadata": dict(metadata)})
        # The one sanctioned write: RewardMap refuses __setitem__ so an in-memory-only
        # measurement cannot masquerade as a recorded one. Persisting is the next line.
        dict.__setitem__(candidate.rewards, channel, record)
        await self.update_candidate(candidate)

    async def evaluate(
        self,
        candidate: Candidate,
        *,
        split: str = PRIMARY_SPLIT,
        task_ids: Sequence[str] | None = None,
        minimum_attempts: int | None = None,
    ) -> EvaluationResult:
        """Run the configured evaluation component over *candidate*'s artifact.

        Optional by design: a strategy that scores itself skips this and calls
        :meth:`record_reward` directly. The association between the result and the
        candidate is owned here, not by the evaluation component.

        Args:
            candidate: Whose artifact to evaluate.
            split: Which of :attr:`datasets` to run against.
            task_ids: Restrict the run to these task ids.
            minimum_attempts: Raise the evaluator's attempt count to at least this.

        Raises:
            KeyError: if *split* is not one of the run's datasets.
        """
        dataset = self.datasets[split]
        if task_ids is not None:
            dataset = dataset.subset(list(task_ids))
        # Force a unique job name per candidate so concurrent candidates never collide on
        # one results directory when a fixed job_name is configured. job_name/n_attempts
        # are Harbor's vocabulary reaching a generic call site — an evaluator leak the
        # registered evaluation component closes.
        options = self._evaluator.options.model_dump()
        options["job_name"] = f"{candidate.label}-{dataset.id}"
        if minimum_attempts is not None:
            configured = options.get("n_attempts")
            if not isinstance(configured, int):
                raise ValueError("Evaluator options must define integer n_attempts to raise the attempt floor")
            options["n_attempts"] = max(configured, minimum_attempts)
        result = await self._evaluator.run(
            agent=self.candidate_dir(candidate),
            dataset=dataset,
            options=type(self._evaluator.options).model_validate(options),
        )
        if self._reporter is not None:
            self._reporter.candidate_evaluated(
                label=candidate.label,
                split=split,
                metrics=result.aggregate_metrics,
                objective_metrics=self._objective_metrics,
                artifacts=self.root / "eval-and-optimize" / "results" / result.id,
                reason=missing_objective_reason(result.trials, result.aggregate_metrics, self._objective_metrics),
            )
        return result

    # -- Progress ------------------------------------------------------------

    async def report_progress(
        self,
        *,
        completed: int,
        total: int | None = None,
        unit: str = "step",
        note: str | None = None,
    ) -> None:
        """Report how far the strategy has got, as a counter rather than a fraction.

        A counter is always producible; a fraction usually is not — an opaque strategy
        cannot say how many trials it will run, and even a round-based one stops early
        on convergence. When *total* is known a consumer may render a bar; when it is
        not, ``note`` is where a strategy says what it is doing instead.
        """
        self._run.progress_completed = completed
        self._run.progress_total = total
        self._run.progress_unit = unit
        self._run.progress_note = note
        await self._backend.update_run(workspace=self.workspace, run=self._run)
        if self._reporter is not None:
            self._reporter.progress(phase=note or unit, completed=completed, total=total, unit=unit)

    def component(self, role: str, name: str, **kwargs: Any) -> Any:
        """Resolve a component by name and build it against this run.

        Supplies the run-scoped arguments a component cannot know for itself — the model
        tiers, the run root, the evaluator, the primary dataset — to whichever of them the
        constructor names, and passes *kwargs* through untouched. Explicit *kwargs* win.

        Raises:
            LookupError: if nothing is registered under that name.
        """
        component = resolve(role, name)
        # Two spellings of the same run root; both are offered so a component keeps
        # whichever it names.
        supplied: dict[str, Any] = {
            "workspace": self.root,
            "working_dir": self.root,
            "evaluator": self._evaluator,
            # A trace reader takes this rather than a platform client, so no component
            # signature names a platform type and storage can move without touching one.
            "load_trace": self.load_trace,
            # Train, not the primary split: a Builder runs bounded repair loops against
            # whatever it is handed, and `validation` is held out for scoring.
            "dataset": self.datasets.get("train", self.datasets[PRIMARY_SPLIT]),
            **kwargs,
        }
        return component(**validated_config(component, _accepted(component.__init__, supplied)))

    @property
    def objective_metrics(self) -> list[MetricTarget]:
        """The run's effective objectives, after any Insight suite narrowed them."""
        return list(self._objective_metrics)

    @property
    def regression_metrics(self) -> list[MetricTarget]:
        """The run's effective guardrail metrics, alongside :attr:`objective_metrics`."""
        return list(self._regression_metrics)

    def note(self, message: str) -> None:
        """Say what is happening, for a human watching the run.

        Narration only: it touches no entity and never raises, so a strategy may call it
        as freely as it likes. Use :meth:`report_progress` for anything a consumer
        should be able to read back off the run.
        """
        if self._reporter is not None:
            self._reporter.note(message)
