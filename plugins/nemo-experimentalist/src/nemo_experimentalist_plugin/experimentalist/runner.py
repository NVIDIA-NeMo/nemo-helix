# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The experiment runner — the composition root.

It does three things: **prepare inputs → run one strategy → persist and publish.**
Inputs are an agent directory, an optional ``Insight``, and the eval datasets; the
strategy is whatever the run was configured with; persisting and publishing are the
host's, not the strategy's.

The runner is the only code that holds an
:class:`~nemo_experimentalist_plugin.experimentalist.experimentalist_backend.ExperimentalistBackend`.
Everything a strategy is allowed to reach goes through the
:class:`~nemo_experimentalist_plugin.experimentalist.context.ExperimentContext` it builds.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from nemo_experimentalist_plugin.config import EvolutionaryOptimizerConfig, with_insight_objective
from nemo_experimentalist_plugin.entities import (
    Candidate,
    Dataset,
    DatasetRef,
    ExperimentRun,
)
from nemo_experimentalist_plugin.experimentalist.components.dataset_staging import (
    distribute_insight_suite_tasks,
    stage_eval_author_inputs,
)
from nemo_experimentalist_plugin.experimentalist.components.evaluator.factory import (
    DatasetFactory,
    EvaluatorFactory,
)
from nemo_experimentalist_plugin.experimentalist.components.holdout_utils import (
    ensure_heldout_hidden,
    restore_heldout_splits,
)
from nemo_experimentalist_plugin.experimentalist.context import RUN_LAYOUT, ExperimentContext
from nemo_experimentalist_plugin.experimentalist.experimentalist_backend import (
    ExperimentalistBackend,
)
from nemo_experimentalist_plugin.experimentalist.reporting import RunReporter
from nemo_experimentalist_plugin.experimentalist.result import ExperimentalistResult
from nemo_experimentalist_plugin.experimentalist.roles import Strategy

logger = logging.getLogger(__name__)

#: Names the winner's artifact must not carry into the user's workspace, named by the
#: owner that generates them rather than as one flat list — a third-party strategy's
#: real output has to survive this, and it cannot if the reason for each entry is lost.
_STRATEGY_ARTIFACTS = frozenset({"architecture.md"})
_EVALUATOR_ARTIFACTS = frozenset({"harbor_wrapper.py", "dind_environment.py"})


@dataclass(frozen=True)
class PreparedInputs:
    """Everything the strategy runs against, after the runner has materialized it."""

    agent_dir: Path
    agent_name: str
    ethos: Path | None
    insight_ref: str | None
    datasets: dict[str, Dataset]


class ExperimentRunner:
    """Wires one run up and calls it in order.

    Args:
        backend: The run's data-access backend. Not shared with the strategy.
        strategy: The optimization strategy to run.
        config: Resolved run configuration.
        workspace: NeMo Platform workspace.
        root: Working directory for run artifacts.
        agent: Baseline agent — a local directory, a git ``url@ref``, or None to take
            the agent the Insight names.
        ethos: Optional URI of a markdown description of the agent under test.
        insight: Optional Insight id or local Insight file.
        train_dataset: Dataset reference the strategy develops against.
        validation_dataset: Dataset reference the winner is selected on.
        task_template: Evaluator-specific task template; required with an Insight.
        reporter: Optional human narration sink.
    """

    def __init__(
        self,
        *,
        backend: ExperimentalistBackend,
        strategy: Strategy,
        config: EvolutionaryOptimizerConfig,
        workspace: str,
        root: Path,
        agent: str | Path | None,
        ethos: str | None = None,
        insight: str | Path | None = None,
        train_dataset: DatasetRef,
        validation_dataset: DatasetRef,
        task_template: DatasetRef | None = None,
        reporter: RunReporter | None = None,
    ) -> None:
        if agent is None and insight is None:
            raise ValueError("One of 'insight' or 'agent' must be set.")
        if insight is not None and task_template is None:
            raise ValueError("'task_template' is required when 'insight' is set.")
        self._backend = backend
        self._strategy = strategy
        self._config = config
        self._workspace = workspace
        self._root = root.resolve()
        self._agent = agent
        self._ethos = ethos
        self._insight = insight
        self._train_dataset = train_dataset
        self._validation_dataset = validation_dataset
        self._task_template = task_template
        self._reporter = reporter
        # One object for the whole run, so every component runs on the tiers the run
        # record reports and not on whatever the environment says at each construction.
        self._eo = self._root / "eval-and-optimize"

    async def run(self) -> ExperimentalistResult:
        """Prepare inputs, run the strategy, then persist and publish the outcome."""
        self._preflight()
        for subdir in ("agents", "analysis", "results"):
            (self._eo / subdir).mkdir(parents=True, exist_ok=True)
        evaluator = EvaluatorFactory().build_evaluator(
            self._config.outcome_evaluator, self._config.outcome_evaluator_config, experiment_dir=self._root
        )
        inputs = await self._prepare_inputs()
        run, resuming = await self._open_run(inputs)
        ctx = ExperimentContext(
            backend=self._backend,
            workspace=self._workspace,
            run=run,
            root=self._root,
            agent_dir=inputs.agent_dir,
            ethos=inputs.ethos,
            datasets=inputs.datasets,
            evaluator=evaluator,
            resuming=resuming,
            reporter=self._reporter,
            objective_metrics=self._config.objective_function,
            regression_metrics=self._config.regression_metrics,
        )
        if resuming:
            await self._seed_narration_baseline(ctx)

        # Finalizing is inside the handler, not after it. It resolves the winner's
        # artifact, copies it out and persists the result — any of which can fail — and
        # if it did, the run entity stayed `running` forever with no result written,
        # while the work had actually finished.
        try:
            winner = await self._strategy.run(ctx)
            return await self._finalize(ctx, run, inputs, winner)
        except Exception:
            run.status = "failed"
            await self._backend.update_run(workspace=self._workspace, run=run)
            raise

    # -- Prepare -------------------------------------------------------------

    def _preflight(self) -> None:
        """Fail fast when persistence is enabled but the tool it needs is missing."""
        storage = self._config.storage
        if (storage.archive_candidates or storage.publish_winner) and shutil.which("git") is None:
            raise ValueError(
                "Candidate persistence is enabled (storage.archive_candidates/publish_winner) "
                "but 'git' is not on PATH, so nothing can be persisted. Install git, or disable "
                "storage to run without persistence."
            )

    async def _prepare_inputs(self) -> PreparedInputs:
        """Materialize the agent, the datasets, and — with an Insight — its eval suite."""
        dataset_factory = DatasetFactory()
        train_ref, validation_ref, template_ref = (
            self._train_dataset,
            self._validation_dataset,
            self._task_template,
        )

        insight = None
        if self._insight is not None:
            insight = await self._backend.get_insight(workspace=self._workspace, insight_id=str(self._insight))
            assert template_ref is not None
            staged = await stage_eval_author_inputs(
                self._root,
                train_dataset=train_ref,
                validation_dataset=validation_ref,
                task_template=template_ref,
                client=self._backend.client,
                workspace=self._workspace,
            )
            train_ref, validation_ref, template_ref = (
                staged.train_dataset,
                staged.validation_dataset,
                staged.task_template,
            )

        datasets: dict[str, Dataset] = {
            # allow_empty in insight mode: the Eval Author fills these splits, so they are
            # legitimately empty when the run starts.
            "train": dataset_factory.build_dataset(
                self._config.outcome_evaluator, train_ref, allow_empty=insight is not None
            ),
            "validation": dataset_factory.build_dataset(
                self._config.outcome_evaluator, validation_ref, allow_empty=insight is not None
            ),
        }

        agent_ref = self._agent if self._agent is not None else (insight.agent if insight is not None else None)
        if agent_ref is None:
            raise ValueError("Insight or agent is required")
        agent_dir = self._eo / "source-agent"
        await self._backend.get_agent_code(
            workspace=self._workspace,
            agent=agent_ref,
            dest=agent_dir,
            clone_depth=self._config.source.clone_depth,
        )

        ethos: Path | None = None
        if self._ethos is not None:
            ethos = await self._backend.get_ethos(
                workspace=self._workspace,
                ethos=self._ethos,
                dest=self._root / "ETHOS.md",
            )

        if insight is not None:
            assert template_ref is not None
            # Lazy: a run without an Insight never authors an eval suite, so it must not
            # fail to import when the Eval Author package is absent.
            from nemo_experimentalist_plugin.eval_author.agent import EvalAuthor  # noqa: PLC0415

            authored = await EvalAuthor(
                experiment_dir=self._root, config=self._config.eval_author, reporter=self._reporter
            ).run(
                insight=insight,
                agent_path=agent_dir,
                task_template=dataset_factory.build_task_template(self._config.outcome_evaluator, template_ref),
                train_dataset=datasets["train"],
                validation_dataset=datasets["validation"],
                client=self._backend.client,
            )
            datasets["train"] = authored.train_dataset
            datasets["validation"] = authored.validation_dataset

            # The authored verifiers emit their own metric keys, so they become the
            # run's objective and the configured targets demote to guardrails. Without
            # this the run keeps optimizing `reward`, which those verifiers never emit,
            # and no candidate is eligible to win.
            self._config = with_insight_objective(self._config, authored.metric_keys)

            # The suite is production-trace evidence: it belongs in the splits the loop
            # actually evaluates, not parked beside them. Held-out splits are restored
            # for the write and re-hidden afterwards, so the distribution can see them.
            if authored.insight_suite is not None:
                restore_heldout_splits(self._root)
                try:
                    distribute_insight_suite_tasks(authored.insight_suite, datasets["train"], datasets["validation"])
                finally:
                    ensure_heldout_hidden(self._root)

        return PreparedInputs(
            agent_dir=agent_dir,
            agent_name=str(agent_ref),
            ethos=ethos,
            insight_ref=str(self._insight) if self._insight is not None else None,
            datasets=datasets,
        )

    async def _open_run(self, inputs: PreparedInputs) -> tuple[ExperimentRun, bool]:
        """Re-open this run if one already exists here, else create it.

        Resume is strategy-independent at this level: the runner restores the
        ``ExperimentRun`` and its inputs, and the strategy rebuilds its own state from
        ``ctx.candidates()``. A strategy that cannot do that is refused loudly rather
        than silently restarted — these runs cost hours, so the silent restart is the
        expensive failure.
        """
        existing = self._load_run()
        if existing is None and await self._has_candidate_records():
            raise ValueError(
                f"{self._eo / 'run.json'} is missing or unreadable, but {self._eo / 'candidates'} still "
                "holds candidate records. Starting fresh would mint a new run id, and every candidate "
                "already built here would become invisible to it — hours of work still on disk and "
                "unreachable. Restore run.json, or move the directory aside to start over."
            )
        if existing is not None:
            if existing.status == "completed":
                # Resume exists for a run that was interrupted. A completed one has its
                # winner chosen, its report written and possibly a PR opened; re-opening
                # marks it running again and lets the strategy rewrite all of it. Refusing
                # costs a fresh directory and keeps a finished result finished.
                raise ValueError(
                    f"Run {existing.id!r} in {self._eo / 'run.json'} already completed. Point "
                    "--experiment-dir at a fresh directory to start a new run, or delete that "
                    "run to redo it."
                )
            if not self._strategy.supports_resume:
                raise ValueError(
                    f"{type(self._strategy).__name__} does not support resume, but "
                    f"{self._eo / 'run.json'} already holds run {existing.id!r}. Point --experiment-dir "
                    "at a fresh directory, or delete that run to start over."
                )
            logger.info("[RESUME] re-opening run %s", existing.id)
            existing.status = "running"
            await self._backend.update_run(workspace=self._workspace, run=existing)
            return existing, True

        run = ExperimentRun(
            workspace=self._workspace,
            agent=inputs.agent_name,
            insight=inputs.insight_ref,
            config_snapshot={
                **self._config.model_dump(mode="json"),
            },
            status="running",
        )
        return await self._backend.create_run(workspace=self._workspace, run=run), False

    async def _seed_narration_baseline(self, ctx: ExperimentContext) -> None:
        """Point the reporter's delta at the baseline it will not see re-evaluated.

        On resume the baseline is not scored again, so without this the first newly
        evaluated candidate becomes the delta reference and every later delta is
        measured against the wrong thing. Narration only; never fails a run.
        """
        if self._reporter is None:
            return
        try:
            baseline = next((c for c in await ctx.candidates() if c.ancestor is None), None)
        except Exception as exc:  # noqa: BLE001 - narration must never fail the run
            logger.debug("[RESUME] could not seed the narration baseline: %s", exc)
            return
        if baseline is not None and baseline.rewards["validation"].metrics:
            self._reporter.seed_baseline(baseline.rewards["validation"].metrics or {})

    async def _has_candidate_records(self) -> bool:
        """Whether this directory already holds committed candidates.

        Read off the filesystem rather than through the backend: the backend lists by
        run id, and the run id is exactly what is unavailable when this is asked.
        """
        candidates_dir = self._eo / "candidates"
        return candidates_dir.is_dir() and any(candidates_dir.glob("*.json"))

    def _load_run(self) -> ExperimentRun | None:
        """Read this directory's ``run.json``, or None when it is absent or unreadable."""
        run_path = self._eo / "run.json"
        if not run_path.exists():
            return None
        try:
            # ExperimentRun's wrap validator restores the computed id.
            return ExperimentRun.model_validate_json(run_path.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.warning("[RESUME] Could not parse run.json: %s", exc)
            return None

    # -- Persist and publish -------------------------------------------------

    async def _finalize(
        self,
        ctx: ExperimentContext,
        run: ExperimentRun,
        inputs: PreparedInputs,
        winner: Candidate | None,
    ) -> ExperimentalistResult:
        """Close the run out: report, workspace copy, terminal result, draft PR."""
        restore_heldout_splits(self._root)
        report_path = self._eo / "OPTIMIZATION.md"
        candidates = await ctx.candidates()
        baseline = next((c for c in candidates if c.ancestor is None), None)
        summary = _render_summary(run.progress_completed, run.progress_unit, baseline, winner)

        # The strategy writes the real report. When it produced nothing usable, leave a
        # compact one behind so the Insight sections below have a document to append to.
        if not report_path.exists() or not report_path.read_text().strip():
            report_path.write_text(f"# Optimization Report\n\n## Compact Run Summary\n\n{summary}\n")

        if winner is not None:
            self._copy_winner_to_workspace(ctx.candidate_dir(winner))

        result = ExperimentalistResult(
            summary=summary,
            run_id=run.id or "",
            progress_completed=run.progress_completed,
            winner=winner,
        )
        run.status = "completed"
        run.winner_agent = winner.label if winner is not None else None
        await self._backend.update_run(workspace=self._workspace, run=run)
        await self._backend.persist_result(workspace=self._workspace, result=result)

        if self._config.storage.publish_winner and winner is not None and winner.ancestor is not None:
            try:
                url = await self._backend.publish_candidate(workspace=self._workspace, candidate=winner)
                if url:
                    logger.info("[PUBLISH] opened draft PR/MR for winner %s: %s", winner.label, url)
            except Exception as exc:  # noqa: BLE001 - publishing must never fail the run
                logger.warning("[PERSISTENCE] publish failed for candidate %s; continuing: %s", winner.label, exc)
        return result

    def _copy_winner_to_workspace(self, winner_dir: Path) -> None:
        """Copy the winner's artifact to the workspace root, minus every owner's scaffolding."""
        # The run's own layout is never copied over. `fork` already strips these on the way
        # in, but `commit_candidate` accepts any artifact under the candidate root, so a
        # strategy that builds without forking -- the documented extension point -- can
        # commit one containing `eval-and-optimize`, and the rmtree below would delete the
        # run it is finalizing.
        skip = _STRATEGY_ARTIFACTS | _EVALUATOR_ARTIFACTS | RUN_LAYOUT
        if not winner_dir.is_dir():
            logger.warning("[FINAL] winner artifact %s is not a directory; skipping workspace copy", winner_dir)
            return
        for entry in winner_dir.iterdir():
            if entry.name in skip:
                continue
            dst = self._root / entry.name
            if entry.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(entry, dst)
            else:
                shutil.copy2(entry, dst)


def _render_summary(completed: int, unit: str, baseline: Candidate | None, winner: Candidate | None) -> str:
    """One-line outcome summary, used when the strategy's own report is missing."""
    details: list[str] = []
    if winner is not None:
        if winner.rewards["validation"].metrics:
            details.append(f"validation_reward={winner.rewards['validation'].metrics}")
    suffix = f", {', '.join(details)}" if details else ""
    winner_str = winner.label if winner is not None else "none"
    return f"Optimization complete: {completed} {unit}(s) completed, winner={winner_str}{suffix}"


__all__ = ["ExperimentRunner", "PreparedInputs", "Strategy"]
