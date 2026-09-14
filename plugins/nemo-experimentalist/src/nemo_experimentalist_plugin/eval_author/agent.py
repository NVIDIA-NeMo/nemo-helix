# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical first-level Eval Author agent.

Experimentalist consumes this implementation to curate evaluator datasets
before beginning insight-driven optimization.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from nemo_experimentalist_plugin.entities import (
    Dataset,
    DatasetValidationError,
    ResourceRef,
    Task,
    TrialResult,
)
from nemo_experimentalist_plugin.eval_author import traces
from nemo_experimentalist_plugin.eval_author.materialization import InsightSuite, validate_metric_contracts
from nemo_experimentalist_plugin.eval_author.models import EvalAuthorConfig, EvalAuthorResult, MetricAuthoringResult
from nemo_experimentalist_plugin.experimentalist.components.tools import GuardedShellTools
from nemo_experimentalist_plugin.experimentalist.components.trace_analyzer import (
    Diagnostic,
    TraceAnalyzer,
    TraceAnalyzerConfig,
)
from nemo_experimentalist_plugin.experimentalist.components.trace_explorer import (
    SearchResult,
    SessionData,
    SessionSummary,
    TraceExplorer,
    TurnInfo,
)
from nemo_experimentalist_plugin.experimentalist.reporting import RunReporter
from nemo_insights_plugin.entities import Insight
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.nooa_model_client import get_default_model, get_fast_model
from nooa import Agent, CodeActStrategy, strategy
from nooa.agentdoc import doc
from nooa.agents import TokenBudgetSummarizer
from nooa.config import CodeActConfig
from nooa.config.summarizer_config import TokenBudgetConfig
from nooa.tools import TodoManager

logger = logging.getLogger(__name__)


class EvalAuthor(Agent):
    """Insights are failure modes of an agent in production.

    The role of the Eval Author is to create or augment the evaluation suite in such a way that it can be used to detect the failure mode.
    This suite will be used for optimization and regression testing.
    """

    def __init__(
        self,
        experiment_dir: Path,
        config: EvalAuthorConfig | None = None,
        reporter: RunReporter | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Eval Author for the given experiment directory.

        Args:
            experiment_dir: Absolute path to the experiment root.
            config: Tuning parameters; defaults to ``EvalAuthorConfig()``.
            reporter: Optional parent run narrator (Experimentalist insight mode).
                When set, emits mid-run progress lines; never owns header/footer.
            **kwargs: Forwarded to ``Agent.__init__``.
        """
        super().__init__(llm=kwargs.pop("llm", None) or get_default_model(), **kwargs)
        self._config = config or EvalAuthorConfig()
        self._reporter = reporter
        self.experiment_dir = experiment_dir
        # Set by the standalone entry point before it queries production traces. The
        # Experimentalist path leaves both unset and passes a client to ``run`` instead.
        self.client: AsyncNemoClient | None = None
        self.workspace: str | None = None
        self.shell = GuardedShellTools(cwd=experiment_dir)
        self.todos = TodoManager()
        self.context["trace_documentation"] = doc(
            TraceExplorer,
            SessionSummary,
            TurnInfo,
            SessionData,
            SearchResult,
            inline_depth=2,
        )
        TokenBudgetSummarizer.install(
            self,
            llm=get_fast_model(),
            config=TokenBudgetConfig(max_tokens=self._config.max_summary_tokens),
        )

    def _intake(self) -> tuple[AsyncNemoClient, str]:
        """Return the client and workspace for Intake reads, or name what is missing."""
        if self.client is None or self.workspace is None:
            raise traces.TraceQueryError(
                "Production trace tools need a platform client and a workspace. Set "
                "EvalAuthor.client and EvalAuthor.workspace before you call them."
            )
        return self.client, self.workspace

    async def query_spans(
        self,
        filter: dict[str, Any] | None = None,
        group_by: str | None = None,
        sort: str | None = None,
        mode: str = "summary",
        limit: int = traces.DEFAULT_ROW_LIMIT,
    ) -> dict[str, Any]:
        """Query production spans from Intake, flat or rolled up into groups.

        Use this to find which traces are worth opening, not to read what happened in
        them. A span is one step of a trajectory, so on its own it hides what led to it
        and what followed: a slow call may be waiting on the retry before it, and an
        error may be one the next span handles. Every row carries ``trace_ref``, so open
        the trace with ``read_trace`` before you quote a span, treat it as evidence, or
        turn it into a test case.

        Write your own filter. The server does the narrowing, which is exact, so
        prefer one precise filter over counting a wide result yourself. Group by
        ``trace_id`` to turn any filter into the set of traces that match it, newest
        first, then pass those ids to ``query_traces``. Grouped rows also sort by
        ``-span_count`` when you want the busiest traces instead of the newest.

        See ``nemo_experimentalist_plugin.eval_author.traces.query_spans`` for the filter vocabulary.
        """
        client, workspace = self._intake()
        return await traces.query_spans(
            client,
            workspace=workspace,
            filter=filter,
            group_by=group_by,
            sort=sort,
            mode=mode,
            limit=limit,
        )

    async def query_traces(
        self,
        filter: dict[str, Any] | None = None,
        sort: str | None = None,
        mode: str = "preview",
        limit: int = traces.DEFAULT_ROW_LIMIT,
    ) -> dict[str, Any]:
        """Query whole traces, with the rollups that the server computes.

        A trace row carries the exact ``span_count`` and ``error_count`` of the whole
        trace, which a capped span query cannot give you. There is no ``agent_name``
        filter here, because only spans carry it; use ``find_agent_traces`` instead.

        See ``nemo_experimentalist_plugin.eval_author.traces.query_traces`` for the filter vocabulary.
        """
        client, workspace = self._intake()
        return await traces.query_traces(client, workspace=workspace, filter=filter, sort=sort, mode=mode, limit=limit)

    async def find_agent_traces(
        self,
        agent: str,
        since: datetime | None = None,
        limit: int = traces.DEFAULT_TRACE_LIMIT,
    ) -> dict[str, Any]:
        """Find the most recent production traces of one agent, newest first.

        Start here when you need real traces and hold no trace refs. No single
        endpoint answers this, so it groups spans by trace to find the recent ones
        and then reads their summaries. Open the ones worth reading with
        ``read_trace``.

        Args:
            agent: Value that the agent reports to Intake as ``agent_name``.
            since: Optional lower bound on span start time.
            limit: Maximum number of traces to return.

        Returns:
            ``{"traces": [...], "count": int, "truncated": bool}``. Each trace carries
            ``trace_ref``, ``trace_id``, ``started_at``, ``status``, ``span_count``,
            ``error_count``, ``duration_ms``, and ``name``.
        """
        client, workspace = self._intake()
        return await traces.find_agent_traces(client, agent=agent, workspace=workspace, since=since, limit=limit)

    async def read_trace(self, ref: str) -> TraceExplorer:
        """Read one production trace in full.

        Args:
            ref: A ``trace_ref`` from ``find_agent_traces``, or one from an Insight.

        Returns:
            A ``TraceExplorer`` over the trace. See the trace documentation in context.
        """
        client, workspace = self._intake()
        return await traces.read_trace(client, ref, workspace=workspace)

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=15, cell_timeout=60.0)))
    async def discover_runner(self, dataset: Dataset) -> str:
        """Discover how this dataset's evaluation runner works.

        ``self.context["dataset_documentation"]`` contains the full API reference
        for this dataset type — read it first.

        Then read a few task directories from the dataset (resolve ``task.uri``
        from ``dataset.list_tasks()`` to get the on-disk path) and inspect
        the actual files.

        Return a concise summary of the conventions observed (file layout,
        how to write add metrics).
        """  # noqa: D413
        ...

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=60, cell_timeout=3600.0)))
    async def author_insight_metrics(
        self,
        insight: Insight,
        diagnostics: list[tuple[str, Diagnostic]],
        train_dataset: Dataset,
        validation_dataset: Dataset,
        insight_suite: Dataset,
        runner_conventions: str,
        validation_feedback: str | None = None,
    ) -> MetricAuthoringResult:
        """Author verifier metrics for the materialized tasks that capture the insight.

        Args:
            insight: The insight whose failure mode the tasks should detect.
            diagnostics: Per-trace ``(trace_ref, Diagnostic)`` pairs for concrete evidence.
            train_dataset: Staged training tasks to augment for optimization feedback.
            validation_dataset: Staged validation tasks to augment for scoring.
            insight_suite: The materialized tasks recreated from the Insight's production traces.
            runner_conventions: Summary of how this dataset's runner works (from ``discover_runner``).
                Use this as the authoritative reference for what artifacts exist at
                evaluation runtime, how tasks are structured, and how to add metrics.
            validation_feedback: Actionable failures from mandatory validation of
                the previous metric-authoring attempt. If provided, repair every reported
                file before returning.

        Refer to ``self.context["dataset_documentation"]`` for the dataset-specific API
        and metric authoring conventions (file layout, how to add/remove/modify a metric).

        **Scope: every task in all three datasets**

        Add at least one new Insight-specific metric key to every task in
        ``train_dataset``, ``validation_dataset``, and ``insight_suite``. Use the same
        new metric key set and scoring semantics everywhere. Preserve every existing
        verifier metric, including ordinary ``reward`` or ``score`` values.

        Task-specific verifier edits are allowed and expected when existing verifier
        layouts differ. Only edit verifier files. Do not modify task instructions,
        environments, solutions, or other agent-visible inputs.

        Name each new metric after the root-cause behavior, not a trace id or surface
        symptom. Measure the current Harbor run from runtime artifacts such as OTLP
        traces or agent outputs. Do not hard-code scores for the production traces that
        motivated the Insight.

        Resolve the trace directory from ``os.environ["TRACE_DIR"]``. Never hard-code a
        trace path and never supply a fallback default. When a metric's evidence source is
        missing or empty, exit non-zero rather than writing a value: a ``0.0`` for absent
        evidence is indistinguishable from a measured failure.

        Only the verifier directory reaches the evaluation container, mounted at ``/tests``.
        The task instruction is not there: it reaches the agent as a prompt, so no file for it
        exists at scoring time. Read reference text from a file the task ships in its verifier
        directory; ``validate()`` rejects a path that no file provides.

        In every task's configured verifier directory, write ``metric-contract.json``
        containing exactly ``{"metric_keys": ["key_one", "key_two"]}``. The list must
        contain the same newly authored keys for every task and must exactly match the
        ``metric_keys`` returned in ``MetricAuthoringResult``.

        **Validate while authoring**

        After verifier edits, call ``await train_dataset.validate()``,
        ``await validation_dataset.validate()``, and ``await insight_suite.validate()``.
        These perform evaluator-specific static checks without launching trials. If one
        raises ``DatasetValidationError``, repair its diagnostics and revalidate all three.
        If ``validation_feedback`` is provided, fix every reported failure before returning.

        **Metric quality**

        Focus the metric on the root cause, not the surface symptom.  Prefer graded
        partial scores over binary 0/1 whenever the evidence supports measuring severity,
        frequency, or progress toward the desired behavior.  Use binary scores only when
        the behavior is genuinely all-or-nothing.  For LLM judges: instruct the judge to
        return calibrated partial-credit floats with a short rationale, not pass/fail.

        **Metric scale — higher is always better, never use raw counts**

        Every metric value must be a float in ``[0.0, 1.0]`` where ``1.0`` means perfect
        and ``0.0`` means complete failure.  Raw counts (error_count, call_count, …) are
        forbidden as metric values — they have no upper bound and make optimization
        direction ambiguous.  Convert counts to rates or invert them:

        - Error rate → ``max(0.0, 1.0 - errors / total_calls)``
        - Presence of a behavior → ``1.0`` if present, ``0.0`` if absent
        - Partial credit → fraction of required steps completed correctly

        Example: the symptom is that the agent fails to return X when a customer asks for Y.
        The root cause is that the agent does not retrieve the full set of relevant database
        objects for Y, so X is missing from its context.  Measure whether the agent retrieves
        all required objects, not merely whether X appears in the final answer.

        Return ``MetricAuthoringResult`` with the unique, non-empty ``metric_keys`` added
        by the verifier edits plus a concise summary. Include at least one
        Insight-specific key beyond generic ``reward`` or ``score``.
        """  # noqa: D413
        ...

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=20, cell_timeout=60.0)))
    async def fill_task_template(
        self,
        trace_ref: str,
        task_template: Task,
        client: AsyncNemoClient,
        workspace: str,
    ) -> Task:
        """Fill a pre-staged task template with values from a production trace.

        Steps:
        1. Read the staged task directory at ``task_template.uri`` (file:// URI).
           The caller has already copied the template to its durable candidate path;
           edit this directory in place and do not copy or rename it.
        2. Fetch the trace via ``client`` and populate the template's placeholders
           with values from the trace (instruction, environment config, etc.).
           Leave unfillable placeholders as-is.
        3. Keep ``task.toml`` parseable and keep ``[task] name`` in ``org/name``
           format. The caller will deterministically finalize the name and provenance.
        4. Return a Task whose ``uri`` still points at the staged directory.

        Args:
            trace_ref: Production trace identifier.
            task_template: Pre-staged task copy; resolve its directory from ``task_template.uri``.
            client: NeMo Platform client for fetching the trace.
            workspace: Workspace the trace belongs to.
        """
        ...

    def _trace_trial(
        self,
        insight: Insight,
        task: Task,
        trace_ref: str,
        index: int,
        insight_id: str,
    ) -> TrialResult:
        """Return TrialResult wrapper for an insight production trace."""
        return TrialResult(
            id=f"insight-trace-{index}",
            task_id=task.id,
            status="completed",
            trace=ResourceRef(uri=f"intake://{trace_ref}", description="Production trace attached to the insight."),
            metadata={
                "source": "insight",
                "trace_ref": trace_ref,
                "insight_id": insight_id,
            },
        )

    async def run(
        self,
        insight: Insight,
        agent_path: Path,
        task_template: Task,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        *,
        client: AsyncNemoClient,
    ) -> EvalAuthorResult:
        """Curate an evaluation suite and always close the owned shell session."""
        try:
            return await self._run(
                insight=insight,
                agent_path=agent_path,
                task_template=task_template,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                client=client,
            )
        finally:
            await self.shell.close()

    async def _run(
        self,
        insight: Insight,
        agent_path: Path,
        task_template: Task,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        *,
        client: AsyncNemoClient,
    ) -> EvalAuthorResult:
        """Curate an evaluation suite from an Insight and its production traces.

        Args:
            insight: The Insight to investigate with relevant traces.
            agent_path: Agent root, relative to ``experiment_dir`` or absolute.
            task_template: Parsed evaluator task containing explicit placeholders.
            train_dataset: Staged training dataset to augment in place.
            validation_dataset: Staged validation dataset to augment in place.
            client: Existing NeMo Platform client used for Intake requests.
        """
        reporter = self._reporter
        if reporter is not None:
            reporter.progress(phase="eval author · starting")

        resolved_agent = self.experiment_dir / agent_path
        insight_id = insight.id
        if not insight_id:
            raise ValueError("Eval Author requires a persisted Insight with a durable id")
        refs = insight.trace_refs[: self._config.max_traces]

        if not refs:
            if reporter is not None:
                reporter.note("no trace refs — nothing to analyze")
                reporter.progress(phase="eval author · complete")
            return EvalAuthorResult(
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                summary="No trace refs on insight — nothing to analyze.",
            )

        insight_suite = InsightSuite(
            experiment_dir=self.experiment_dir,
            insight_id=insight_id,
            task_template=task_template,
        )
        try:
            staged_tasks = insight_suite.stage(refs)
            total_tasks = len(staged_tasks)
            if reporter is not None:
                reporter.progress(
                    phase="eval author · materializing tasks",
                    completed=0,
                    total=total_tasks,
                    unit="task",
                )
            for completed, staged in enumerate(staged_tasks, start=1):
                await self.fill_task_template(staged.trace_ref, staged.task, client, insight.workspace)
                insight_suite.validate(staged)
                if reporter is not None:
                    reporter.progress(
                        phase="eval author · materializing tasks",
                        completed=completed,
                        total=total_tasks,
                        unit="task",
                    )
            materialized_dataset = insight_suite.promote_local(refs, staged_tasks)
        except BaseException:
            insight_suite.discard()
            raise
        tasks = list(materialized_dataset.list_tasks())
        trials = [
            self._trace_trial(insight, task, ref, index, insight_id)
            for index, (task, ref) in enumerate(zip(tasks, refs, strict=True), start=1)
        ]
        diagnostics: list[tuple[str, Diagnostic]] = []
        analyzer_config = TraceAnalyzerConfig(max_summary_tokens=self._config.max_summary_tokens)
        analyzers = [TraceAnalyzer(experiment_dir=self.experiment_dir, config=analyzer_config) for _ in trials]
        if reporter is not None:
            reporter.progress(
                phase="eval author · analyzing traces",
                completed=0,
                total=len(trials),
                unit="trace",
            )

        async def load_trace(reference: ResourceRef) -> TraceExplorer:
            """Resolve a trace ref with this run's client, matching the Experimentalist's
            `ctx.load_trace`: the analyzer takes a loader rather than a platform client,
            so its signature names no platform type."""
            return await TraceExplorer.from_ref(reference, client, insight.workspace)

        raw_diagnostics: list[Diagnostic | BaseException] = list(
            await asyncio.gather(
                *[
                    analyzer.run(
                        trial=trial,
                        task=task,
                        agent_path=resolved_agent,
                        insight=insight,
                        load_trace=load_trace,
                    )
                    for analyzer, trial, task in zip(analyzers, trials, tasks, strict=True)
                ],
                return_exceptions=True,
            )
        )
        analysis_statuses: dict[str, tuple[str, str | None]] = {}
        for task, ref, result in zip(tasks, refs, raw_diagnostics, strict=True):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, BaseException):
                logger.warning("Trace analysis failed for %s: %s", ref, result)
                if reporter is not None:
                    reporter.note(f"trace analysis failed for {ref}: {result}")
                analysis_statuses[task.id] = ("failed", str(result))
                continue
            diagnostics.append((ref, result))
            analysis_statuses[task.id] = ("completed", None)
        insight_suite.record_analysis(analysis_statuses)
        if reporter is not None:
            reporter.progress(
                phase="eval author · analyzing traces",
                completed=len(diagnostics),
                total=len(trials),
                unit="trace",
            )

        self.context["dataset_documentation"] = doc(type(materialized_dataset), inline_depth=1)
        if reporter is not None:
            reporter.progress(phase="eval author · discovering runner")
        runner_conventions = await self.discover_runner(train_dataset)
        if reporter is not None:
            reporter.progress(phase="eval author · authoring metrics")

        validation_feedback: str | None = None
        for repair_attempt in range(self._config.max_validation_repair_attempts + 1):
            authoring_result = await self.author_insight_metrics(
                insight,
                diagnostics,
                train_dataset,
                validation_dataset,
                materialized_dataset,
                runner_conventions,
                validation_feedback=validation_feedback,
            )
            try:
                validation_errors: list[str] = []
                for label, dataset in (
                    ("train", train_dataset),
                    ("validation", validation_dataset),
                    ("insight", materialized_dataset),
                ):
                    try:
                        await dataset.validate()
                    except DatasetValidationError as exc:
                        validation_errors.append(f"{label} dataset:\n{exc}")
                if validation_errors:
                    raise DatasetValidationError("\n".join(validation_errors))
                validate_metric_contracts(
                    {
                        "train": train_dataset,
                        "validation": validation_dataset,
                        "insight": materialized_dataset,
                    },
                    metric_keys=authoring_result.metric_keys,
                )
            except DatasetValidationError as exc:
                if repair_attempt >= self._config.max_validation_repair_attempts:
                    raise
                logger.warning(
                    "Eval Author Insight metric validation failed; requesting repair attempt %d/%d: %s",
                    repair_attempt + 1,
                    self._config.max_validation_repair_attempts,
                    exc,
                )
                if reporter is not None:
                    reporter.progress(
                        phase="eval author · repairing metrics",
                        completed=repair_attempt + 1,
                        total=self._config.max_validation_repair_attempts,
                        unit="attempt",
                    )
                validation_feedback = str(exc)
                continue

            finalized_suite = insight_suite.finalize()
            if reporter is not None:
                reporter.progress(phase="eval author · complete")
            return EvalAuthorResult(
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                insight_suite=finalized_suite.dataset,
                insight_suite_identity=finalized_suite.identity,
                metric_keys=authoring_result.metric_keys,
                summary=authoring_result.summary,
            )

        raise AssertionError("unreachable")


def build_eval_author_agent(
    experiment_dir: Path,
    config: EvalAuthorConfig | None = None,
    reporter: RunReporter | None = None,
) -> EvalAuthor:
    """Build and return a configured top-level Eval Author agent."""
    return EvalAuthor(experiment_dir=experiment_dir, config=config, reporter=reporter)
