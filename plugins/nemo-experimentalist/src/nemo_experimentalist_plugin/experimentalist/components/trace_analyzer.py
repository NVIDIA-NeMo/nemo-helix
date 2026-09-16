# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast  # noqa: D100, F401
import hashlib
import logging
import os  # noqa: F401
import re  # noqa: F401
from collections import Counter, defaultdict  # noqa: F401
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from nemo_experimentalist_plugin.entities import DependencyRuntime, MetricResult, Task, TrialResult
from nemo_experimentalist_plugin.experimentalist.components import cache
from nemo_experimentalist_plugin.experimentalist.components.tools import GuardedShellTools, WorkspaceTool
from nemo_experimentalist_plugin.experimentalist.seam import TraceLoader
from nemo_helix_plugin.nooa_model_client import get_default_model, get_fast_model
from nemo_insights_plugin.entities import Insight
from nooa import Agent, CodeActStrategy, strategy
from nooa.agentdoc import doc, spec
from nooa.agents import TokenBudgetSummarizer
from nooa.config import CodeActConfig
from nooa.config.summarizer_config import TokenBudgetConfig
from nooa.skill_registry import SkillRegistry
from nooa.tools import Match, TodoManager
from pydantic import BaseModel, Field

from .rationale import Rationale
from .trace_explorer import SearchResult, SessionData, SessionSummary, TraceExplorer, TurnInfo
from .util import load_framework_skills


class TraceAnalyzerConfig(BaseModel):
    """Configuration for TraceAnalyzer tuning parameters."""

    max_summary_tokens: int = Field(
        default=80_000,
        description="Max tokens the token-budget summarizer may use.",
    )


logger = logging.getLogger(__name__)


def _trace_cache_key(trace_uri: str) -> str:
    parsed = urlparse(trace_uri)
    if parsed.scheme == "file" and parsed.netloc in ("", "localhost"):
        return cache.trace_hash(Path(unquote(parsed.path)).expanduser())
    if parsed.scheme == "":
        return cache.trace_hash(Path(trace_uri).expanduser())
    digest = hashlib.sha256(trace_uri.encode()).hexdigest()
    return f"trace-uri-{digest}"


class TraceOverview(BaseModel):
    """High-level structural overview of a trace execution."""

    outcome: Literal["SUCCESS", "FAILURE"] | None
    status: str
    number_of_sessions: int
    total_turns: int
    errors_present: bool
    metrics: dict[str, MetricResult]


class DecisionPoint(BaseModel):
    """A single span where the agent made an incorrect or suboptimal decision."""

    span_index: int
    observation: str  # what the agent did or concluded at this span
    mistake: str  # what was wrong or missing about it


class StepAnalysis(BaseModel):
    """Trajectory summary and key decision points from trace analysis."""

    trajectory_summary: str
    decision_points: list[DecisionPoint]


class Diagnostic(BaseModel):
    """Root-cause diagnosis for a single agent trial."""

    outcome: Literal["SUCCESS", "FAILURE"]
    summary: str
    failure_point: int | None
    root_cause: str


def _outcome_from_eval_passed(eval_passed: bool | None) -> Literal["SUCCESS", "FAILURE"] | None:
    if eval_passed is None:
        return None
    return "SUCCESS" if eval_passed else "FAILURE"


class TraceAnalyzer(Agent):
    """Perform deep trace analysis for a single task.

    Spawned by TraceRootCauseAnalyzer.run in parallel — one instance per task.
    Never call this sequentially; always use asyncio.gather.

    When using TraceExplorer return types, prefer attribute access. If you
    encounter a type whose fields aren't in your system prompt, call
    ``doc(TheType)`` before assuming attribute names — don't guess.

    """

    def __init__(
        self,
        experiment_dir: Path,
        config: TraceAnalyzerConfig | None = None,
        framework_skills_dirs: list[Path] | None = None,
        **kwargs: Any,
    ):
        """Initialize the trace analyzer for the given experiment directory.

        Args:
            experiment_dir: Absolute path to the experiment root.
            config: Tuning parameters; defaults to ``TraceAnalyzerConfig()`` if ``None``.
            framework_skills_dirs: Optional list of directories containing framework skills to load.
            **kwargs: Forwarded to ``Agent.__init__``.

        """
        super().__init__(llm=kwargs.pop("llm", None) or get_default_model(), **kwargs)
        self._config = config or TraceAnalyzerConfig()
        self._experiment_dir = experiment_dir
        self.shell = GuardedShellTools(cwd=experiment_dir)
        self.workspace = WorkspaceTool(workspace=experiment_dir)
        self.todos = TodoManager()
        self.context["file_match"] = doc(Match)
        self.context["workspace_tool_documentation"] = doc(WorkspaceTool)
        self.context["trace_documentation"] = doc(
            TraceExplorer,
            SessionSummary,
            TurnInfo,
            SessionData,
            SearchResult,
            inline_depth=2,
        )
        self.skills: SkillRegistry = SkillRegistry(self)
        spec(self, "skills", hidden=True)
        load_framework_skills(self.skills, framework_skills_dirs or [])
        TokenBudgetSummarizer.install(
            self,
            llm=get_fast_model(),
            config=TokenBudgetConfig(max_tokens=self._config.max_summary_tokens),
        )

    async def get_overview(self, trace: TraceExplorer, trial: TrialResult) -> TraceOverview:
        """Return a high-level structural overview of the trace.

        Args:
            trace: The loaded trace to inspect.
            trial: Trial result whose status, error, and metrics are attached
                to the overview without assuming evaluator-specific semantics.

        Returns:
            TraceOverview: outcome, number of sessions, total turns, and whether
            runtime errors are present.

        """
        overview_data = await trace.get_overview_data()  # type: ignore[misc] (nooa lacks py.typed marker, so pyright ignores its type annotations)

        if not overview_data:
            return TraceOverview(
                outcome="FAILURE",
                status=trial.status,
                number_of_sessions=0,
                total_turns=0,
                errors_present=trial.error is not None,
                metrics=trial.metrics,
            )

        stats = overview_data.stats
        return TraceOverview(
            outcome=_outcome_from_eval_passed(stats.eval_passed),
            status=trial.status,
            number_of_sessions=stats.session_count,
            total_turns=stats.turn_count,
            errors_present=stats.runtime_errors > 0 or trial.error is not None,
            metrics=trial.metrics,
        )

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=40, cell_timeout=3600.0)))
    async def analyze_trajectory(
        self,
        trace: TraceExplorer,
        trial: TrialResult,
        task: Task,
        overview: TraceOverview,
        rationale: Rationale,
        agent_path: Path,
        runtime: DependencyRuntime | None,
        insight: Insight | None = None,
        selection_reason: str = "",
        objective_metrics: list[dict[str, Any]] | None = None,
        regression_metrics: list[dict[str, Any]] | None = None,
    ) -> StepAnalysis:
        """Trace the agent's path and find where it went wrong.

        Args:
            trace: The loaded trace to inspect from the trial result.
            task: The input task to analyze.
            trial: The output trial result to analyze of the input task.
            overview (TraceOverview): result from get_overview
            rationale (Rationale): reference solution; may have empty steps
            insight: Production insight. If present, use its title and
                description as the analysis lens.
            selection_reason: Why the trial was selected for in-depth analysis.
            objective_metrics: Evaluator metric dimensions the optimization run
                aims to improve.
            regression_metrics: Evaluator metric dimensions that must not worsen.
            agent_path: Local path to the agent root.
            runtime: Dependency runtime to use for analyzing the trace.

        Returns:
            StepAnalysis: result with trajectory_summary and decision_points.

        Use overview for context on the trace structure.
        If rationale has steps, use it as compass: where did the agent's reasoning diverge?
        If insight is present, look specifically for the behavior described by
        ``insight.title`` and ``insight.description``. If ``agent_path`` is present,
        inspect agent code there when it helps explain the trace.
        Use ``selection_reason`` to focus on the evidence that motivated this
        analysis. Interpret the trace and ``trial.metrics`` against
        ``objective_metrics`` (dimensions to improve) and ``regression_metrics``
        (dimensions to preserve). Do not invent scalar formulas, weights, or
        thresholds for evaluator-defined metrics.
        Frame decision points around the objective metric shortfall. A regression
        metric may identify a guardrail risk, but is not by itself the failure this
        optimization round is trying to diagnose.

        All TraceExplorer methods are async — always `await` them.

        The framework enters ``task.start_deps()`` before this method. Inspect
        ``self.context["dependencies"]`` to see the active dependency runtime,
        including start, readiness, and stop command specs.

        1. Use ``task.inputs``, ``task.resources``, and ``task.metric_specs`` for
           task-side context. Do not assume Harbor paths or hidden verifier/oracle
           resources exist.
        2. Use ``trial.outputs``, ``trial.resources``, ``trial.metrics``,
           ``trial.error``, and ``trial.metadata`` for run-side evidence.
        3. Read ResourceRef descriptions and metadata first; load referenced files
           only when needed.
        4. Review what each trace method/tool call returned.
        5. Search for errors, bad decisions, missed verifier signals, and missing side effects.
        6. Sample turns at beginning, middle, and end for long sessions.
        7. For every problematic span, ask WHY given what the agent knew at that point.

        Return analysis text and 1-5 key decision point span indices.
        """  # noqa: D413
        ...

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=30, cell_timeout=3600.0)))
    async def diagnose(
        self,
        trace: TraceExplorer,
        analysis: StepAnalysis,
        insight: Insight | None = None,
        objective_metrics: list[dict[str, Any]] | None = None,
        regression_metrics: list[dict[str, Any]] | None = None,
    ) -> Diagnostic:
        """Determine the primary root cause and produce a Diagnostic.

        Args:
            trace (TraceExplorer): the loaded trace to inspect from the trial result.
            analysis (StepAnalysis): result from analyze_trajectory
            insight: Optional production insight. If present, diagnose the root cause
                of that insight's failure behavior.
            objective_metrics: Evaluator metric dimensions the optimization run
                aims to improve.
            regression_metrics: Evaluator metric dimensions that must not worsen.

        Returns:
            Diagnostic: result with outcome, summary, failure_point, and root_cause.

        Ask for each candidate: 'If we fixed this one thing, would the agent pass?'
        If yes → root cause. If not → dig deeper.

        Required evidence:
        - Direct quote or span from one of analysis.decision_points (each has span_index, observation, mistake)
        - Count of turns with this pattern (use `await trace.search(...)` — all TraceExplorer methods are async)
        - Why earlier turns didn't catch or correct it

        Return a complete Diagnostic with outcome, summary, failure_point, and root_cause.
        If insight is present and the trace does not match the insight's behavior,
        set ``outcome="SUCCESS"`` with ``failure_point=None``.
        Use the metric contract to distinguish objective shortfalls from regression
        risks. The root cause must explain an objective metric's underperformance;
        describe regression effects only as constraints. Do not invent scalar
        formulas, weights, or thresholds.
        """  # noqa: D413
        ...

    async def run(
        self,
        trial: TrialResult,
        task: Task,
        agent_path: Path,
        rationale: Rationale | None = None,
        insight: Insight | None = None,
        selection_reason: str = "",
        objective_metrics: list[dict[str, Any]] | None = None,
        regression_metrics: list[dict[str, Any]] | None = None,
        load_trace: TraceLoader | None = None,
    ) -> Diagnostic:
        """Run the full trace analysis pipeline for one agent trial.

        Args:
            trial: The trial to analyze.
            task: The task to analyze.
            agent_path: Required local path to the agent root.
            rationale: Optional reference solution produced by Rationalizer; used as
                a compass for trajectory analysis. Pass ``None`` to skip comparison.
            insight: Optional production insight to use as the failure lens.
            selection_reason: Why the analyzer selected this trial for inspection.
            objective_metrics: Evaluator metric dimensions the optimization run
                aims to improve.
            regression_metrics: Evaluator metric dimensions that must not worsen.
            load_trace: Resolves a trace reference to a TraceExplorer, wherever the
                trace is stored. Supplied by the context; ``None`` skips trace analysis.

        Returns:
            Diagnostic: outcome, summary, failure point span index, and root cause.

        """
        if not trial.trace:
            return Diagnostic(
                outcome="FAILURE",
                summary=f"Trial {trial.id} has no trace reference.",
                failure_point=None,
                root_cause=(
                    "Trace data was not provided by the evaluator, so trajectory-level root cause analysis could not run."
                ),
            )

        try:
            if load_trace is None:
                raise ValueError("no trace loader was supplied")
            trace = await load_trace(trial.trace)
        except Exception as exc:
            logger.warning("Skipping unloadable trace %s for trial %s: %s", trial.trace.uri, trial.id, exc)
            return Diagnostic(
                outcome="FAILURE",
                summary=f"Trace could not be loaded: {type(exc).__name__}: {exc}",
                failure_point=None,
                root_cause=(
                    "Trace data could not be loaded before analysis. Inspect the trial trace ResourceRef, evaluator "
                    "trace artifact, Intake client/workspace configuration, and any trial error metadata."
                ),
            )

        # The cache key intentionally omits the insight lens. Within the optimization
        # loop the Eval Author (insight-conditioned) and the TraceRootCauseAnalyzer (no insight)
        # analyze disjoint trace sets in a single experiment_dir, so a given trace is
        # only ever diagnosed through one lens and cannot collide here. If a future
        # caller analyzes the same trace under different insights in one experiment_dir,
        # fold the insight identity into this key to avoid returning a stale-lens Diagnostic.
        metric_contract_key = (
            f"{_trace_cache_key(trial.trace.uri)}:selection-reason:{selection_reason}:"
            f"objective-metrics:{objective_metrics or []}:regression-metrics:{regression_metrics or []}"
        )
        key = cache.trace_uri_hash(metric_contract_key)

        cached = cache.load(self._experiment_dir, key, Diagnostic)
        if cached is not None:
            return cached

        rationale = rationale or Rationale(task_name=task.id, steps=[])

        async with task.start_deps() as runtime:
            overview = await self.get_overview(trace, trial)
            analysis = await self.analyze_trajectory(
                trace=trace,
                trial=trial,
                task=task,
                overview=overview,
                rationale=rationale,
                insight=insight,
                selection_reason=selection_reason,
                objective_metrics=objective_metrics,
                regression_metrics=regression_metrics,
                agent_path=agent_path,
                runtime=runtime,
            )
            diagnostic = await self.diagnose(
                trace,
                analysis,
                insight,
                objective_metrics,
                regression_metrics,
            )
            cache.store(self._experiment_dir, key, diagnostic)
        return diagnostic
