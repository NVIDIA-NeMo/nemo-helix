# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from nemo_experimentalist_plugin.entities import Candidate, Dataset, RewardRecord, TrialResult
from nemo_experimentalist_plugin.experimentalist import roles
from nemo_experimentalist_plugin.experimentalist.components.trace_explorer import TraceExplorer  # noqa: F401
from nemo_experimentalist_plugin.experimentalist.seam import PRIMARY_SPLIT, StrategyContext, TraceLoader
from nemo_helix_plugin.nooa_model_client import get_fast_model
from nooa import Agent, CodeActStrategy, strategy
from nooa.agentdoc import doc
from nooa.config import CodeActConfig
from pydantic import BaseModel, Field

from .goal_tree import GoalNode, GoalTree, GoalTreeConfig, GoalTreeGenerator, leaf_weights_by_id, traverse_tree

logger = logging.getLogger(__name__)

#: Concurrent scoring calls. `max_trajectory_tasks` caps tasks, not the nodes each is
#: scored against, so the product is what needs bounding.
_MAX_CONCURRENT_SCORING = 8


def _trajectory_detail_from_reward(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        reward_val = value.get("reward", value.get("score", 0.0))
        explanation = value.get("reason") or value.get("explanation") or ""
    else:
        reward_val = getattr(value, "reward", getattr(value, "score", value))
        explanation = getattr(value, "reason", "") or getattr(value, "explanation", "")
    detail: dict[str, Any] = {"reward": float(reward_val)}
    if explanation:
        detail["explanation"] = str(explanation)
    return detail


def _complete_trace_groups(
    traces_by_task: dict[str, dict[str, TrialResult]], agent_ids: list[str]
) -> dict[str, dict[str, TrialResult]]:
    required_agents = set(agent_ids)
    return {
        task_id: by_agent
        for task_id, by_agent in traces_by_task.items()
        if set(by_agent) == required_agents and all(by_agent[aid] for aid in required_agents)
    }


class GroupLeafScore(BaseModel):
    """A numeric score and qualitative reason for a single agent on a goal-tree leaf."""

    score: float = Field(ge=0.0, le=1.0)
    reason: str
    span_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Span IDs from the trace that contain the key evidence cited in `reason`. "
            "Empty when the trace has no turns to look them up from, which is the normal "
            "case for an agent that makes no LLM calls."
        ),
    )


class GoalTreeTrajectoryScorer(Agent, roles.TrajectoryScorer):
    """Score a group of agent traces against a goal-tree leaf node."""

    config_type = GoalTreeConfig

    name = "goal-tree"

    def __init__(
        self,
        workspace: Path,
        llm: Any | None = None,
        load_trace: TraceLoader | None = None,
        config: GoalTreeConfig | None = None,
        max_trajectory_tasks: int | None = None,
        framework_skills_dirs: list[Path] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the scorer for the given workspace.

        Args:
            workspace: Path to the workspace directory.
            llm: Language model instance; defaults to the configured fast model.
            load_trace: Resolves a trace reference to a TraceExplorer. Supplied by the
                context; ``run`` takes the run's own if none was given here.
            **kwargs: Additional arguments passed to parent Agent class.

        Raises:
            ValueError: if workspace does not exist or is invalid.

        """
        super().__init__(llm=llm or get_fast_model(), **kwargs)
        self.workspace = workspace
        self._load_trace = load_trace
        self.context["trace_explorer_documentation"] = doc(TraceExplorer)
        self._goal_config = config or GoalTreeConfig()
        self._max_trajectory_tasks = max_trajectory_tasks
        self._framework_skills_dirs = framework_skills_dirs or []

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=15, cell_timeout=120.0)))
    async def score_group(
        self,
        node: GoalNode,
        trials: dict[str, TrialResult],
        dataset: Dataset,
    ) -> dict[str, GroupLeafScore]:  # pyright: ignore[reportReturnType]  # ty: ignore[invalid-return-type]
        """Compute the relative group advantage score for a group of traces coming from different agents for a given node.
        All scores must be strictly ordered: score_a < score_b < ... < score_n (no ties).

        Args:
            node: The node to score. Contains the goal description and the expected output.
            trials: Trials to score.  Each TrialResult carries a ``trace`` reference and a
                ``metrics`` dict populated by the task evaluator (e.g. ``{"score": 0.82}``).
                Metrics reflect overall task success — do NOT use them for scoring.
            dataset: The dataset to score.

        ## Loading traces

        Load each trace with
        `await self._load_trace(trial.trace)`.
        If the trace is not available, skip the trial.

        Use TraceExplorer to inspect the full span hierarchy: sessions, turns, LLM messages, tool
        calls, code execution outputs, sub-agent spans, errors, and observations.
        All TraceExplorer query methods are async — await them.
        Descend into sub-agent spans; key evidence is often buried inside nested agents, not in the root span.

        ## Scoring against the goal node

        Score each agent *exclusively* on how well its trace satisfies `node.goal`.
        Do NOT use `trial.metrics` to determine or adjust scores — `TrialResult.metrics`
        are task-level outcome signals, not node-level evidence.  Never mention metric values
        in reasoning as justification for a score.

        Apply proportional partial credit:
        - Agent performs all required steps correctly → 0.8–1.0
        - Agent performs most steps correctly but fails or skips one key requirement → 0.5–0.7
        - Agent attempts the right approach but with significant errors that undermine the goal → 0.2–0.4
        - Agent does not address the goal or uses a fundamentally wrong approach → 0.0–0.2

        ## Writing reasons and grounding span IDs

        Each reason must cite specific evidence observed in the trace — exact function names, tool outputs,
        quoted values, or code snippets that directly relate to `node.goal`.  Generic descriptions
        ("the agent tried X") are not sufficient.  Good examples:
        - "trace shows readUInt32BE at offset 24 → 0x400520, matching ELF e_entry"
        - "grepped /app/morpheus for ImageMath imports, found zero matches in source files"
        - "WritingAgent output lists five domains: daily life, workplace, marriage, parenting, emotional well-being"

        Bad (insufficient):
        - "the agent verified the package version"
        - "trace shows the agent covered the required domains"

        Also populate `span_ids` with the IDs of the spans that contain the key evidence you cited.
        Span IDs MUST be retrieved from TraceExplorer — `await explorer.get_span_id(session_id, turn_index)`
        for a single turn, or `await explorer.get_turn_data(session_id, turn_index)` when you also need
        the turn's contents — and copied verbatim from the returned values, never abbreviated, guessed,
        or constructed from the overview text.  Include only spans that directly support the score —
        not every span.

        A trace with no turns is expected, not a failure. Both lookups above are indexed by
        turn, so an agent that makes no LLM calls — deterministic, rule-based, or purely
        tool-driven — has nothing for them to return, and they yield `None` however many times
        you call them. When `get_overview()` reports `Turns: 0`, leave `span_ids` empty and
        ground the reason in the call graph instead: name the methods that ran, the order they
        ran in, and their status. That is sufficient evidence for such a trace. Do not retry the
        turn lookups, and do not put the abbreviated ids from the overview text into `span_ids`.

        Returns:
            dict[str, GroupLeafScore]: scores indexed by agent ID; each entry includes a numeric
            score, a reason citing specific trace evidence, and the span IDs that ground the reason.
        """
        explorers: dict[str, TraceExplorer] = {}
        if self._load_trace is None:
            raise ValueError(
                "No trace loader was supplied, so no trajectory can be read. `run` takes the "
                "run's own from the context; a direct caller must pass `load_trace`."
            )
        for agent_id, trial in trials.items():
            if not trial.trace:
                continue
            explorer = await self._load_trace(trial.trace)
            explorers[agent_id] = explorer
            print(f"Agent {agent_id} trace overview:")
            print(await explorer.get_overview())
            print(await explorer.get_errors())
        ...

    async def run(
        self,
        ctx: StrategyContext,
        *,
        candidates: list[Candidate],
        round_num: int = 0,
        analysis: str | None = None,
    ) -> dict[str, RewardRecord]:
        """Score how each candidate got to its result, as one reward record each.

        The goal tree this scorer ranks against is its own: built on first use from the
        train split, refined from each round's analysis, and persisted under the run's
        analysis directory.
        """
        self._load_trace = ctx.load_trace
        await self._ensure_goal_tree(ctx.datasets["train"], ctx.ethos)
        if analysis is not None:
            await self._update_goal_tree(
                round_num=round_num, analysis=analysis, dataset=ctx.datasets["train"], ethos=ctx.ethos
            )
        results = await self._reward_trajectories(dataset=ctx.datasets[PRIMARY_SPLIT], candidates=candidates)
        by_label = {c.label: c for c in candidates}
        scored: dict[str, RewardRecord] = {}
        for label, payload in results.items():
            candidate = by_label.get(label)
            if candidate is None:
                continue
            candidate.trajectory_detail = payload["details"]
            scored[candidate.id] = RewardRecord(metrics=payload["reward"])
        return scored

    def _goal_tree_path(self, round_num: int) -> Path:
        return self.workspace / "eval-and-optimize" / "analysis" / f"round-{round_num}-goal.json"

    def _latest_goal_tree_path(self) -> Path | None:
        analysis_dir = self.workspace / "eval-and-optimize" / "analysis"
        best: tuple[int, Path] | None = None
        for p in analysis_dir.glob("round-*-goal.json"):
            try:
                n = int(p.stem.split("-")[1])
            except (IndexError, ValueError):
                continue
            if best is None or n > best[0]:
                best = (n, p)
        return best[1] if best else None

    def _load_goal_tree(
        self,
        path: Path,
        goal_config: GoalTreeConfig,
        context: str,
    ) -> GoalTree | None:
        try:
            return GoalTree.from_path(path, config=goal_config)
        except (OSError, ValueError) as exc:
            logger.warning(f"[TRAJ] Failed to load {context} goal tree {path}: {exc}")
            return None

    async def _ensure_goal_tree(self, dataset: Dataset, ethos: Path | None) -> None:
        """Generate and persist the round-0 goal tree if it does not already exist."""
        tree_path = self._goal_tree_path(0)
        if tree_path.exists():
            return
        try:
            tree = await GoalTreeGenerator(
                workspace=self.workspace,
                config=self._goal_config,
                framework_skills_dirs=self._framework_skills_dirs,
            ).generate(dataset, ethos=ethos)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[TRAJ] Failed to generate initial goal tree; continuing without trajectory scoring: {exc}")
            return
        tree_path.parent.mkdir(parents=True, exist_ok=True)
        tree_path.write_text(tree.to_json())

    async def _update_goal_tree(
        self, *, round_num: int, analysis: str, dataset: Dataset, ethos: Path | None = None
    ) -> None:
        """Refine and persist the goal tree for round *round_num* + 1."""
        next_goal_path = self._goal_tree_path(round_num + 1)
        if next_goal_path.exists():
            return
        goal_tree_path = self._latest_goal_tree_path()
        if goal_tree_path is None:
            return
        goal_config = self._goal_config
        goal_tree = self._load_goal_tree(goal_tree_path, goal_config, context="analysis")
        if goal_tree is None:
            return
        generator = GoalTreeGenerator(
            workspace=self.workspace,
            config=goal_config,
            framework_skills_dirs=self._framework_skills_dirs,
        )
        try:
            updated_tree = await generator.update(goal_tree, analysis, round_num, dataset, ethos=ethos)
        except Exception as exc:  # noqa: BLE001 - refinement is optional; the previous tree still scores
            logger.warning(
                f"[TRAJ] Failed to refine goal tree for round {round_num + 1}; keeping the previous one: {exc}"
            )
            return
        next_goal_path.parent.mkdir(parents=True, exist_ok=True)
        next_goal_path.write_text(updated_tree.to_json())

    async def _reward_trajectories(self, *, dataset: Dataset, candidates: list[Candidate]) -> dict[str, dict[str, Any]]:
        """Score candidates against the goal tree; return trajectory results keyed by candidate label."""
        tree_path = self._latest_goal_tree_path()
        if tree_path is None:
            logger.info("[TRAJ] No goal tree found, skipping trajectory scoring")
            return {}

        goal_tree = self._load_goal_tree(tree_path, self._goal_config, context="trajectory scoring")
        if goal_tree is None:
            logger.info("[TRAJ] Invalid goal tree, skipping trajectory scoring")
            return {}

        pending = [c.label for c in candidates]
        if len(pending) == 1:
            logger.info("[TRAJ] Only one agent to score, skipping GRA")
            return {}

        logger.info(f"[TRAJ] Scoring {len(pending)} agents using GRA")

        nodes = traverse_tree(goal_tree.root)
        node_weights = leaf_weights_by_id(goal_tree.root)
        traces_by_task: dict[str, dict[str, TrialResult]] = {}
        for candidate in candidates:
            if "validation" not in candidate.rewards:
                continue
            for trial in candidate.rewards["validation"].trials or []:
                if trial.trace is None:
                    continue
                traces_by_task.setdefault(trial.task_id, {}).setdefault(candidate.label, trial)

        # Only require agents that actually contributed traces; a candidate with no
        # traces at all (e.g. its validation run failed entirely) must not cause every
        # task to be dropped from scoring.
        agents_with_traces = {label for task_traces in traces_by_task.values() for label in task_traces}
        if len(agents_with_traces) < 2:
            logger.info("[TRAJ] Fewer than two agents have traces, skipping trajectory scoring")
            return {}
        complete_traces = _complete_trace_groups(traces_by_task, list(agents_with_traces))
        dropped = len(traces_by_task) - len(complete_traces)
        if dropped:
            logger.info(f"[TRAJ] Skipping {dropped} tasks missing traces for one or more agents")
        traces_by_task = complete_traces

        task_names = sorted(traces_by_task)
        total_tasks = len(task_names)
        max_tasks = self._max_trajectory_tasks
        if max_tasks and total_tasks > max_tasks:
            task_names = task_names[:max_tasks]
            traces_by_task = {t: traces_by_task[t] for t in task_names}
            logger.info(f"[TRAJ] Selected first {max_tasks}/{total_tasks} complete tasks")

        if not traces_by_task:
            logger.info("[TRAJ] No complete trace groups found, skipping")
            return {}
        if self._goal_config is None:
            return {}

        scores_by_node_trial_agent: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        keys = [(node.id, task_id) for node in nodes for task_id in traces_by_task]
        logger.info(f"[TRAJ] Starting {len(keys)} GRA scoring tasks...")

        # Bounded, and tolerant of one call failing. The fan-out is nodes x tasks and every
        # element is an LLM call, so a wide goal tree would otherwise issue them all at
        # once; and this whole channel is optional, so one unreadable trace must not end a
        # run that has already spent hours.
        limit = asyncio.Semaphore(_MAX_CONCURRENT_SCORING)

        async def _score(node: GoalNode, task_id: str) -> dict[str, GroupLeafScore]:
            async with limit:
                return await self.score_group(node, traces_by_task[task_id], dataset)

        scoring_results = await asyncio.gather(
            *[_score(node, task_id) for node in nodes for task_id in traces_by_task],
            return_exceptions=True,
        )
        logger.info("[TRAJ] Scoring completed")
        for (node_id, task_id), group_scores in zip(keys, scoring_results, strict=True):
            if isinstance(group_scores, BaseException):
                logger.warning("[TRAJ] scoring %s on %s failed; skipping: %s", node_id, task_id, group_scores)
                continue
            for aid, gs in group_scores.items():
                scores_by_node_trial_agent[node_id][task_id][aid] = _trajectory_detail_from_reward(gs)

        trajectory_results: dict[str, dict[str, Any]] = {}
        for aid in pending:
            details: dict[str, dict[str, dict[str, Any]]] = {}
            for node in nodes:
                details[node.id] = {
                    tid: by_agent[aid]
                    for tid, by_agent in scores_by_node_trial_agent[node.id].items()
                    if aid in by_agent
                }
            node_means = {
                node.id: sum(item["reward"] for item in details[node.id].values()) / len(details[node.id])
                for node in nodes
                if details[node.id]
            }
            trajectory_reward = sum(node_weights[node.id] * node_means.get(node.id, 0.0) for node in nodes)
            trajectory_results[aid] = {
                "details": details,
                "reward": {
                    "aggregate": trajectory_reward,
                    **node_means,
                },
            }
            logger.info(f"[TRAJ] {aid}: trajectory_reward={trajectory_reward:.3f}")

        return trajectory_results
