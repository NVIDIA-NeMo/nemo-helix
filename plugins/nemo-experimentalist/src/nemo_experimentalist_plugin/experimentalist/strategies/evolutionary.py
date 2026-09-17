# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The default strategy: a Pareto evolutionary loop over candidate agents.

Orchestrates baseline → [stop? → select → train-eval → analyze → propose → build →
record → validation-eval] across rounds, delegating every step to a resolved component.
"""

from __future__ import annotations

import asyncio
import logging
import random
import shutil
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

from nemo_experimentalist_plugin.config import EvolutionaryOptimizerConfig
from nemo_experimentalist_plugin.entities import (
    Candidate,
    Dataset,
    EvaluationResult,
    Proposal,
)
from nemo_experimentalist_plugin.experimentalist import roles
from nemo_experimentalist_plugin.experimentalist.components.holdout_utils import (
    ensure_heldout_hidden,
    restore_heldout_splits,
)
from nemo_experimentalist_plugin.experimentalist.components.importer import IMPORT, import_proposal
from nemo_experimentalist_plugin.experimentalist.components.models import (
    EvolutionTree,
    missing_objective_reason,
)
from nemo_experimentalist_plugin.experimentalist.components.tools import (
    GuardedShellTools,
    WorkspaceTool,
)
from nemo_experimentalist_plugin.experimentalist.components.util import load_framework_skills
from nemo_experimentalist_plugin.experimentalist.registry import get_component, resolve
from nemo_experimentalist_plugin.experimentalist.seam import StrategyContext, TraceLoader
from nemo_platform_plugin.nooa_model_client import get_default_model, get_fast_model
from nooa import Agent, CodeActStrategy, strategy
from nooa.agentdoc import doc, spec
from nooa.agents import TokenBudgetSummarizer
from nooa.config import CodeActConfig
from nooa.config.summarizer_config import TokenBudgetConfig
from nooa.skill import Skill
from nooa.skill_registry import SkillRegistry
from nooa.tools import Match

logger = logging.getLogger(__name__)


class AnalysisSkill(Skill):
    """Standard format for the per-round analysis file the optimizer writes each round.

    The round analysis merges every agent's per-agent analysis into one markdown file
    (``eval-and-optimize/analysis/round-N.md``). It is the record a human or the next
    round reads to understand what happened: which agents lead on which reward
    dimensions, where they diverge, and which failures are systematic vs mechanical.
    Follow this format exactly and fill every section with real data from the per-agent
    analyses and metadata — no placeholders.

    ---

    ```markdown
    # Round N Analysis

    ## Rewards

    **Show every dimension as its own column** so agents with different strengths are visible
    and easier to reason about for merging. An agent may have lower values on some dimensions
    but higher on others — that's a candidate to merge with a complementary agent.

    Train Reward:
    | Agent | <dim1> | <dim2> | <dim3> | ... | Ancestor | vs. Ancestor |
    | ----- | ------ | ------ | ------ | --- | -------- | ------------ |
    | agent-3 | 0.61 | 0.72 | 0.51 | ... | agent-0 | +0.16 |
    | agent-1 | 0.58 | 0.68 | 0.48 | ... | agent-0 | +0.13 |
    | agent-0 | 0.45 | 0.55 | 0.40 | ... | --- | baseline |

    Validation Reward:
    | Agent | <dim1> | <dim2> | <dim3> | ... | Ancestor | vs. Ancestor |
    | ----- | ------ | ------ | ------ | --- | -------- | ------------ |
    | agent-3 | 0.51 | 0.62 | 0.44 | ... | agent-0 | -0.10 |
    | agent-1 | 0.48 | 0.58 | 0.41 | ... | agent-0 | -0.10 |
    | agent-0 | 0.45 | 0.55 | 0.40 | ... | --- | baseline |

    ## Trajectory Rewards

    Trajectory rewards measure intermediate step quality (goal-tree subgoal rollup).
    Read from metadata:

    ```python
    candidate = self.workspace.get_metadata(agent_id)
    traj = candidate.rewards["validation-trajectory"].metrics or {}
    # e.g. {"aggregate": 0.74, "parse-cli-input": 0.31, "search-web-sources": 0.78}
    details = candidate.trajectory_detail or {}
    # details[node_id][task_id] = {"reward": 0.7, "explanation": "..."}
    ```

    | Agent | <step1> | <step2> | ... | aggregate |
    | ----- | ------- | ------- | --- | --------- |
    | agent-3 | 0.45 | 0.85 | ... | 0.78 |
    | agent-1 | 0.32 | 0.90 | ... | 0.75 |
    | agent-0 | 0.31 | 0.78 | ... | 0.74 |

    [Omit if the `validation-trajectory` channel is absent or empty. Show the `aggregate` key as
    the overall column and each node_id entry as its node column.
    When a trajectory reward explains a selection tradeoff or outcome-reward disagreement,
    quote/paraphrase the relevant `trajectory_detail` explanation.]

    ## Divergent Trial Analysis

    For each agent pair with divergent results:

    ### agent-A vs agent-B

    | Task | agent-A | agent-B | Winner's Key Insight |
    |------|---------|---------|----------------------|
    | task-003 | ✓ 1.0 | ✗ 0.0 | Checked container type before classifying |
    | task-007 | ✗ 0.0 | ✓ 1.0 | Verified code reachability not just presence |

    **Divergence Pattern**: agent-A excels at X, agent-B excels at Y.
    **Transferable Insight**: agent-B's reachability check could be added to agent-A.

    ## Complementary Failures

    Agent pairs with complementary strengths (each passes tasks the other fails):

    - **agent-A + agent-B**: 5 complementary tasks (ensemble would score 0.85 vs 0.65/0.70 individual)
    - **agent-C + agent-D**: 2 complementary tasks

    ## Failure Patterns
    For each systematic failure pattern (≥2 tasks), describe:
    - Root cause description (in your own words, as specific as the evidence allows)
    - How many tasks affected (e.g. "5/8 tasks")
    - Concrete example: task name, what agent did, what it should have done
    - Trace evidence: session + turn + direct quote from the trace

    Example:
    - **Agent interprets "net revenue" as "gross revenue"** (4/7 tasks): In task `q3-analysis`,
      turn 3: agent wrote "I'll use total_sales which represents net revenue" — but total_sales
      is pre-discount. Root cause confirmed via trace.search("net revenue") returning 0 matches
      before the wrong computation.
      **agent-2 passes this task** by explicitly checking the column metadata for "net" vs "gross".

    ## Root Causes
    For each root cause, answer these three questions explicitly:
    1. **What** specifically did the agent do wrong? (concrete action or output)
    2. **Why** did it do that? (missing knowledge, wrong assumption, context issue, code bug)
    3. **What would fix it?** (the minimal change that addresses the root cause directly)
    4. **Which agents don't have this root cause?** — and what's different about them

    Do NOT write vague statements like "the agent misunderstood the task" without
    specifying what it misunderstood and why. Do NOT write "improve the prompt" without
    specifying what information is missing and where in the agent's reasoning it would help.

    ## Mechanical/Infrastructure Errors

    Errors that are NOT agent logic problems — fix these on the infrastructure side:

    | Task | Error Type | Issue | Suggested Fix |
    |------|------------|-------|---------------|
    | task-005 | environment | `ModuleNotFoundError: cryptography` | Add to container requirements |
    | task-012 | timeout | Agent hit 5min limit mid-analysis | Increase task timeout to 10min |
    | task-018 | dataset | Ground truth expects format X but instruction says Y | Fix instruction.md |
    | task-023 | api | CVE server returned 503 | Add retry logic to cve-server |
    | task-001 | config | `max_iterations` set to 100 | Increase `max_iterations` to 1000 |

    **Do NOT optimize agents to work around these issues.** Report them for infra fixes.

    Common mechanical error categories:
    - **config**: wrong config parameters, missing config, config limits exceeded
    - **environment**: missing packages, wrong versions, container config
    - **timeout**: legitimate work cut short by time limits
    - **dataset**: incorrect ground truth, ambiguous instructions, missing fixtures
    - **api/service**: external service failures, rate limits, network issues
    - **scorer**: evaluation bug, wrong expected format, edge case not handled

    **What qualifies as a mechanical error:**
    - Crashes or `sys.exit` on recoverable errors (e.g., scraping fails → exit instead of fallback)
    - Wrong output path, missing required fields, schema violations
    - Unhandled exceptions that silently discard valid data
    - Missing error handling where failure is expected (network calls, file I/O)
    - Misconfigured agent, llms, config, etc.

    **What is NOT a mechanical error** (use root cause analysis + improvements instead):
    - Shallow reasoning, poor query strategy, suboptimal architecture
    - The code works but produces low-quality output
    ```

    Fill in every section with real data from the per-agent analyses. No placeholders.
    Return the complete markdown content as a string — the caller writes it to disk.
    """


class EvolutionaryStrategy(Agent, roles.Strategy):
    """Merge each round's per-agent analyses, and write the run's optimization report."""

    #: This strategy's settings are the run config itself, not a slice of it.
    config_type = EvolutionaryOptimizerConfig

    name = "evolutionary"

    #: This loop resumes from its own round-analysis files plus ``ctx.candidates()``,
    #: so the runner may re-open an existing run and hand it back.
    supports_resume: ClassVar[bool] = True

    def __init__(
        self,
        working_dir: Path,
        config: EvolutionaryOptimizerConfig | None = None,
        framework_skills_dirs: list[Path] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm=kwargs.pop("llm", None) or get_default_model(), **kwargs)
        self.working_dir = working_dir.resolve()
        self.config = config or EvolutionaryOptimizerConfig()
        self._config = self.config
        self._workspace_path = self.working_dir
        self._framework_skills_dirs: list[Path] = framework_skills_dirs or []

        self.shell = GuardedShellTools(cwd=self.working_dir)
        self.workspace = WorkspaceTool(workspace=self.working_dir)
        self.context["file_match"] = doc(Match)
        self.context["workspace_tool_documentation"] = doc(WorkspaceTool)
        self.skills: SkillRegistry = SkillRegistry(self)
        spec(self, "skills", hidden=True)
        load_framework_skills(self.skills, self._framework_skills_dirs)
        self.skills.register("ext.analysis_skill", AnalysisSkill())
        self.skills.activate(["cmd.*", "ext.*"])
        TokenBudgetSummarizer.install(
            self,
            llm=get_fast_model(),
            config=TokenBudgetConfig(max_tokens=self.config.max_summary_tokens),
        )

    @staticmethod
    def _coder_config(config: EvolutionaryOptimizerConfig) -> dict[str, Any]:
        """The builder's settings, with the run's model catalog filled in if it named one.

        A mapping, not a model: the builder validates it against its own `config_type`, so
        a builder from another package is configured the same way. The run-level
        `model_catalog_path` is a default the builder's own setting overrides.
        """
        settings = dict(config.builder_config)
        if config.model_catalog_path is not None and settings.get("model_catalog_path") is None:
            settings["model_catalog_path"] = config.model_catalog_path
        return settings

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, ctx: StrategyContext) -> Candidate | None:
        """Run optimization and always close the owned shell session."""
        try:
            return await self._run(ctx)
        finally:
            await self.shell.close()

    async def _run(self, ctx: StrategyContext) -> Candidate | None:
        """Run the Pareto evolutionary optimization loop.

        Args:
            ctx: The run's context — its datasets, its candidates, and the verbs for
                measuring and recording them. Nothing else is reachable from here.

        Returns:
            The winning Candidate, or None when no candidate was ever scored. The
            runner turns that into the run's terminal result.
        """
        # The host may have narrowed the metric contract after authoring an Insight
        # suite, and it settles that before the strategy starts. Take it from the
        # context rather than the config captured at construction, so every component
        # resolved below ranks on what the run is actually being scored against.
        config = self.config.model_copy(
            update={
                "objective_function": ctx.objective_metrics,
                "regression_metrics": ctx.regression_metrics,
            }
        )
        self._check_proposer_builder_pairing(config)
        agents_dir, analysis_dir, _ = self._init_structure()
        train_eval_dataset = ctx.datasets["train"]
        ethos_path = ctx.ethos

        # ---- Resume or fresh start ---------------------------------------
        # Round analysis files are this strategy's own private state, so detecting the
        # last round is its own business; ``ctx.resuming`` only tells it that the runner
        # re-opened an existing run.
        if (round_num := self._detect_last_round()) is not None:
            logger.info(f"[RESUME] round {round_num}")
            await self._roll_back_to(ctx=ctx, from_round=round_num)
            evolution_tree = EvolutionTree.from_candidates(await ctx.candidates())
            candidates: list[Candidate] = list(evolution_tree.survivors(round_num))
        else:
            round_num = 0
            logger.info("phase=baseline round=0")
            await ctx.report_progress(completed=0, total=config.max_rounds, unit="round", note="baseline")

            # ---- Fetch + build baseline agent (agent-0) ------------------
            await self._ensure_baseline(ctx=ctx, config=config)
            evolution_tree = EvolutionTree.from_candidates(await ctx.candidates())
            candidates = list(evolution_tree.survivors(0))

            # ---- Baseline validation evaluation (round 0) ----------------
            validation_candidate_results = await self._evaluate_validation_candidates(
                ctx=ctx,
                candidates=candidates,
            )
            await self._record_baseline_validation(
                ctx=ctx, baseline=candidates[0], results=validation_candidate_results
            )

        phase: Literal["exploration", "exploitation"] = "exploration" if round_num % 2 == 0 else "exploitation"

        # ---- Pareto optimization loop (shared by fresh start and resume) --
        # The round budget bounds the loop itself rather than being a component's opinion:
        # a terminator that never stops — or none at all — must not produce an unbounded
        # run. The terminator decides whether to stop *early*.
        while round_num < config.max_rounds:
            prior_analysis = (
                await self._load_round_analysis(analysis_dir=analysis_dir, round_num=round_num - 1)
                if round_num > 0
                else None
            )
            if config.terminator is not None:
                terminator = self._terminator(ctx, config)
                # The whole history, not this round's working set: convergence counts distinct
                # generations and compares an older Pareto front against the current one.
                # `candidates` holds only survivors plus the previous round's builds, so
                # eliminated candidates would be invisible and the generation window could
                # never fill -- a converged run would then pay for every remaining round.
                decision = await terminator.run(
                    round_num=round_num,
                    candidates=[node.candidate for node in evolution_tree.nodes.values()],
                    prior_analysis=prior_analysis,
                )
                if decision.stop:
                    logger.info(f"phase=terminate reason={decision.reason}")
                    break

            # The selector sees slim copies so per-trial detail never reaches an LLM
            # prompt, but the population must keep the full records: a later
            # record_reward persists the whole candidate, so carrying a slim copy
            # forward would erase every other channel's trials in the store.
            # Matched on label, not id: `survivors` is a generated method, so its return
            # value can come back either as the objects themselves or re-specified as
            # JSON. A candidate's id is a computed property over a private attribute and
            # is not a model field, so the JSON route cannot carry it and validation
            # mints a fresh one. The label is an ordinary field and survives both routes,
            # and is unique within a run.
            by_label = {c.label: c for c in candidates}
            chosen = (
                await self._selector(config).survivors([c.slim() for c in candidates], k=config.max_survivors)
                if len(candidates) > 1
                else list(candidates)
            )
            survivors = [by_label[s.label] for s in chosen if s.label in by_label]
            if len(survivors) != len(chosen):
                logger.warning(
                    "Selector returned %d candidates this round but only %d are in the population; "
                    "the rest are dropped and their originals killed.",
                    len(chosen),
                    len(survivors),
                )
            if chosen and not survivors:
                # Nothing the selector named exists here, so this says nothing about which
                # candidates are worth keeping. Killing is durable and kills everything:
                # the run would spend its remaining rounds with an empty population and
                # could not recover, on resume either. Keep them and let the next round
                # ask again.
                logger.warning("No returned survivor resolved; keeping the population for this round.")
                survivors = list(candidates)
            survived = {s.label for s in survivors}
            for candidate in [c for c in candidates if c.label not in survived]:
                await ctx.update_candidate(candidate, killed_generation=round_num)

            # Only the analyzer consumes these, so a diagnosis-blind run does not pay for
            # them — which is what makes `analyzer: null` cheaper and not merely quieter.
            train_candidate_results = (
                {}
                if config.analyzer is None
                else await self._evaluate_train_candidates(
                    ctx=ctx,
                    survivors=survivors,
                    round_num=round_num,
                    max_train_batch_tasks=config.max_train_batch_tasks,
                    train_batch_seed=config.train_batch_seed,
                )
            )
            for survivor in survivors:
                if survivor.label in train_candidate_results:
                    await ctx.record_reward(
                        survivor,
                        channel="train",
                        result=train_candidate_results[survivor.label],
                    )
            analysis = await self._analyze_round(
                load_trace=ctx.load_trace,
                analysis_dir=analysis_dir,
                dataset=train_eval_dataset,
                evaluations=train_candidate_results,
                survivors=[c.slim() for c in survivors],
                round_num=round_num,
                config=config,
                ethos_path=ethos_path,
            )
            proposals = await self._propose_improvements(
                analysis=analysis,
                evolution_tree=evolution_tree,
                round_num=round_num,
                phase=phase,
                config=config,
            )
            if not proposals:
                # Nothing to build, so another round would re-analyze the same population
                # and pay for it. Finalize on what has been evaluated instead.
                logger.info("phase=terminate reason=no proposals")
                break
            new_candidates = await self._build_candidates(
                ctx=ctx,
                dataset=train_eval_dataset,
                proposals=proposals,
                generation=round_num + 1,
                config=config,
            )
            for c in new_candidates:
                evolution_tree.add(c)

            candidates = survivors + new_candidates
            round_num += 1
            phase = "exploration" if round_num % 2 == 0 else "exploitation"
            await ctx.report_progress(
                completed=round_num,
                total=config.max_rounds,
                unit="round",
                note="evaluating candidates",
            )
            # Announce candidates before the (batched) validation eval, so the narration
            # reports work beginning, not completed. Mirror
            # _evaluate_validation_candidates' own filter so we only announce candidates
            # that will actually be evaluated (cached survivors already carry one).
            pending_validation = [c for c in candidates if "validation" not in c.rewards]
            for i, candidate in enumerate(pending_validation, start=1):
                ctx.note(f"{candidate.label} ({i}/{len(pending_validation)}): {candidate.description}")

            validation_candidate_results = await self._evaluate_validation_candidates(
                ctx=ctx,
                candidates=candidates,
            )
            for candidate in candidates:
                if candidate.label in validation_candidate_results:
                    await ctx.record_reward(
                        candidate,
                        channel="validation",
                        result=validation_candidate_results[candidate.label],
                    )
            if config.trajectory_scorer is not None:
                scorer = self._trajectory_scorer(ctx, config)
                # round_num - 1: the counter has already advanced past the round this
                # analysis describes, and the scorer names its state after that round.
                # Best-effort: trajectory scoring is optional enrichment on top of the
                # validation channel, so a failure here (an unreadable trace, an LLM error
                # the scorer could not absorb) must never end a run that has already spent
                # hours producing validation rewards.
                try:
                    scored = await scorer.run(ctx, candidates=candidates, round_num=round_num - 1, analysis=analysis)
                except Exception as exc:  # noqa: BLE001 - see above
                    logger.warning("[TRAJ] trajectory scoring failed (run continues): %s", exc)
                    scored = {}
                for candidate in candidates:
                    if (record := scored.get(candidate.id)) is not None:
                        await ctx.record_reward(candidate, channel="validation-trajectory", result=record)

            for candidate in new_candidates:
                await ctx.archive_candidate(candidate)

        return await self._finalize(evolution_tree=evolution_tree, selector=self._selector(config))

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=100, cell_timeout=3600.0)))
    async def merge_analysis(
        self,
        agent_ids: list[Candidate],
        round: int,
        per_agent_analyses: list[str],  # noqa: A002
    ) -> str:  # pyright: ignore[reportReturnType]
        """Merge per-agent analyses, compare agents, and write the round analysis file.

        ## Step 1: Compare agents at reward level

        Read each agent's per-dimension train rewards from metadata:

        ```python
        rewards = {c.label: self.workspace.get_metadata(c.label).rewards["train"].metrics or {} for c in agent_ids}
        all_candidates = [
            self.workspace.get_metadata(agent_id).slim() for agent_id in self.workspace.list_agents()
        ]
        baseline = next((candidate for candidate in all_candidates if candidate.is_baseline), None)
        ```

        - Compare siblings: which optimization strategy worked better this round?
        - Compare to ancestors: did the change actually fix the targeted root cause?
        - When any Insight Suite rewards are present, compare those dimensions to the
          round-zero baseline separately from train and validation rewards.

        ## Step 2: Analyze divergent and complementary patterns

        Using the per-dimension rewards above, identify where agents diverge (one
        leads, another trails on a dimension) and where their strengths are
        complementary. Ground observations in the per-agent analyses passed in.

        ## Step 3: Build the round analysis markdown

        First, discover reward dimensions from metadata:
        ```python
        candidate = self.workspace.get_metadata(agent_ids[0].label).slim()
        train_reward = candidate.rewards["train"].metrics or {}
        dim_keys = sorted(train_reward.keys())
        insight_dim_keys = sorted({key for reward in insight_rewards.values() for key in reward})
        ```

        Follow the `ext.analysis_skill` format exactly for every section (Rewards tables,
        including the conditional Insight Suite Reward table; Trajectory Rewards; Divergent
        Trial Analysis; Complementary Failures; Failure Patterns; Root Causes;
        Mechanical/Infrastructure Errors).

        If at least one agent has a non-empty `insight_reward`, the round analysis must name
        every available Insight Suite dimension and show its values in the separate Insight
        Suite Reward table. Never blend those metrics into train/validation rewards or imply
        that they affected ranking. These metrics may steer this analysis, the goal tree, and
        the proposer only as adaptive/development feedback; label claims accordingly and never
        present them as independent validation evidence. Fill in every included section with
        real data. No placeholders.
        Return the complete markdown content as a string.
        """
        ...

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=100, cell_timeout=3600.0)))
    async def write_final_report(self, best_agent_id: str) -> None:  # pyright: ignore[reportReturnType]
        """Write the final optimization report to eval-and-optimize/OPTIMIZATION.md.

        ## MANDATORY: use self.workspace — DO NOT ls / cat / find manually

        ```python
        agent_ids = self.workspace.list_agents()
        candidate = self.workspace.get_metadata(agent_id).slim()
        analysis  = self.workspace.read_analysis_file(n)
        ```

        ## Steps
        1. Collect metadata for every agent.
        2. Read round analysis files.
        3. Build the lineage tree by following ancestor chains.
        4. Write eval-and-optimize/OPTIMIZATION.md with format:
           - Summary (baseline vs best rewards, rounds completed, total agents)
           - Reward Breakdown table (one row per agent, per-dimension columns)
           - Insight Suite Metrics table when available
           - Lineage Tree (ASCII tree with rewards and optimization type)
           - Round-by-Round Analysis
           - Optimization Insights

        When both the round-zero baseline and best agent have non-empty `insight_reward`,
        the Summary must state whether the Insight-specific scenarios improved and the
        Insight Suite Metrics table must show every available dimension with baseline,
        winner, and signed delta columns. Keep this table separate from generic train and
        validation rewards. Omit it only when Insight Suite rewards are unavailable.

        Fill in every section with real data. Every agent must appear in the lineage tree.
        Mark the best agent with * BEST.
        """
        ...

    # ------------------------------------------------------------------
    # Private helpers — concrete (no LLM, no evaluator)
    # ------------------------------------------------------------------

    def _init_structure(self) -> tuple[Path, Path, Path]:
        """Create the eval-and-optimize workspace directories; return (agents_dir, analysis_dir, results_dir)."""
        agents_dir = self.working_dir / "eval-and-optimize" / "agents"
        analysis_dir = self.working_dir / "eval-and-optimize" / "analysis"
        results_dir = self.working_dir / "eval-and-optimize" / "results"
        agents_dir.mkdir(parents=True, exist_ok=True)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        return agents_dir, analysis_dir, results_dir

    def _detect_last_round(self) -> int | None:
        """Return the last completed round by scanning analysis files, or None if starting fresh."""
        analysis_dir = self.working_dir / "eval-and-optimize" / "analysis"
        if not analysis_dir.exists():
            return None
        rounds = []
        for f in analysis_dir.glob("round-*.md"):
            try:
                rounds.append(int(f.stem.split("-")[1]))
            except (IndexError, ValueError):
                continue
        return max(rounds) if rounds else None

    async def _roll_back_to(self, *, ctx: StrategyContext, from_round: int) -> None:
        """Roll back everything produced after *from_round* so the loop can re-enter cleanly.

        Which candidates those are comes from the stored records rather than a directory
        walk, so the rollback follows the entity contract rather than our own layout.
        Candidates are discarded, not deleted: the record and artifact both survive, and
        ``ctx.candidates()`` stops returning them. Evaluator scratch *is* removed, since
        it is keyed by label and the re-run would otherwise read a previous round's
        results. Stale ``killed_generation`` markers whose killing round was itself rolled
        back are cleared, or those survivors stay dead.
        """
        eo = self.working_dir / "eval-and-optimize"
        results_dir, analysis_dir = eo / "results", eo / "analysis"
        smoke_dataset_dir, smoke_results_dir = eo / "smoke-dataset", eo / "smoke-results"

        for candidate in await ctx.candidates():
            if candidate.generation > from_round:
                await ctx.discard_candidate(candidate)
                for scratch in (
                    *results_dir.glob(f"{candidate.label}-*"),
                    smoke_results_dir / candidate.label,
                    smoke_dataset_dir / candidate.label,
                ):
                    if scratch.is_dir():
                        shutil.rmtree(scratch)
            elif candidate.killed_generation is not None and candidate.killed_generation > from_round:
                await ctx.update_candidate(candidate, killed_generation=None)

        for pattern in ("round-*.md", "round-*-goal.json"):
            for f in analysis_dir.glob(pattern):
                try:
                    n = int(f.stem.split("-")[1])
                except (IndexError, ValueError):
                    continue
                if n > from_round:
                    f.unlink()

    @staticmethod
    async def _record_baseline_validation(
        *, ctx: StrategyContext, baseline: Candidate, results: dict[str, EvaluationResult]
    ) -> None:
        """Record the baseline's validation reward, if this round actually measured one.

        Absent whenever the baseline was already scored before the crash that sent us
        here: `_ensure_baseline` keeps the existing baseline rather than minting a second,
        and `_evaluate_validation_candidates` only returns candidates it actually ran. A
        round-0 resume therefore has nothing to record, and indexing the map raised
        KeyError on a round-0 resume.

        Raises:
            ValueError: if the baseline measured here carries no objective metric.
                Every later comparison is against this one measurement, and no later
                gate notices its absence -- unmeasured candidates are mutually
                incomparable, so ranking becomes a no-op and the run pays for every
                round before reporting no winner.
        """
        measured = results.get(baseline.label)
        if measured is None:
            return
        reason = missing_objective_reason(measured.trials, measured.aggregate_metrics, ctx.objective_metrics)
        if reason is not None:
            raise ValueError(
                f"Baseline {baseline.label} produced no objective metric: {reason}. The run is scored against "
                "this measurement, so fix the evaluation before running again."
            )
        await ctx.record_reward(baseline, channel="validation", result=measured)

    async def _ensure_baseline(self, *, ctx: StrategyContext, config: EvolutionaryOptimizerConfig) -> None:
        """Create the baseline candidate unless this run already has one.

        A crash during round 0 leaves ``run.json`` behind but no round analysis, so the
        runner resumes and this branch runs again. Candidate ids are uuids, so a
        second commit mints a duplicate baseline that is re-evaluated, offered to the
        Proposer, and published to the same branch as the original.
        """
        if any(candidate.is_baseline for candidate in await ctx.candidates()):
            logger.info("[RESUME] baseline already committed; not creating a second one")
            return
        await self._create_baseline_agent(ctx=ctx, config=config)

    async def _create_baseline_agent(
        self,
        *,
        ctx: StrategyContext,
        config: EvolutionaryOptimizerConfig,
    ) -> Candidate:
        """Build the run's baseline: the agent under test, committed unchanged.

        An ordinary build of an ordinary Proposal — it is the baseline only because
        nothing precedes it. The architecture doc is generated afterwards, into the
        artifact the Candidate already addresses.
        """
        proposal = import_proposal("baseline: the agent under test, unchanged")
        builder = cast("type[roles.Builder]", resolve("builder", IMPORT))()
        baseline = await builder.build(ctx, proposal, generation=0)
        await self._generate_architecture_doc(ctx=ctx, agent_dir=ctx.candidate_dir(baseline), config=config)
        return baseline

    async def _load_round_analysis(
        self,
        *,
        analysis_dir: Path,
        round_num: int,
    ) -> str | None:
        """Return the cached analysis markdown for *round_num*, or None if absent."""
        path = analysis_dir / f"round-{round_num}.md"
        return path.read_text() if path.exists() else None

    def _format_evolution_table(self, evolution_tree: EvolutionTree) -> str:
        if not evolution_tree.nodes:
            return "(none yet — this is the first round)"
        return evolution_tree.to_markdown_table()

    async def _evaluate_agent(
        self,
        ctx: StrategyContext,
        candidate: Candidate,
        split: str,
        task_ids: list[str] | None = None,
        minimum_attempts: int | None = None,
    ) -> tuple[Candidate, EvaluationResult]:
        """Evaluate one candidate and return the candidate/result pair, for ``gather``."""
        result = await ctx.evaluate(
            candidate,
            split=split,
            task_ids=task_ids,
            minimum_attempts=minimum_attempts,
        )
        return (candidate, result)

    # ------------------------------------------------------------------
    # Private step methods — implementations
    # ------------------------------------------------------------------

    async def _generate_architecture_doc(
        self,
        *,
        ctx: StrategyContext,
        agent_dir: Path,
        config: EvolutionaryOptimizerConfig,
    ) -> None:
        """Ask the run's Builder to document *agent_dir*, unless it already is.

        The configured Builder, not the CodeEditBuilder: a Builder that writes no architecture doc
        must not have one written for it by a component the config did not name.
        """
        if (agent_dir / "architecture.md").exists():
            return
        await self._new_builder(ctx=ctx, dataset=ctx.datasets["train"], config=config).describe(agent_dir)

    async def _evaluate_validation_candidates(
        self,
        *,
        ctx: StrategyContext,
        candidates: list[Candidate],
    ) -> dict[str, EvaluationResult]:
        """Evaluate candidates on the validation split; skip any that already have a reward."""
        pending = [c for c in candidates if "validation" not in c.rewards]
        if not pending:
            return {}
        splits = frozenset({"validation"})
        restore_heldout_splits(self.working_dir, splits=splits)
        try:
            candidate_results = await asyncio.gather(*[self._evaluate_agent(ctx, c, "validation") for c in pending])
        finally:
            ensure_heldout_hidden(self.working_dir, splits=splits)
        return {candidate.label: result for candidate, result in candidate_results}

    def _check_proposer_builder_pairing(self, config: EvolutionaryOptimizerConfig) -> None:
        """Fail before the run spends anything if no proposal could ever be built.

        A Proposer emitting only kinds the Builder rejects produces empty rounds that
        look like a run doing work. Only checked when the Proposer declares what it
        emits; an undeclared one is still caught per proposal, a round at a time.
        """
        produces = cast("type[roles.Proposer]", resolve("proposer", config.proposer)).produces
        accepts = cast("type[roles.Builder]", resolve("builder", config.builder)).accepts
        if produces and not produces & accepts:
            raise ValueError(
                f"proposer {config.proposer!r} emits {sorted(produces)} but builder "
                f"{config.builder!r} accepts {sorted(accepts) or 'nothing'}; no proposal could be built"
            )

    async def _evaluate_train_candidates(
        self,
        *,
        ctx: StrategyContext,
        max_train_batch_tasks: int | None,
        train_batch_seed: int,
        survivors: list[Candidate],
        round_num: int,
    ) -> dict[str, EvaluationResult]:
        """Evaluate survivors on the train split.

        In full-dataset mode (``max_train_batch_tasks is None``) the eval set is
        identical every round, so a survivor's cached ``train_reward`` is reused and
        only survivors without one are evaluated. In batch mode the sampled batch
        changes per round, so cached rewards are not comparable within a round: every
        survivor is re-evaluated on the same freshly sampled batch.
        """
        dataset = ctx.datasets["train"]
        if max_train_batch_tasks is None:
            # Full-dataset mode: the eval set is identical every round, so evaluate only
            # survivors that lack a reward and reuse each survivor's cached train reward.
            pending = [s for s in survivors if "train" not in s.rewards]
            evaluated: list[tuple[Candidate, EvaluationResult]] = []
            if pending:
                evaluated = await asyncio.gather(*[self._evaluate_agent(ctx, c, "train") for c in pending])
            results = {candidate.label: result for candidate, result in evaluated}
            for survivor in survivors:
                if survivor.label in results:
                    continue
                if "train" not in survivor.rewards:
                    continue
                results[survivor.label] = EvaluationResult(
                    id=f"{survivor.label}-train",
                    aggregate_metrics=survivor.rewards["train"].metrics or {},
                    trials=list(survivor.rewards["train"].trials or []),
                )
            return results

        # Batch mode: the sampled batch changes per round, so a cached reward computed on
        # a different task set is not comparable within a round. Re-evaluate every survivor
        # on the same freshly sampled batch instead of reusing cached rewards.
        if max_train_batch_tasks <= 0:
            raise ValueError("max_train_batch_tasks must be positive or None")
        all_task_ids = sorted(task.id for task in dataset.list_tasks())
        if not all_task_ids:
            raise ValueError(f"No train tasks found in dataset {dataset.id!r}")
        rng = random.Random(train_batch_seed + round_num)
        batch_size = min(max_train_batch_tasks, len(all_task_ids))
        task_ids = sorted(rng.sample(all_task_ids, batch_size))
        evaluated = await asyncio.gather(*[self._evaluate_agent(ctx, c, "train", task_ids=task_ids) for c in survivors])
        return {candidate.label: result for candidate, result in evaluated}

    async def _analyze_round(
        self,
        *,
        analysis_dir: Path,
        dataset: Dataset,
        evaluations: dict[str, EvaluationResult],
        survivors: list[Candidate],
        round_num: int,
        config: EvolutionaryOptimizerConfig,
        load_trace: TraceLoader | None = None,
        ethos_path: Path | None = None,
    ) -> str:
        """Run TraceRootCauseAnalyzer per survivor, merge analyses, persist to disk.
        ``None`` those traces are skipped (local ``file://`` traces still load).

        Returns the merged analysis markdown string.
        """
        analysis_path = analysis_dir / f"round-{round_num}.md"
        if analysis_path.exists():
            return analysis_path.read_text()
        if config.analyzer is None:
            # Still written, empty. This file is the strategy's resume marker as much as
            # its analysis: `_detect_last_round` globs for it, so skipping the write on a
            # diagnosis-blind run leaves a restart unable to tell which rounds finished —
            # it starts from zero and every earlier cohort stays alive in the store.
            analysis_path.parent.mkdir(parents=True, exist_ok=True)
            analysis_path.write_text("")
            return ""

        per_agent = await asyncio.gather(
            *[
                get_component(
                    "root-cause-analyzer",
                    config.analyzer,
                    workspace=self.working_dir,
                    config=config.analyzer_config,
                    framework_skills_dirs=self._framework_skills_dirs,
                    load_trace=load_trace,
                ).run(
                    candidate=s,
                    dataset=dataset,
                    evaluation=evaluations[s.label],
                    peer_evaluations={k: v for k, v in evaluations.items() if k != s.label},
                    round_num=round_num,
                    ethos=ethos_path,
                )
                for s in survivors
            ]
        )
        analysis = await self.merge_analysis(survivors, round_num, [str(a) for a in per_agent])
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_path.write_text(analysis)
        return analysis

    async def _propose_improvements(
        self,
        *,
        analysis: str,
        evolution_tree: EvolutionTree,
        round_num: int,
        phase: Literal["exploration", "exploitation"],
        config: EvolutionaryOptimizerConfig,
    ) -> list[Proposal]:
        """Run Proposer; return up to ``config.max_candidates`` build requests."""
        proposer = get_component(
            "proposer",
            config.proposer,
            workspace=self.working_dir,
            config=config.proposer_config,
            objective_metrics=config.objective_function,
            regression_metrics=config.regression_metrics,
            framework_skills_dirs=self._framework_skills_dirs,
        )
        return await proposer.run(
            analysis=analysis,
            candidates=[n.candidate for n in evolution_tree.nodes.values()],
            round_num=round_num,
            max_candidates=config.max_candidates,
            hint=phase,
        )

    async def _build_candidates(
        self,
        *,
        ctx: StrategyContext,
        dataset: Dataset,
        proposals: list[Proposal],
        generation: int,
        config: EvolutionaryOptimizerConfig,
    ) -> list[Candidate]:
        """Build each proposal with the CodeEditBuilder and commit the ones that succeed.

        A failed build produces no Candidate at all: the proposal is discarded and its
        forked directory is left behind as scratch. That is the point of committing only
        after the build validates — there is no half-finished record to resurrect on
        resume, and no killed marker to remember to write.

        The Builder owns the whole span, forking included, so every way a build can fail
        — an ancestor whose artifact has gone, a smoke eval that will not pass — arrives
        here the same way: the Builder raised. One bad proposal is not a reason to end a
        run that has already spent hours, so it is logged and dropped.
        """
        builder_cls = cast("type[roles.Builder]", resolve("builder", config.builder))
        buildable = [proposal for proposal in proposals if proposal.kind in builder_cls.accepts]
        for proposal in proposals:
            if proposal.kind not in builder_cls.accepts:
                logger.warning(
                    "No builder for a %r proposal: %r accepts %s. Dropping %r.",
                    proposal.kind,
                    config.builder,
                    sorted(builder_cls.accepts) or "nothing",
                    proposal.description,
                )

        outcomes = await asyncio.gather(
            *[
                self._new_builder(ctx=ctx, dataset=dataset, config=config).build(ctx, proposal, generation=generation)
                for proposal in buildable
            ],
            return_exceptions=True,
        )

        # Cancellation is not a build failure and must not be swallowed as one:
        # `CancelledError` derives from BaseException, so the filter below would let a
        # cancelled candidate through as if it had built, and it would go on to be
        # evaluated and ranked. Re-raise so the whole round unwinds instead.
        for outcome in outcomes:
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome

        built: list[Candidate] = []
        for proposal, outcome in zip(buildable, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                # `return_exceptions=True` means a failure arrives as a value, not a
                # raise, so the reason is only ever seen if it is logged here. A build
                # that never became a candidate is the one thing a run cannot cheaply
                # reproduce, so keep the exception and its traceback.
                logger.warning(
                    "Build failed for proposal %r — %s: %s",
                    proposal.description,
                    type(outcome).__name__,
                    outcome,
                    exc_info=outcome,
                )
                continue
            built.append(outcome)
        return built

    def _terminator(self, ctx: StrategyContext, config: EvolutionaryOptimizerConfig) -> roles.Terminator:
        """Resolve this run's terminator, with the run's own convergence window."""
        terminator = config.terminator
        if terminator is None:
            raise ValueError("terminator must be configured before resolving the terminator component")
        return cast(
            "roles.Terminator",
            ctx.component(
                "terminator",
                terminator,
                config=config.terminator_config,
                min_rounds_before_stopping=config.min_rounds_before_stopping,
                objective_metrics=config.objective_function,
                regression_metrics=config.regression_metrics,
            ),
        )

    def _trajectory_scorer(self, ctx: StrategyContext, config: EvolutionaryOptimizerConfig) -> roles.TrajectoryScorer:
        """Resolve this run's trajectory scorer, with the run's own per-round task cap."""
        trajectory_scorer = config.trajectory_scorer
        if trajectory_scorer is None:
            raise ValueError("trajectory_scorer must be configured before resolving the trajectory scorer component")
        return cast(
            "roles.TrajectoryScorer",
            ctx.component(
                "trajectory-scorer",
                trajectory_scorer,
                config=config.trajectory_scorer_config,
                max_trajectory_tasks=config.max_trajectory_tasks,
                framework_skills_dirs=self._framework_skills_dirs,
            ),
        )

    def _selector(self, config: EvolutionaryOptimizerConfig) -> roles.Selector:
        """Resolve this run's selector."""
        return get_component(
            "selector",
            config.selector,
            config=config.selector_config,
            objective_metrics=config.objective_function,
            regression_metrics=config.regression_metrics,
        )

    def _new_builder(
        self, *, ctx: StrategyContext, dataset: Dataset, config: EvolutionaryOptimizerConfig
    ) -> roles.Builder:
        """Resolve and construct this run's Builder, one per build.

        One instance per build because a Builder is stateful — the CodeEditBuilder holds a shell
        session and a todo list — and builds run concurrently.

        """
        return get_component(
            "builder",
            config.builder,
            workspace=self.working_dir,
            config=self._coder_config(config),
            framework_skills_dirs=self._framework_skills_dirs,
            evaluator=ctx.outcome_evaluator,
            dataset=dataset,
            source_path=config.source.source_path,
            entrypoint=config.source.entrypoint,
        )

    async def _finalize(self, *, evolution_tree: EvolutionTree, selector: roles.Selector) -> Candidate | None:
        """Pick the winner and write this strategy's own report; return the winner.

        Restoring held-out splits, copying the winner out, closing the run entity and
        the Insight-suite report sections are the runner's.
        """
        best = selector.winner([n.candidate for n in evolution_tree.nodes.values()])
        if best is None:
            logger.warning("[FINAL] no candidates to finalize")
            return None

        evolution_tree.mark_best(best.id or best.label)
        try:
            # The report writer reads the workspace, which files agents by display handle.
            await self.write_final_report(best.label)
        except Exception as exc:  # noqa: BLE001 - the runner falls back to a compact summary
            logger.warning(f"[FINAL] Failed to write final report: {exc}")
        return best
