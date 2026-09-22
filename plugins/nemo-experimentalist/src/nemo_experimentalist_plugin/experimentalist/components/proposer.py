# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Any, ClassVar, Literal, cast, get_args

from nemo_experimentalist_plugin.entities import Candidate, MetricTarget, Proposal, local_path_from_uri
from nemo_experimentalist_plugin.experimentalist import roles
from nemo_experimentalist_plugin.experimentalist.components.models import (
    OptimizationType,
)
from nemo_helix_plugin.nooa_model_client import get_default_model, get_fast_model
from nooa import Agent, CodeActStrategy, strategy
from nooa.agentdoc import doc, spec
from nooa.agents import TokenBudgetSummarizer
from nooa.config import CodeActConfig
from nooa.config.summarizer_config import TokenBudgetConfig
from nooa.skill_registry import SkillRegistry
from pydantic import BaseModel, Field

from .cards import Optimize
from .tools import WorkspaceTool
from .util import load_framework_skills

logger = logging.getLogger(__name__)


#: What this Proposer produces and the built-in CodeEditBuilder accepts. An opaque
#: discriminator, not a global enumeration — it only has to match a Builder.
CODE_CHANGE = "code-change"


class CodeChange(BaseModel):
    """The ``code-change`` payload schema, owned by this Proposer and the CodeEditBuilder.

    Nothing here belongs on ``Candidate``: it is what the Builder was *asked* to
    produce, which the committed Candidate keeps as ``generated_from`` provenance
    alongside the artifact it actually produced.
    """

    root_cause: str = Field(
        min_length=1,
        description=(
            "The diagnosed cause of poor performance from the analysis — WHY the agent fails, "
            "not what would fix it. Must complete: 'The agent underperforms because...' "
            "Do not mention the proposed change or any remedy here."
        ),
    )
    optimization_type: OptimizationType
    task_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Task ids from the analysis that most directly exercise this root cause — "
            "the concrete tasks the agent fails (or barely passes) for this reason. "
            "The coder uses these to validate the fix, so pick tasks whose failure "
            "evidence in the analysis is caused by `root_cause`. 2-3 is typical."
        ),
    )


class Improvement(BaseModel):
    """One proposed change, as the LLM returns it, before it becomes a Proposal."""

    ancestor: str = Field(description="Candidate id to branch from.")
    optimization: str = Field(
        min_length=1,
        description=(
            "Graph-level description of the change in terms of the architecture diagram: "
            "which node is added, removed, or modified; which edge changes; which prompt is rewritten. "
            "Must NOT reference source file paths, line numbers, or implementation details."
        ),
    )
    root_cause: str = Field(
        min_length=1,
        description=(
            "The diagnosed cause of poor performance from the analysis — WHY the agent fails, "
            "not what would fix it. Must complete: 'The agent underperforms because...' "
            "Do not mention the proposed change or any remedy here."
        ),
    )
    optimization_type: OptimizationType
    task_ids: list[str] = Field(
        default_factory=list,
        description="Task ids whose failure evidence is caused by `root_cause`.",
    )

    def as_proposal(self) -> Proposal:
        """The Layer A message a Builder receives, with this pair's payload inside it."""
        return Proposal(
            ancestor=self.ancestor,
            description=self.optimization,
            kind=CODE_CHANGE,
            payload=CodeChange(
                root_cause=self.root_cause,
                optimization_type=self.optimization_type,
                task_ids=self.task_ids,
            ).model_dump(),
        )


class ProposerConfig(BaseModel):
    """Configuration for Proposer tuning parameters."""

    max_summary_tokens: int = Field(
        default=80_000,
        description="Max tokens the token-budget summarizer may use.",
    )


class CodeChangeProposer(Agent, roles.Proposer):
    """Propose the next round's isolated optimization candidates."""

    config_type = ProposerConfig

    name = "code-change"
    produces: ClassVar[frozenset[str]] = frozenset({CODE_CHANGE})

    def __init__(
        self,
        workspace: Path,
        config: ProposerConfig | None = None,
        framework_skills_dirs: list[Path] | None = None,
        objective_metrics: list[MetricTarget] | None = None,
        regression_metrics: list[MetricTarget] | None = None,
        **kwargs: Any,
    ):
        super().__init__(llm=kwargs.pop("llm", None) or get_default_model(), **kwargs)
        self._config = config or ProposerConfig()
        self._objective_metrics = objective_metrics or []
        self._regression_metrics = regression_metrics or []
        self._workspace_path = workspace.resolve()
        self.workspace = WorkspaceTool(workspace=self._workspace_path)
        self.context["workspace_tool_documentation"] = doc(WorkspaceTool)

        self.skills: SkillRegistry = SkillRegistry(self)
        spec(self, "skills", hidden=True)
        load_framework_skills(self.skills, framework_skills_dirs or [])

        self.optimize = Optimize()
        TokenBudgetSummarizer.install(
            self,
            llm=get_fast_model(),
            config=TokenBudgetConfig(max_tokens=self._config.max_summary_tokens),
        )

    @staticmethod
    def _evolution_table(candidates: list[Candidate]) -> str:
        """Render the population as the markdown table the prompt reads."""
        if not candidates:
            return "(none yet — this is the first round)"
        rows = ["| agent | generation | ancestor | validation reward |", "| --- | --- | --- | --- |"]
        for c in sorted(candidates, key=lambda c: (c.generation, c.label)):
            reward = c.rewards["validation"].metrics or {}
            ancestor = next((o.label for o in candidates if o.id == c.ancestor), "—")
            rows.append(f"| {c.label} | {c.generation} | {ancestor} | {reward or '—'} |")
        return "\n".join(rows)

    async def run(
        self,
        *,
        analysis: str,
        candidates: list[Candidate],
        round_num: int,
        max_candidates: int,
        hint: str | None = None,
    ) -> list[Proposal]:
        """Return up to max_candidates targeted improvement proposals.

        Args:
            analysis: merged round analysis markdown with root causes already enumerated.
            candidates: the scored population to propose against; lineage is on each one.
            round_num: the round these Proposals are for.
            max_candidates: maximum number of Proposals to return.
            hint: optional steer from the strategy, e.g. to explore rather than refine.

        Returns:
            list[Proposal]: up to max_candidates build requests, each routed to a
            Builder by its ``kind``.

        """
        all_types = set(get_args(OptimizationType))
        tried_types = sorted(
            {
                kind
                for c in candidates
                if (kind := (c.generated_from.payload or {}).get("optimization_type")) in all_types
            }
        )
        available_types = sorted(all_types - set(tried_types))
        proposal_survivors = [c for c in candidates if c.killed_generation is None]

        survivor_context: list[dict[str, Any]] = []
        for s in proposal_survivors:
            # Through the artifact the record addresses, never `agents/<label>/`: the
            # label is a display handle and nothing may derive storage from it.
            arch_text = f"(architecture.md missing for {s.label})"
            meta: dict[str, Any] = {}
            try:
                candidate = self.workspace.get_metadata(s.label)
                meta = candidate.slim().model_dump()
                artifact = local_path_from_uri(candidate.artifact.uri)
                artifact = artifact if artifact.is_dir() else artifact.parent
                arch_text = (artifact / "architecture.md").read_text()
            except Exception:  # noqa: BLE001 - a survivor without a readable doc is still proposable
                pass
            survivor_context.append(
                {
                    "id": s.id,
                    "label": s.label,
                    "reward": s.rewards["validation"].metrics or {},
                    "trajectory_reward": s.rewards["validation-trajectory"].metrics or {},
                    "metadata": meta,
                    "architecture": arch_text,
                }
            )

        improvements = await self._run_with_context(
            analysis=analysis,
            evolution_history=self._evolution_table(candidates),
            tried_types=tried_types,
            available_types=available_types,
            survivors=survivor_context,
            cards_index=doc(self.optimize),
            phase=cast("Literal['exploration', 'exploitation']", hint or "exploration"),
            max_candidates=max_candidates,
            objective_metrics=[t.model_dump() for t in self._objective_metrics],
            regression_metrics=[t.model_dump() for t in self._regression_metrics],
        )
        # allowed_types is every type, not just the untried ones (#1163). available_types
        # still reaches the prompt, so novelty stays a *preference*: a type that worked in
        # an earlier round can be used again when it is the right tool.
        usable = self._usable_improvements(
            improvements,
            known_ancestors={c.id for c in proposal_survivors},
            allowed_types=all_types,
        )
        kept = self._validate_improvements(improvements=usable, max_candidates=max_candidates)
        return [improvement.as_proposal() for improvement in kept]

    @staticmethod
    def _usable_improvements(
        improvements: list[Improvement], *, known_ancestors: set[str], allowed_types: set[str]
    ) -> list[Improvement]:
        """Keep the improvements a Builder could actually build, dropping the rest.

        Two ways a model gets this wrong, and neither is worth a run. ``ancestor`` is a
        candidate id while survivors carry both id and label, so a plausible "agent-2" is
        exactly what comes back. And ``optimization_type`` has twenty valid values, so a
        near-miss like "edit_method" for "edit_concrete_method" is equally likely.

        Both checks run *after* the CodeAct loop, so raising buys no retry: it unwinds
        through the strategy and ends a run that has already spent hours. Same policy as
        the Builder — one bad proposal is not a bad round.
        """
        usable: list[Improvement] = []
        for improvement in improvements:
            if improvement.ancestor not in known_ancestors:
                logger.warning(
                    "Dropping a proposal whose ancestor %r is not a survivor id (%s): %s",
                    improvement.ancestor,
                    sorted(known_ancestors),
                    improvement.optimization,
                )
            elif improvement.optimization_type not in allowed_types:
                logger.warning(
                    "Dropping a proposal with unusable optimization_type %r (allowed: %s): %s",
                    improvement.optimization_type,
                    sorted(allowed_types),
                    improvement.optimization,
                )
            else:
                usable.append(improvement)
        if improvements and not usable:
            raise ValueError(
                f"None of the Proposer's {len(improvements)} improvements were usable: every one named "
                f"an ancestor outside {sorted(known_ancestors)} or a type outside {sorted(allowed_types)}"
            )
        return usable

    @staticmethod
    def _validate_improvements(*, improvements: list[Improvement], max_candidates: int) -> list[Improvement]:
        """Reject a batch that is malformed as a whole, and return the ones worth building.

        Deduplicates by optimization text and truncates to the round's budget — the cut
        falls on the kept list, since a surplus improvement past it may be the only
        usable one.

        A repeated ``optimization_type`` is fine (#1163): there are twenty of them and
        two genuinely different changes to the same method share one, so it is not
        evidence of a malformed batch. Duplicate *text* is, and is dropped below.
        """
        if not improvements:
            raise ValueError("Proposer returned no improvements")

        kept: list[Improvement] = []
        seen_descriptions: set[str] = set()
        for improvement in improvements:
            description = improvement.optimization.strip()
            if description in seen_descriptions:
                logger.warning("dropping improvement with duplicate optimization text: %r", description)
                continue
            seen_descriptions.add(description)
            kept.append(improvement)

        if not kept:
            raise ValueError(
                f"Proposer returned {len(improvements)} improvements, none of them usable; "
                "see the warnings above for why each was dropped"
            )

        # Truncate last: a surplus improvement past the cut may be the only usable
        # one, so the cut has to fall on the kept list rather than the raw one.
        if len(kept) > max_candidates:
            logger.warning(
                "Proposer returned %d usable improvements; keeping the first %d",
                len(kept),
                max_candidates,
            )
            kept = kept[:max_candidates]
        return kept

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=20, cell_timeout=3600.0)))
    async def _run_with_context(
        self,
        analysis: str,
        evolution_history: str,
        tried_types: list[str],
        available_types: list[str],
        survivors: list[dict[str, Any]],
        cards_index: str,
        phase: Literal["exploration", "exploitation"],
        max_candidates: int,
        objective_metrics: list[dict[str, Any]],
        regression_metrics: list[dict[str, Any]],
    ) -> list[Improvement]:
        """Pick up to `max_candidates` targeted improvements grounded in root causes.

        Args:
        - analysis (str): round analysis with root causes already enumerated
        - evolution_history (str): markdown table of prior rounds, for context
        - tried_types (list[str]): optimization_types already attempted
        - available_types (list[str]): types not yet tried — PREFER THESE
        - survivors (list[dict]): each {id, metrics, trajectory_reward, metadata,
          architecture}; these are your branching candidates with their architecture.md
          already loaded
        - cards_index (str): pre-rendered `doc(self.optimize)` showing the card index.
          Load a specific card on demand via `print(doc(self.optimize.<name>))`.
        - phase (Literal): "exploration" = novel directions; "exploitation" = improve current best
        - max_candidates (int): max number of Improvements to return (typically 3)
        - objective_metrics (list[dict]): evaluator dimensions to improve
        - regression_metrics (list[dict]): evaluator dimensions to preserve

        Returns:
        - list[Improvement]: up to `max_candidates` Improvements.

        ## Working level
        You are a graph-level architect. Reason exclusively from the architecture
        diagrams in `survivors[*].architecture` — do NOT read source files or
        inspect implementation details. Your job is to identify WHAT changes at
        the diagram level, not HOW it is implemented in code.

        ## Per-improvement requirements
        For each Improvement you propose:
        1. Identify ONE root cause from the analysis that limits an active objective:
           the specific reason the agent
           underperforms, stated as a diagnosis ("The agent fails because X is absent /
           misconfigured / too vague"). Do NOT include the proposed remedy here — that
           belongs in `optimization`. If you cannot articulate the failure cause
           independently of the fix, drop the hypothesis.
        2. Pick the ancestor from `survivors` most affected by that root cause.
        3. Load the matching card with `doc(self.optimize.<name>)` and pick ONE
           optimization_type that the card covers, preferring `available_types`.
        4. Read the ancestor's architecture diagram (in `survivors[*].architecture`)
           to understand the current graph shape. If the change touches a skill,
           find it in the diagram — do NOT open source files.
        5. Write `root_cause` (pure diagnosis — why the agent fails, no remedy)
           and `optimization` (the graph-level change that addresses it). These
           must be independently readable: `root_cause` explains the failure,
           `optimization` describes the diagram delta (node added/removed/modified,
           edge changed, prompt rewritten). No file paths, no line numbers, no
           code snippets in either field.
        6. Set `task_ids` to the tasks in the analysis whose failure evidence is
           caused by this `root_cause` — the concrete tasks the coder should use to
           validate the fix. Pick 2-3 tasks the ancestor actually fails (or barely
           passes) for this reason; do not pad with unrelated passing tasks.

        ## Metric contract
        - Optimize these objective metrics: `objective_metrics`. Every proposed
          change must have a concrete, evidence-based path to improving at least one
          of these dimensions.
        - Preserve these regression metrics: `regression_metrics`. They are
          guardrails, not proposal targets: do not choose a root cause or frame an
          optimization solely around improving a regression metric. Instead, ensure
          the proposed objective improvement does not sacrifice them.
        - Evaluator metric values are authoritative. Do not invent an aggregate,
          scalarization, weighting, or threshold.

        ## Branching rules
        - Branch from any survivor — usually the top scorer, but a lower-scoring
          ancestor is the right base when it uniquely passes tasks the top scorer fails.
        - If two survivors have complementary failures (each passes what the other
          fails), propose a cross-pollination improvement that merges their approaches.

        ## Phase
        - exploration: novel directions, even speculative ones
        - exploitation: refine the current best survivor. When the obvious targeted
          improvements have already been tried, do NOT stop — take the best remaining
          direction, an untried type when one fits and a tried one when it does not.

        ## MANDATORY
        - Return between 1 and max_candidates Improvements. NEVER return [].
        - Each Improvement.optimization_type MUST be in `available_types` or
          `tried_types`. Prefer `available_types`; reuse a tried type when it is
          the tool that actually addresses the root cause. A type that worked in
          an earlier round is not spent.
        - Each improvement in a round should target a different degree of freedom;
          two improvements touching the same axis are only allowed when no other
          viable direction exists.
        - Each Improvement must make exactly ONE targeted change. Do not bundle
          prompt edits with code edits, config changes with tool additions, cleanup
          with behavior changes, or multiple unrelated fixes in one candidate.
        - `optimization` MUST be a graph-level description. Reject any draft that
          names a source file, a line number, or any code construct.
        """
        ...
