# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nemo_experimentalist_plugin.entities import Dataset
from nemo_helix_plugin.nooa_model_client import get_fast_model
from nooa import Agent, CodeActStrategy, strategy
from nooa.agentdoc import doc, spec
from nooa.agents import TokenBudgetSummarizer
from nooa.config import CodeActConfig
from nooa.config.summarizer_config import TokenBudgetConfig
from nooa.skill_registry import SkillRegistry
from nooa.tools import Match, ShellTools, TodoManager
from pydantic import BaseModel, Field, model_validator

from .tools import WorkspaceTool
from .util import load_framework_skills


class GoalTreeConfig(BaseModel):
    """Configuration for goal tree generation and validation.

    Enforces hierarchical depth and node count constraints during tree generation.
    """

    max_depth: int = Field(default=3, gt=0, description="Maximum tree depth allowed")
    max_initial_depth: int = Field(
        default=2,
        gt=0,
        description="Maximum depth for initial tree generation",
    )
    min_initial_nodes: int = Field(
        default=3,
        gt=0,
        description="Minimum nodes required in initial tree",
    )
    max_initial_nodes: int = Field(
        default=4,
        gt=0,
        description="Maximum nodes allowed in initial tree",
    )

    @model_validator(mode="after")
    def validate_constraints(self) -> GoalTreeConfig:
        """Validate that depth and node constraints are internally consistent.

        Returns:
            GoalTreeConfig: the validated config instance.

        Raises:
            ValueError: if ``max_initial_depth`` exceeds ``max_depth`` or
                ``min_initial_nodes`` exceeds ``max_initial_nodes``.

        """
        if self.max_initial_depth > self.max_depth:
            raise ValueError(f"max_initial_depth ({self.max_initial_depth}) cannot exceed max_depth ({self.max_depth})")
        if self.min_initial_nodes > self.max_initial_nodes:
            raise ValueError(
                f"min_initial_nodes ({self.min_initial_nodes}) cannot exceed "
                f"max_initial_nodes ({self.max_initial_nodes})"
            )
        return self

    def validate_tree(self, tree: GoalTree) -> GoalTree:
        """Validate a goal tree against this config's constraints.

        Args:
            tree: The goal tree to validate.

        Returns:
            GoalTree: the validated tree, unchanged.

        Raises:
            ValueError: if the tree violates depth or node count constraints.

        """

        def visit(node: GoalNode, depth: int) -> None:
            if depth > self.max_depth:
                raise ValueError(f"goal tree exceeds max depth {self.max_depth} at node {node.id!r}")
            if node.added_at_generation is None and depth > self.max_initial_depth:
                raise ValueError(
                    f"initial goal tree nodes exceed max depth {self.max_initial_depth} at node {node.id!r}"
                )
            for child in node.children:
                visit(child, depth + 1)

        visit(tree.root, depth=1)
        initial_nodes = [node for node in tree._iter_nodes() if node.added_at_generation is None]
        if not (self.min_initial_nodes <= len(initial_nodes) <= self.max_initial_nodes):
            raise ValueError(
                f"initial goal tree must have {self.min_initial_nodes}-"
                f"{self.max_initial_nodes} nodes, got {len(initial_nodes)}"
            )
        return tree


class GoalNode(BaseModel):
    """A single node in the goal tree with a weighted objective and optional children."""

    id: str = Field(description="Stable kebab-case identifier, unique within the tree.")
    goal: str = Field(description="One sentence stating the observable behavior this node scores.")
    weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Weight within parent. Children weights sum to 1.0.",
    )
    children: list[GoalNode] = Field(default_factory=list)

    added_at_generation: int | None = None
    added_because: str | None = None
    added_by: str | None = None


class GoalTree(BaseModel):
    """A hierarchical rubric describing the sub-capabilities required to complete a task."""

    task_id: str = Field(description="Identifier of the task this tree scores.")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 UTC timestamp when this tree was generated.",
    )
    frozen: bool = Field(default=True)
    root: GoalNode

    @model_validator(mode="after")
    def validate_structure(self) -> GoalTree:
        """Validate tree structural invariants.

        Invariants:
          - Root weight is 1.0.
          - Children weights at each non-leaf node sum to 1.0 (within 1e-3).
          - Node IDs are unique across the tree.

        Returns:
            GoalTree: the validated tree instance.

        Raises:
            ValueError: if root weight is not 1.0, any node ID is duplicated,
                any node has an empty goal, or sibling weights do not sum to 1.0.

        """
        if abs(self.root.weight - 1.0) > 1e-3:
            raise ValueError(f"root weight must be 1.0, got {self.root.weight}")

        seen_ids: set[str] = set()

        def visit(node: GoalNode) -> None:
            if node.id in seen_ids:
                raise ValueError(f"duplicate node id {node.id!r}")
            seen_ids.add(node.id)
            if not node.goal.strip():
                raise ValueError(f"node {node.id!r} has empty goal")
            if node.children:
                child_weight_sum = sum(c.weight for c in node.children)
                if abs(child_weight_sum - 1.0) > 1e-3:
                    raise ValueError(
                        f"children of {node.id!r} have weights summing to {child_weight_sum:.3f}; expected 1.0"
                    )
            for child in node.children:
                visit(child)

        visit(self.root)
        return self

    def _iter_nodes(self) -> list[GoalNode]:
        """Return all nodes in the tree in depth-first order.

        Returns:
            list[GoalNode]: all nodes from root to leaves, depth-first.

        """

        def visit(node: GoalNode) -> list[GoalNode]:
            return [node, *[desc for child in node.children for desc in visit(child)]]

        return visit(self.root)

    def to_json(self) -> str:
        """Return the goal tree serialized as a JSON string.

        Returns:
            str: a pretty-printed JSON representation of this tree.

        """
        return json.dumps(self.model_dump(), indent=2)

    @classmethod
    def from_path(
        cls,
        path: Path,
        config: GoalTreeConfig | None = None,
    ) -> GoalTree:
        """Load and validate a goal tree from a JSON file.

        Args:
            path: Path to the goal tree JSON file.
            config: Configuration with validation constraints. If None, uses defaults.

        Returns:
            GoalTree: the loaded and validated goal tree.

        Raises:
            FileNotFoundError: if ``path`` does not exist.
            ValueError: if the JSON fails validation or violates config constraints.

        """
        if config is None:
            config = GoalTreeConfig()
        tree = cls.model_validate(json.loads(path.read_text()))
        return config.validate_tree(tree)


def traverse_tree(node: GoalNode) -> list[GoalNode]:
    """Return all leaf nodes reachable from the given node (depth-first).

    Args:
        node: the root of the subtree to traverse.

    Returns:
        list[GoalNode]: leaf nodes in depth-first order.

    """
    if not node.children:
        return [node]
    leaves: list[GoalNode] = []
    for child in node.children:
        leaves.extend(traverse_tree(child))
    return leaves


def leaf_weights_by_id(node: GoalNode, cumulative_weight: float = 1.0) -> dict[str, float]:
    """Return a mapping of leaf node IDs to their cumulative path weights.

    Args:
        node: the root of the subtree to process.
        cumulative_weight: accumulated weight from ancestor nodes; defaults to 1.0.

    Returns:
        dict[str, float]: mapping from leaf node ID to its cumulative path weight.

    """
    node_weight = cumulative_weight * node.weight
    if not node.children:
        return {node.id: node_weight}
    weights: dict[str, float] = {}
    for child in node.children:
        weights.update(leaf_weights_by_id(child, node_weight))
    return weights


def find_node(root: GoalNode, node_id: str) -> GoalNode | None:
    """Return the first node with the given ID, or None if not found.

    Args:
        root: the root node to search from.
        node_id: the node ID to locate.

    Returns:
        GoalNode | None: the matching node, or None if not present in the subtree.

    """
    if root.id == node_id:
        return root
    for child in root.children:
        found = find_node(child, node_id)
        if found:
            return found
    return None


GoalNode.model_rebuild()


class GoalTreeGenerator(Agent):
    """Generates the goal tree consumed by the trajectory scorer.

    The generator MUST NOT see the agent's implementation, harbor wrapper, or any
    prior trace. It works only from scoring-side context (use case description,
    verifier behavior, sample ground truth), so the tree describes what the *task*
    requires rather than what the *current implementation* does.
    """

    def __init__(
        self,
        workspace: Path,
        config: GoalTreeConfig,
        framework_skills_dirs: list[Path] | None = None,
        **kwargs: Any,
    ):
        """Initialize the goal tree generator.

        Args:
            workspace: the root directory containing use-case data and skills.
            config: the constraints and validation rules for generated trees.
            framework_skills_dirs: Optional list of directories containing framework skills to load.
            **kwargs: additional arguments passed to the parent Agent.

        """
        super().__init__(llm=kwargs.pop("llm", None) or get_fast_model(), **kwargs)
        self._config = config
        self._workspace_path = workspace.resolve()

        self.shell = ShellTools(cwd=str(workspace))
        self.workspace = WorkspaceTool(workspace=workspace)
        self.todos = TodoManager()
        self.context["file_match"] = doc(Match)
        self.skills: SkillRegistry = SkillRegistry(self)
        spec(self, "skills", hidden=True)
        load_framework_skills(self.skills, framework_skills_dirs or [])
        TokenBudgetSummarizer.install(self, llm=get_fast_model(), config=TokenBudgetConfig(max_tokens=80_000))

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=40, cell_timeout=3600.0)))
    async def _generate(self, dataset: Dataset, ethos: Path | None = None) -> GoalTree:  # pyright: ignore[reportReturnType]
        """Generate a hierarchical goal tree describing what a successful agent run looks like.

        # Inputs you should consult

        - `ETHOS.md` in the workspace root. Read it via self.shell. This is the
          canonical description of the domain and what the agent is being asked to do.
        - 3 to 5 examples from the dataset. Use task inputs, visible
          resources, and metric specs to understand the task shape and scoring surface.

        # Inputs you MUST NOT consult

        - The agent's source code, harbor_wrapper.py, dind_environment.py, or any file under
          eval-and-optimize/agents/. The tree describes what the *task* requires, not what
          the *current implementation* does. Mirroring the implementation penalises
          candidates that deviate, defeating the purpose of trajectory scoring.
        - Any trace from a prior run (eval-and-optimize/results/).

        # Tree shape — phases, not outcome dimensions

        Produce a compact GoalTree that obeys the numeric limit arguments passed to
        this call:
        - total depth must be at most {self._config.max_initial_depth} for initial nodes.
        - total initial node count must be between {self._config.min_initial_nodes}
          and {self._config.max_initial_nodes}, including the root.
        - absolute depth must be at most {self._config.max_depth}.

        **Critical**: structure the children of root as **sequential execution phases** of a
        correct agent run, not as dimensions of the final output. The purpose is dense partial
        rewards for long traces: an agent that completes phases 1-2 of 3 but fails at phase 3
        should receive meaningful partial credit. Outcome-only nodes (label match, explanation quality) make
        this impossible because they award zero until the final answer is produced.

        - Root: one statement of the overall task goal. Weight = 1.0.
        - Children of root: the ordered execution stages a
          correct run passes through. Each phase is observable mid-trace (tool calls made,
          intermediate data retrieved, intermediate conclusions written). Assign weights
          proportional to how much of the task's difficulty the phase represents. They MUST
          sum to 1.0.
        - Do not add grandchildren unless {self._config.max_initial_depth} allows them and
          the extra structure is necessary. Every leaf should be scoreable from the trace.

        # Minimize nodes, maximize signal

        **Hard rule**: initial NODE count must stay within {self._config.min_initial_nodes}
        and {self._config.max_initial_nodes}, including the root.

        ## Signal test — apply to every proposed leaf

        Before keeping a leaf, ask: *"Would two agent variants on this task plausibly score
        differently on this criterion?"* If every plausibly-implemented agent will trivially
        pass (or trivially fail) the criterion, the leaf has zero signal — DROP IT.

        ## Anti-patterns — do NOT emit these leaves

        - **"Agent parses its input"** / "agent reads the --prompt CLI argument" / "agent
          parses the research question". Any agent that runs at all passes this; any agent
          that doesn't run produces no trace at all. Zero signal.
        - **"Agent writes the required final artifact"** / "the expected file exists
          after the run". The verifier already checks terminal artifact existence,
          schema, service state, or final answer validity. A separate leaf adds
          little unless it captures a meaningful intermediate step toward that artifact.
        - **"Agent does N X times"** - avoid hard constraints on the number of X.

        ## Good leaf examples

        Leaves should be where *judgment* differentiates agents:

        - "Agent's follow-up search queries refine or expand on the initial round (judged by
          whether the queries target gaps in the initial sources, not by mere string
          inequality)." → genuinely needs LLM judgment.
        - "Agent's report cites high-authority sources (academic, official documentation,
          peer-reviewed venues) over marketing/SEO content." → needs source-quality
          judgment.
        - "Agent's intermediate evidence-gathering output addresses the specific aspect
          required by the question, not just the topic keywords." → distinguishes broad vs
          focused research.
        - Do not duplicate the verifier's checks.

        Example phase decomposition for a research-and-classify task:
          1. gather-intelligence   — retrieve CVE advisory, SBOM, relevant docs
                                     → intermediate artifact: evidence notes file
          2. verify-and-assess     — confirm component presence and reachability
                                     → intermediate artifact: assessment summary
          3. synthesize-verdict    — derive status and label from the evidence

        Every node MUST have a non-empty goal that can be judged from observable trace
        events and filesystem artifacts:

        - Trace tool calls: "Retrieve the CVE advisory and package metadata before judging reachability."
        - Intermediate artifacts: "Write evidence notes before producing the final classification."
        - Terminal artifacts: "Produce a final report that parses as the expected schema."
        - Intermediate data logged: "Record the retrieved version before deciding affectedness."

        Avoid goals that require reading the agent's mind ("the agent intended to...") or
        that can only be evaluated after the final answer exists ("status matches ground truth").
        If you cannot describe a goal in observable mid-trace or filesystem terms, drop the
        node.

        # Hidden verifier boundary

        The verifier files you read are for rubric design only. They are often hidden from
        the task agent during Harbor/Terminal-Bench execution. Therefore:

        - Do NOT create a phase that requires the agent to read, find, list, inspect, run,
          execute, or invoke the resources of the task.
        - Do NOT assume the agent can see hidden expected outputs or benchmark tests.
        - If a useful phase involves checking work, phrase it as agent-visible validation:
          local commands, visible tests, program invocations, service probes, output-file
          reads, or artifact/schema checks derived from the prompt and visible workspace.
        - Use hidden verifier details only to choose better observable proxies, such as
          "run the implemented converter on visible sample input and inspect the output
          format" rather than "run the verifier".

        # IDs

        Use kebab-case identifiers stable across the run (e.g. "parse-input",
        "reason-about-cues", "emit-schema"). They appear in trajectory rewards and selection logs,
        so prefer descriptive over short.

        # Output

        Return a GoalTree(task_id=task_id, root=GoalNode(...)). The framework validates
        weights, depth, initial node count, and ID uniqueness; invalid trees are rejected and the call retries.
        """
        ...

    async def generate(
        self,
        dataset: Dataset,
        ethos: Path | None = None,
    ) -> GoalTree:
        """Generate and validate a goal tree. Caller is responsible for persistence.

        Returns:
            GoalTree: a freshly generated, structurally valid goal tree.

        """
        tree = await self._generate(dataset, ethos=ethos)
        validated_tree = GoalTree.model_validate(tree.model_dump())
        return self._config.validate_tree(validated_tree)

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=40, cell_timeout=1800.0)))
    async def _update(
        self,
        dataset: Dataset,
        goal_tree: GoalTree,
        analysis: str,
        round_num: int,
        ethos: Path | None = None,
    ) -> GoalTree:  # pyright: ignore[reportReturnType]  # ty: ignore[invalid-return-type]
        """Propose a reweighted goal tree informed by a round of agent analysis.

        # What you receive

        - `goal_tree`: the current frozen rubric. Inspect `goal_tree.root` and its
          children to understand what sub-capabilities are already measured.
        - `analysis`: the merged round analysis markdown. It contains the
          population-level Root Causes, Failure Patterns, and Mechanical Errors
          sections. Use systematic Root Causes as context for what went wrong,
          not as a checklist of nodes to add.
        - `round_num`: the current generation number. Any newly added node MUST
          set `added_at_generation=round_num` and `added_by="analyzer"`.

        # What to look for

        Start from selection and analysis utility, not root-cause coverage. Ask:
        "Is there one additional partial-reward signal that would materially help
        distinguish promising candidates, explain outcome/trajectory disagreements,
        or preserve a useful branch in the next selection step?"

        Prefer returning the tree unchanged when the existing nodes already give
        enough steering signal.

        # Constraints

        - Add at most **one** new leaf per call. Pick only the highest-leverage
          missing partial reward. Do not add one node per root cause.
        - Only add nodes for **systematic** failures (≥2 tasks). One-offs are noise.
        - Each new node must map to an **observable** goal (tool call present,
          output field exists, file written) — never "the agent understood X".
        - Do not duplicate existing nodes. Check all existing node IDs and goals
          before adding.
        - Keep the change minimal. Prefer a narrow leaf under the closest existing
          phase over a broad new top-level phase.
        - Return the **entire updated tree**, not just the new node. You MUST
          reweight every affected sibling set yourself so every non-leaf node's
          child weights sum to exactly 1.0. If you add a top-level phase, reweight
          all root children. If you add a child under a leaf, that new child is the
          only child and must have weight 1.0.
        - Preserve every existing node's id, goal, added_at_generation,
          added_because, added_by, and parent. You may change existing node
          weights only to make room for the new node.
        - Total depth must stay within {self._config.max_depth}.
        - If no missing steering signal exists, return the input tree unchanged.

        # Output

        Return a complete GoalTree. The framework validates weights, depth, and
        ID uniqueness; invalid trees are rejected and the call retries.
        """
        self.context["current_goal_tree"] = goal_tree.to_json()
        self.context["round_analysis"] = analysis
        self.context["round_num"] = round_num
        ...

    async def update(
        self,
        goal_tree: GoalTree,
        analysis: str,
        round_num: int,
        dataset: Dataset,
        ethos: Path | None = None,
    ) -> GoalTree:
        """Produce an updated, validated goal tree. Caller is responsible for persistence.

        Args:
            goal_tree: the current goal tree to refine.
            analysis: round analysis markdown used to inform reweighting.
            round_num: the current generation number, stamped on any new nodes.
            ethos: optional path to a materialized ETHOS.md file.

        Returns:
            GoalTree: the updated, structurally valid goal tree.

        """
        tree = await self._update(dataset, goal_tree, analysis, round_num, ethos=ethos)
        validated_tree = GoalTree.model_validate(tree.model_dump())
        return self._config.validate_tree(validated_tree)
