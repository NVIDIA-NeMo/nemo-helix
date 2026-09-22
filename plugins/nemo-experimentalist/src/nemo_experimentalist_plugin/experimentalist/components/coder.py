# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast  # noqa: D100, F401
import json
import random
import re  # noqa: F401
import shlex
import shutil
from collections import Counter, defaultdict  # noqa: F401
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nemo_experimentalist_plugin.entities import (
    Candidate,
    Dataset,
    EvaluationResult,
    Proposal,
    Task,
    TrialResult,
)
from nemo_experimentalist_plugin.experimentalist import roles
from nemo_experimentalist_plugin.experimentalist.components.evaluator import Evaluator
from nemo_experimentalist_plugin.experimentalist.seam import BuilderContext
from nemo_helix_plugin.nooa_model_client import get_default_model, get_fast_model
from nooa import Agent, CodeActStrategy, strategy
from nooa.agentdoc import doc, spec
from nooa.agents import TokenBudgetSummarizer
from nooa.config import CodeActConfig
from nooa.config.summarizer_config import TokenBudgetConfig
from nooa.skill import Skill
from nooa.skill_registry import SkillRegistry
from nooa.tools import Match, TodoManager
from pydantic import BaseModel, Field

from .cards import Optimize
from .proposer import CODE_CHANGE, CodeChange
from .tools import GuardedShellTools
from .util import load_framework_skills


class CodeEditBuilderConfig(BaseModel):
    """Store tuning parameters for CodeEditBuilder optimization."""

    max_summary_tokens: int = Field(
        default=80_000,
        description="Max tokens the token-budget summarizer may use.",
    )
    max_fix_attempts: int = Field(
        default=2,
        description="Max LLM repair iterations inside integration_check before giving up on a candidate.",
    )
    max_architecture_doc_iterations: int = Field(
        default=100,
        gt=0,
        description="Max CodeAct iterations allowed to write architecture.md. Raise it for agents with many source files.",
    )
    model_catalog_path: Path | None = Field(
        default=None,
        description="Optional YAML model catalog path overriding the packaged assets/models.yaml.",
    )


class IntegrationCheckFailed(RuntimeError):
    """Signal that a candidate cannot pass the integration check within the allotted repair budget."""


class ArchitectureSkill(Skill):
    """Guide to writing architecture documentation for an agent.

    An architecture documentation file is a human- and LLM-readable snapshot of an agent's diagram,
    written as a mermaid class diagram. The optimizer reads it to understand what can be changed and
    proposes improvements targeting specific degrees of freedom.

    ---

    ## Core concepts

    An agent is a graph of components. Regardless of framework, every agent has:

    | Concept | What it is |
    | ------- | ---------- |
    | **Prompt** | The string, template, or policy object that shapes what the LLM is asked |
    | **Method** | A callable the agent invokes to interact with the world (search, SQL, API, bash, files, etc.). A tool at agent disposal. |
    | **Stochastic Method** | A callable the agent invokes to interact with the world using a LLM (e.g. a tool that uses a LLM). A tool at agent disposal. |
    | **Subagent** | Another agent/executor instance the component delegates work to (e.g. a planner agent) |
    | **Model config** | Which LLM model and with what parameters (temperature, max_tokens, etc.) |
    | **Skills** | A collection of methods and knowledge that the agent can use to interact with the world |
    | **Association arrows** | The arrows that connect the components in the diagram |


    Based on the framework of choice, the components look different.
    Some popular frameworks example:

    **NeMo OO Agents**

    ```python
    from nooa import Agent, TextSkill
    from nooa.unifiedllm import get_llm_client

    llm = get_llm_client("aws/anthropic/claude-haiku-4-5-v1")  # model config

    class ResearchAgent(Agent, llm=llm):
        '''You are a research assistant.'''  # system prompt

        def __init__(self):
            super().__init__()
            self.rules = TextSkill(path=".claude/skills/domain-rules")  # skill

        async def search(self, query: str) -> list[str]:      # deterministic method
            return http_search(query)

        async def summarize(self, hits: list[str]) -> str:    # stochastic method (LLM)
            '''Summarize the search hits into a concise paragraph.'''  # method prompt
            ...

        async def run(self, question: str) -> str:            # stochastic method (LLM)
            '''Answer the question by searching and summarizing.'''  # method prompt
            ...
    ```

    **LangGraph / LangChain**

    ```python
    from langgraph.graph import StateGraph, START, END
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model="gpt-4o", temperature=0)    # model config

    @tool
    def search(query: str) -> list[str]:                  # deterministic method
        return http_search(query)

    def call_model(state):                                # stochastic method (LLM node)
        prompt = "You are a research assistant. Answer questions precisely."  # method prompt
        messages = [{"role": "system", "content": prompt}] + list(state["messages"])
        return {"messages": [model.bind_tools([search]).invoke(messages)]}

    graph = StateGraph(State)
    graph.add_node("agent", call_model)
    graph.add_edge(START, "agent")
    ```

    **Deep Agents (LangChain)**

    ```python
    from langchain_nvidia_ai_endpoints import create_deep_agent
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    summarize_llm = ChatOpenAI(model="gpt-4o-mini")       # model config (per-tool)

    @tool
    def search(query: str) -> list[str]:                  # deterministic method
        return http_search(query)

    @tool
    def summarize(hits: list[str]) -> str:                # stochastic method (LLM)
        prompt = "Summarize the search hits into a concise paragraph."  # method prompt
        return summarize_llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": str(hits)},
        ]).content

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-5",              # model config
        tools=[search, summarize],                        # methods
        system_prompt="Answer questions precisely.",      # system prompt
        skills=["domain-rules"],                          # skills
    )
    ```

    Understand how each framework defines the components and the relationships between them.

    ---

    ## Scope — what to include and what to exclude

    The diagram must contain ONLY components the optimizer can improve. Include and exclude strictly:

    **Include:**
    - All components listed above (agents, subagents, methods, tools, skills, prompts, etc.)
    - Default configuration (models being used, tool configurations, etc.)


    **Exclude — do not diagram these:**
    - Any infrastructure code (e.g. web framework, database, logging, monitoring, feature-flag plumbing, etc.) around the agent.
    - Non default configuration (e.g. tool configurations, model configurations, etc.)
    - Anything that's not strictly called by the agent (e.g. code that is not part of the agent's logic, such as setup code, testing code, etc.)
    - Legacy paths or non default paths that are not exercised in the current evaluation

    **Default path tracing:** When the entrypoint dispatches based on a feature flag or if/else,
    follow only the actively exercised branch based on the entrypoint. Check for flags, env variables and default runner.

    When in doubt: if the optimizer changing this component would not affect agent behavior on the task, exclude it.

    ---

    ## Format

    `architecture.md` has two sections: a **Mermaid flowchart** and a **prompts** appendix.

    ### Section 1 — Mermaid flowchart

    > **Mental model, not a code transcript.** The diagram is a friendly visualization of agent
    > topology — it is meant to be readable by humans and useful to the optimizer, not a one-to-one
    > copy of implementation details. For example, a dict of tool callables
    > (`self.TOOLS = {'search': search_fn}`) may be shown as individual method nodes because that
    > conveys the logical structure more clearly. What matters is capturing *what the agent can do
    > and how components relate*, not mirroring every syntactic choice in the source.
    > It is meant to be an execution overview of the agent, not a detailed code transcript.

    A mermaid `flowchart TD`. Every node is one of four types:

    **Node shapes:**

    Use `<br/>` for line breaks inside node labels — `\n` is not valid Mermaid syntax.

    | Node type | Mermaid syntax | When to use |
    | --------- | -------------- | ----------- |
    | Agent | `id{{"Name<br/>model: model-id<br/>system: <truncated><br/>skills: s1 · s2"}}` | The agent itself — hexagon. Omit `skills:` line if none. |
    | Method | `id["name<br/>in:  arg: Type<br/>out: ReturnType"]` | Deterministic callable with real logic (dispatch, transform, branch) — no LLM involved. Rectangle. Pure `await`-sequence methods are invisible — see rule below. |
    | Stochastic method | `id(["name*<br/>in:  arg: Type<br/>out: ReturnType<br/>prompt: <truncated>"])` | LLM-driven callable. Stadium shape. Append `<br/>model: override` only when the method overrides the agent's default model. |
    | Subagent | `subgraph AgentName["AgentName · model: model-id · system: <truncated>"]` ... `end` | Delegates work to another agent. Use a Mermaid subgraph containing the subagent's method nodes. |

    Truncate long strings to ~60 chars with `...` — the prompts appendix holds verbatim content.

    **Arrow types:**

    | Arrow | Meaning |
    | ----- | ------- |
    | `A --> B` | Forced path: A always invokes B — enforced by code (explicit `await self.B()`, `graph.add_edge`, etc.) |
    | `A -.-> B` | Optional path: A may invoke B — the LLM decides at runtime |

    **Critical rule:** prompt-instructed sequences are dashed, not solid. A stochastic method's
    prompt may say "always call X then Y" but the LLM may still deviate — only Python code or
    graph edges that unconditionally invoke a method justify a solid arrow.

    **Sequence vs fan-out:** When an orchestrator method calls A then B then C in code, draw a
    chain `run --> A --> B --> C`, not a fan-out `run --> A`, `run --> B`, `run --> C`. The arrow
    shows execution sequence, not just who owns whom.

    **Tool access:** If a method (class or other callable) is accessible from a stochastic method, draw a dashed arrow FROM
    that method TO the method. A method (class or other callable) is accessible if it is reachable from that method at runtime
    — e.g. `self.shell`, a module-level instance, or a closure variable. Accessibility, not call
    frequency, determines the arrow.

    **Invisible sequencers:** If a method's body consists entirely of `await` calls to other
    methods in sequence, omit it from the diagram. The diagram IS the graph that method builds.
    The agent hexagon connects directly to the first step in the chain.

    **Labeled solid arrows:** Label every solid arrow with the data flowing between steps:
    `A -->|param: Type| B`. If one step's output flows to multiple downstream steps, draw a
    labeled arrow for each downstream node. Dashed arrows are not labeled — LLM-decided data
    flow is not deterministic.

    **Mermaid edge label quoting:** If a label contains `[`, `]`, `<`, or `>`, wrap the entire
    label in double quotes: `A -->|"list[T]"| B`. Plain `|param: Type|` is fine when no special
    characters are present.

    Node labels use `<br/>` for line breaks. Each node type has a fixed field order:

    **Agent / Subagent fields** (include only fields that apply):
    ```
    AgentName
    model: <model-id>
    system: <first ~60 chars of system prompt>...
    skills: skill1 · skill2
    ```

    **Method fields** (deterministic — no LLM, rectangle `[...]`):
    ```
    method_name
    in:  param1: Type, param2: Type
    out: ReturnType
    ```

    **Stochastic method fields** (LLM-driven, stadium `([...])`, always ends with `*`):
    ```
    method_name*
    in:  param1: Type, param2: Type
    out: ReturnType
    prompt: <first ~60 chars>...
    model: <override-id>       ← only when different from agent default
    temperature: N             ← only when explicitly set non-default
    max_tokens: N              ← only when explicitly set non-default
    ```

    ---

    **Example 1 — NeMo OO: agent with skills, LLM tool dispatch**

    Demonstrates: agent hexagon with skills, stochastic methods, tool nodes,
    solid entrypoint from framework, all LLM-driven calls dashed.

    ```python
    class MyAgent(Agent, llm=get_default_model()):
        '''You are a research assistant.'''

        def __init__(self):
            self.shell = ShellTools()
            self.domain = TextSkill(path=".claude/skills/domain.md")

        async def solve_task(self, task_input: str) -> str:
            '''Solve the task using available tools.'''
            ...

        async def summarize(self, text: str) -> str:
            '''Summarize the text into a concise paragraph.'''
            ...
    ```

    ```mermaid
    flowchart TD
        MyAgent{{"MyAgent<br/>model: claude-opus-4-5<br/>system: You are a research assistant.<br/>skills: domain"}}

        shell["ShellTools<br/>in: command: str, path: str, content: str<br/>out: ShellResult | Match"]
        solve_task(["solve_task*<br/>in:  task_input: str<br/>out: str<br/>prompt: Solve the task using available tools."])
        summarize(["summarize*<br/>in:  text: str<br/>out: str<br/>prompt: Summarize the text into a concise paragraph."])

        MyAgent --> solve_task
        solve_task -.-> summarize
        solve_task -.-> shell
        summarize -.-> shell
    ```

    `MyAgent --> solve_task` solid: framework calls `solve_task` as the entrypoint in code.
    Everything else dashed: the LLM inside `solve_task` freely decides what to call.
    Both `solve_task` and `summarize` get dashed tool arrows — every stochastic method can use tools.

    ---

    **Example 2 — LangGraph: hard-wired graph edges, deterministic node, tool choice**

    Demonstrates: `add_edge` produces solid arrows, deterministic method (rectangle, no LLM),
    dashed tool choice when dispatch depends on runtime state.

    ```python
    SYSTEM = "You are a planner. Break the query into sub-steps."

    class PlannerAgent:
        def __init__(self):
            self.llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")
            self.TOOLS = {"search": search_tool, "sql": sql_tool}

        def create_plan(self, state: AgentState) -> AgentState:
            # LLM call
            ...

        def execute_step(self, state: AgentState) -> AgentState:
            # deterministic dispatch — reads state["plan"][0]["tool"], no LLM
            result = self.TOOLS[state["plan"][0]["tool"]](state["plan"][0]["input"])
            state["result"] = result
            return state

        def generate_final_answer(self, state: AgentState) -> AgentState:
            # LLM call
            ...

        def create_graph(self):
            graph = StateGraph(AgentState)
            graph.add_edge("plan", "execute")    # hard-wired
            graph.add_edge("execute", "answer")  # hard-wired
            return graph.compile()
    ```

    ```mermaid
    flowchart TD
        PlannerAgent{{"PlannerAgent<br/>model: meta/llama-3.1-70b-instruct<br/>system: You are a planner. Break the query into sub-steps."}}

        create_plan(["create_plan*<br/>in:  state: AgentState<br/>out: AgentState<br/>prompt: You are a planner. Break the query into sub-steps."])
        execute_step["execute_step<br/>in:  state: AgentState<br/>out: AgentState"]
        generate_final_answer(["generate_final_answer*<br/>in:  state: AgentState<br/>out: AgentState<br/>prompt: Summarize: {state[result]}"])
        search["search_tool<br/>in:  query: str<br/>out: list[str]"]
        sql["sql_tool<br/>in:  query: str<br/>out: str"]

        PlannerAgent --> create_plan
        create_plan --> execute_step
        execute_step -.-> search
        execute_step -.-> sql
        execute_step --> generate_final_answer
    ```

    Solid arrows: `add_edge` calls in Python code. `execute_step` is a rectangle (no LLM).
    Dashed to tools: `execute_step` reads `state["plan"][0]["tool"]` at runtime — either tool.

    ---

    **Example 3 — Code-enforced sequence with model override and generation params**

    Demonstrates: explicit Python `await` chain produces solid arrows; stochastic method
    with a different model and non-default temperature; deterministic orchestrator method.

    ```python
    class WriterAgent(Agent, llm=get_default_model()):
        '''You are a writing assistant. Draft and refine content.'''

        def __init__(self):
            self.style = TextSkill(path=".claude/skills/style.md")
            self.tone  = TextSkill(path=".claude/skills/tone.md")

        async def draft(self, brief: str) -> str:
            '''Write a first draft from the brief.'''
            ...

        @strategy(PredictStrategy(), llm=get_fast_model(), temperature=0.0)
        async def critique(self, draft: str) -> str:
            '''List exactly three weaknesses in the draft. Be terse.'''
            ...

        async def revise(self, draft: str, notes: str) -> str:
            '''Revise the draft to address every critique note.'''
            ...

        async def run(self, brief: str) -> str:
            # explicit await chain — deterministic Python, not LLM-driven
            draft  = await self.draft(brief)
            notes  = await self.critique(draft)
            return await self.revise(draft, notes)
    ```

    ```mermaid
    flowchart TD
        WriterAgent{{"WriterAgent<br/>model: claude-opus-4-5<br/>system: You are a writing assistant. Draft and refine content.<br/>skills: style · tone"}}

        draft(["draft*<br/>in:  brief: str<br/>out: str<br/>prompt: Write a first draft from the brief."])
        critique(["critique*<br/>in:  draft: str<br/>out: str<br/>prompt: List exactly three weaknesses in the draft. Be terse.<br/>model: claude-haiku-4-5<br/>temperature: 0.0"])
        revise(["revise*<br/>in:  draft: str, notes: str<br/>out: str<br/>prompt: Revise the draft to address every critique note."])

        WriterAgent -->|brief: str| draft
        draft -->|draft: str| critique
        draft -->|draft: str| revise
        critique -->|notes: str| revise
    ```

    `run` is invisible — it is a pure `await`-sequence method, so the diagram encodes what it builds.
    `WriterAgent` connects directly to `draft`, the first step, labeled with the input type.
    `draft` fans to both `critique` and `revise` because both consume its output.
    `critique` carries `model:` and `temperature:` because both differ from the agent default.

    ---

    **Example 4 — Multi-agent: subagent as nested subgraph, execution chain**

    Demonstrates: subagent as a Mermaid `subgraph` containing its method nodes, execution
    sequence as a chain (not fan-out), every stochastic method gets dashed arrows to all tools.

    ```python
    class Orchestrator(Agent, llm=get_default_model()):
        '''You are an optimization orchestrator.'''

        def __init__(self):
            self.shell = GuardedShellTools(...)
            self.analyzer = AnalyzerAgent(...)

        async def propose_improvements(self, analysis: str) -> list[Improvement]:
            '''Propose candidate improvements based on the analysis.'''
            ...

        async def implement_improvement(self, candidate: Candidate) -> None:
            '''Implement the proposed improvement in agent source.'''
            ...

        async def run(self, dataset: Dataset) -> None:
            analysis = await self.analyzer.analyze_trace(...)  # explicit subagent call
            proposal = await self.propose_improvements(analysis)
            await self.implement_improvement(proposal)
    ```

    ```mermaid
    flowchart TD
        Orchestrator{{"Orchestrator<br/>model: claude-opus-4-5<br/>system: You are an optimization orchestrator."}}

        shell["GuardedShellTools<br/>in: command: str, path: str, content: str<br/>out: ShellResult | Match"]
        propose(["propose_improvements*<br/>in:  analysis: str<br/>out: list[Improvement]<br/>prompt: Propose candidate improvements based on the analysis."])
        implement(["implement_improvement*<br/>in:  candidate: Candidate<br/>out: None<br/>prompt: Implement the proposed improvement in agent source."])

        subgraph AnalyzerAgent["AnalyzerAgent · model: claude-haiku-4-5 · system: You are a trace analyzer. Identify failure patterns..."]
            analyze_trace(["analyze_trace*<br/>in:  trace_path: str<br/>out: str<br/>prompt: Analyze the trace and identify failure patterns."])
        end

        Orchestrator -->|dataset: Dataset| analyze_trace
        analyze_trace -->|analysis: str| propose
        propose -->|candidate: Candidate| implement
        propose -.-> shell
        implement -.-> shell
    ```

    `run` is invisible — pure `await`-sequence method, the diagram encodes what it builds.
    `Orchestrator` connects directly to `analyze_trace` (first step) with the input data label.
    Solid labeled chain shows exact data contracts between steps — the optimizer knows what breaks if types change.
    Every stochastic method (`propose`, `implement`) gets dashed arrows to all tools Orchestrator owns.

    ---

    **Example 5 — Types and Prompts appendix** (for PlannerAgent from Example 2):

    `AgentState` appears in every arrow label and node signature — the optimizer needs its fields
    to propose schema changes. `str`, `list` etc. are primitives and need no entry.

    ```markdown
    ## Types

    ### AgentState
    - query: str
    - plan: list
    - result: str

    ## Prompts

    ### PlannerAgent.create_plan
    You are a planner. Break the query into sub-steps.

    {state[query]}

    ### PlannerAgent.generate_final_answer
    Summarize {state[result]}
    ```

    ### Section 2 — Types

    Full definitions of every non-primitive custom type that appears in the diagram — in `in:`,
    `out:`, or arrow labels. Primitive types (`str`, `int`, `bool`, `float`, `list[str]`, etc.)
    are self-explanatory and do not need an entry. Include every `TypedDict`, `dataclass`,
    `Pydantic model`, or named type that the optimizer would need to understand in order to
    propose a schema change.

    ```markdown
    ## Types

    ### TypeName
    - field1: Type
    - field2: Type
    - field3: Type | None
    ```

    ### Section 3 — Prompts

    Full verbatim content of every prompt found in the codebase — system prompts, user prompts,
    prompt templates, and prompt-returning methods. This is the primary optimization target.

    ```markdown
    ## Prompts

    ### ClassName.method_name  (or  module.CONSTANT_NAME)
    <full prompt string or method body, verbatim>

    ### ClassName.other_method
    <full prompt string or method body, verbatim>
    ```

    Include:
    - Every prompt that is passed to an LLM that can be acted upon or modified by someone coding the agent.

    ---

    ## ETHOS.md — read first if it exists

    An `ETHOS.md` in the workspace is a durable contract written by the
    agent's author. When present, read it before any source file. It is authoritative on the scope of the agent.

    It helps discern what is agent-pertinent and what is not.
    If ETHOS.md is absent, infer all of the above from source code.

    ---

    ## Writing architecture.md

    1. Check for documentation (specs, readmes) and read it first if present. Understand the scope of the agent.
    2. Read the agent's source `.py` files, starting from the entrypoint, following imports. Use the Ethos's framework and harness to anchor what you're looking for.
    3. Apply the Scope rules above: include only agent components, tools, skills, and prompts. Do not include anything that's infrastructure around it, evaluation code, storage code, etc.
       Only include default paths. If a user has to change configs to use this path, it should not be included.
    4. Collect key imports and write the diagram note.
    5. If the file already EXISTS (it is seeded from the ancestor for every non-baseline agent),
       edit it IN PLACE: apply only the minimal delta for the one change being documented and
       preserve everything else verbatim — same section order, Mermaid dialect, node-shape syntax,
       wording, and type-annotation style. Do **not** re-render or restyle unchanged content.
    6. Only when the file does NOT already exist, author it from scratch: write the diagram, then
       the prompts appendix (verbatim content of all prompt strings found).

    """


def _architecture_doc_codeact(max_iterations: int) -> CodeActConfig:
    """Bound one architecture-doc generation, which reads source through slow shell work."""
    return CodeActConfig(max_iterations=max_iterations, cell_timeout=3600.0)


# Keeps its own Evaluator rather than using ctx.evaluate: its smoke checks run against an
# artifact that is deliberately not yet a Candidate, and ctx.evaluate exists to associate a
# result with one. An internal check has nothing to associate and must not be recorded
# against the run.
class CodeEditBuilder(Agent, roles.Builder):
    """Create and modify agent source code as part of the optimization loop."""

    config_type = CodeEditBuilderConfig

    name = "code-edit"
    accepts = frozenset({CODE_CHANGE})

    def __init__(
        self,
        workspace: Path,
        config: CodeEditBuilderConfig | None = None,
        framework_skills_dirs: list[Path] | None = None,
        *,
        evaluator: Evaluator | None = None,
        dataset: Dataset | None = None,
        source_path: str | None = None,
        entrypoint: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the coder for the given workspace."""
        super().__init__(llm=kwargs.pop("llm", None) or get_default_model(), **kwargs)
        # Architecture extraction requires the same quality-oriented model as
        # the rest of the coding work.
        self._evaluator = evaluator
        self._dataset = dataset
        self._source_path = source_path
        self._entrypoint = entrypoint
        self._architecture_model = get_default_model()
        self._config = config or CodeEditBuilderConfig()
        self._workspace_path = workspace.resolve()
        self.shell = GuardedShellTools(cwd=self._workspace_path)
        self.todos = TodoManager()
        self.context["file_match"] = doc(Match)

        self.skills: SkillRegistry = SkillRegistry(self)
        spec(self, "skills", hidden=True)
        load_framework_skills(self.skills, framework_skills_dirs or [])
        self.skills.register("ext.architecture_skill", ArchitectureSkill())
        # Attach the optimization-card index so apply_change / wire_up_change
        # can consult the card matching change.optimization_type. Without
        # this, the CodeEditBuilder only sees the proposer's prose and the framework
        # skill -- the optimize cards never reach it (gitlab #148).
        self.optimize = Optimize(model_catalog_path=self._config.model_catalog_path)
        self.skills.register("ext.optimize", self.optimize)
        self.skills.activate(["cmd.*", "ext.*"])
        TokenBudgetSummarizer.install(
            self,
            llm=get_fast_model(),
            config=TokenBudgetConfig(max_tokens=self._config.max_summary_tokens),
        )

    async def describe(self, artifact: Path) -> None:
        """Document the artifact's architecture, for the next round's proposal to read."""
        await self.create_architecture_doc(artifact, source_path=self._source_path, entrypoint=self._entrypoint)

    async def list_available_models(self) -> list[str]:
        """Return the curated model IDs allowed for agent-under-test mutations.

        Returns:
            list[str]: Model IDs from the configured Experimentalist catalog.

        """
        catalog = self.optimize.optimize_model_capability.read_model_catalog()
        return [model.model_id for model in catalog.models]

    async def build(self, ctx: BuilderContext, proposal: Proposal, *, generation: int) -> Candidate:
        """Implement *proposal*, verify it integrates, and commit the Candidate for it.

        Args:
            ctx: Reserve a working copy, commit the result — the only two verbs a Builder gets.
            proposal: What to change, and what to change it from.
            generation: Strategy-supplied grouping index, stamped onto the Candidate.

        Returns:
            Candidate: committed, with its artifact validated. A failed build returns
            nothing at all, because it raises.

        Raises:
            RuntimeError: if the CodeEditBuilder was constructed without an evaluator or dataset,
                or if no tasks are available for the integration check.
            IntegrationCheckFailed: if the build cannot pass a smoke eval after the
                bounded number of fix attempts.

        """
        if self._evaluator is None or self._dataset is None:
            raise RuntimeError("CodeEditBuilder needs an evaluator and a dataset to verify a build; none were injected")
        change = CodeChange.model_validate(proposal.payload)
        fork = await ctx.fork(proposal)

        await self.apply_change(fork.workdir, proposal.description, change)
        await self.wire_up_change(fork.workdir, proposal.description, change)
        await self.run_pyright(fork.workdir)
        await self.optimize_subproblem(fork.workdir, proposal.description, change, self._dataset, self._evaluator)
        await self.run_pyright(fork.workdir)
        if not await self.integration_check(fork.workdir, self._dataset, self._evaluator, task_ids=change.task_ids):
            raise IntegrationCheckFailed(f"{fork.workdir.name} smoke eval failed after fix attempts")

        # Seeded only now, after the change is final: during editing the ancestor's
        # architecture doc would describe an agent that no longer exists, and the fork
        # deliberately leaves it out for that reason. Here it becomes the base to edit.
        if fork.upstream is not None:
            upstream_doc = fork.upstream / "architecture.md"
            own_doc = fork.workdir / "architecture.md"
            if upstream_doc.exists() and not own_doc.exists():
                shutil.copy2(upstream_doc, own_doc)
        await self.describe(fork.workdir)

        return await ctx.commit_candidate(proposal=proposal, artifact=fork.workdir, generation=generation)

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=50, cell_timeout=3600.0)))
    async def apply_change(self, workdir: Path, optimization: str, change: CodeChange) -> None:
        """Apply the ONE change described by `optimization`.

        Args:
        - workdir (Path): the agent directory to modify. Every file you touch is under here.
        - optimization (str): graph-level description of the change to make.
        - change (CodeChange): why this change was proposed — `change.root_cause` is the
          diagnosed reason the agent underperforms, `change.optimization_type` says which
          optimization card covers it, and `change.task_ids` are the tasks that exercise it.

        ## Pre-staged destination — DO NOT re-copy from the ancestor
        {workdir} has ALREADY been populated with a copy of the ancestor's source files
        (everything except ``architecture.md``, which describes the ancestor and is
        re-seeded and rewritten after you are done). Runtime harness files are inherited
        when present but are not part of the agent behavior being optimized.
        Open and modify files in place. You do NOT need to copy anything in.

        ## Required reading before editing
        - The framework skill — how to write agents in the target framework (NeMo OO, LangChain, etc.).
        - The optimization card for this change. First run `print(doc(self.optimize))`
          to see the decision tree mapping optimization types to cards. Find which
          card covers `change.optimization_type` (e.g. `add_concrete_method` is
          covered by `optimize_execution`), then load that card's details with
          `print(doc(self.optimize.optimize_execution))`. The card lists approved approaches,
          code examples, and success criteria.
        - If the change involves methods that handle large inputs (scraped pages,
          documents, search results), read `print(doc(self.processing_large_data))`
          for chunking patterns that avoid `max_param_chars` truncation errors.

        ## Constraints
        - Make one change matching `optimization`, addressing `change.root_cause`.
        - No hardcoded knowledge (keyword lists, lookup tables, if/elif on task-specific strings). This applies to BOTH code AND skill edits: skill additions must improve generalizable methodology (process steps, classification rules, evidence requirements), never add dataset-specific facts about specific libraries, frameworks, CVEs, or package internals — that is memorization of training data, not domain knowledge.
        - NEVER read from, copy from, or overwrite files in any OTHER agent directory.
          {workdir} is yours; its sibling directories belong to other candidates and the
          pre-staged files inside it are the only source you should edit.
        - VERIFY INTEGRATION: In the integration test, check traces to confirm new code is actually
          called by the agent. Code that exists but isn't invoked is useless.
        - If the change involves editing an LLM model id, call
          `models = await self.list_available_models()` first and pick one of the
          returned ids. Do NOT invent or abbreviate model ids.
        """
        ...

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=50, cell_timeout=3600.0)))
    async def wire_up_change(self, workdir: Path, optimization: str, change: CodeChange) -> None:
        """Wire up the change to the agent such that it is called by the agent. Load the framework skill and use it to understand how to wire up the change.

        Args:
        - workdir (Path): the agent directory to modify.
        - optimization (str): the change that was just applied.
        - change (CodeChange): why it was proposed, and which optimization card covers it.

        Consult `doc(self.optimize)` to find which card covers
        `change.optimization_type`, then read that card via
        `doc(self.optimize.optimize_<type>)` for what "wired" means in the target framework.
        (e.g. for `add_skill` covered by `optimize_domain_knowledge`, the new skill
        must be reachable by the agent at runtime — a file that is never imported or
        registered is not wired, regardless of framework).
        """
        ...

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=200, cell_timeout=3600.0)))
    async def optimize_subproblem(
        self,
        workdir: Path,
        optimization: str,
        change: CodeChange,
        dataset: Dataset,
        evaluator: Evaluator,
    ) -> None:
        """Validate and iteratively refine the subproblem fix until the agent's behavior improves.

        Args:
        - workdir (Path): the agent directory being refined.
        - optimization (str): graph-level description of the change that was applied.
        - change (CodeChange): `change.root_cause` is the diagnosed reason the agent
          underperforms, and `change.task_ids` are the tasks that exercise it.
        - dataset (Dataset): The dataset to use for the evaluation.
        - evaluator (Evaluator): The evaluator to use for the evaluation.

        ## Workflow

        1. **Pick test tasks**:
           - **Target tasks**: use `change.task_ids` — the tasks the Proposer flagged as
             most directly exercising `change.root_cause`. The fix must improve these.
             Only if `change.task_ids` is empty, fall back to picking 2-3 representative
             tasks that exercise the subproblem yourself.
           - **Regression tasks**: also pick 2-3 tasks the ancestor/baseline already handled
             correctly and that are unrelated to this subproblem. The fix must NOT regress
             these — they guard against the change breaking previously-working behavior.
           - Consider `optimization` to understand what was changed

        2. **Isolate the subproblem**:
           - Read the agent's architecture.md to understand the agent's behavior
           - Read the optimization card for `change.optimization_type` to understand what was changed
           - Read `optimization` and `change.root_cause` to understand the specific subproblem being fixed
           - Run the subproblem fix on the tasks and inspect the results.
             E.g. if the subproblem was the behavior of a single method, call that method on the tasks
             and inspect the results. If it was a single tool, run that tool directly.
             Always prefer isolated calls for speed; only run a full smoke eval if needed.

           Example: Adding a new method to the agent to create unit tests.

            ```python
            class SWEAgent(Agent):
                async def create_unit_tests(self, issue: str, repo: str) -> str:
                    '''Create unit tests for the issue.'''
                    return "Unit tests created."

            ### Test
            task_description = ...
            repo = ...
            print(SweAgent().create_unit_tests(task_description, repo))
            ```

        3. **Reason about whether the behavior improved**:
           - Compare what you observe to what the fix was supposed to achieve
           - Look for concrete evidence in traces and outputs

           **Example**: Subproblem = "agent prematurely concludes without thorough analysis"
           - Fix applied: increased minimum reasoning steps before concluding
           - What to check: Does the trace show multiple reasoning/analysis tool calls?
           - Verdict: subproblem is fixed only if BOTH hold — the target tasks now show the
             improved behavior AND the regression tasks still behave as they did on the baseline
             (no regression). If both → you're done.
           - If no: identify specifically what's still wrong — either the target isn't fixed
             ("train-005 still has only 1 reasoning call") or a regression task broke
             ("train-012 was passing but now fails").

        4. **If not fixed** (max 3 iterations):
           - Make targeted changes to the agent source based on what you observed
           - Repeat from step 2 with the same tasks (for consistency)

        5. **Done** - no cleanup needed

        ## Key principle

        You're validating **behavior change**, not just correctness. The subproblem is about
        how the agent acts (thoroughness, reasoning depth, tool usage patterns, report quality).
        Use the traces as evidence of whether the agent's behavior actually improved.

        """
        ...

    async def integration_check(
        self,
        workdir: Path,
        dataset: Dataset,
        evaluator: Evaluator,
        max_fix_attempts: int | None = None,
        task_ids: Sequence[str] | None = None,
    ) -> bool:
        """Smoke-test the build and attempt bounded LLM repairs if it fails.

        Args:
            workdir: The agent directory to smoke-test.
            dataset: The dataset to use for the evaluation.
            evaluator: The evaluator to use for the evaluation.
            max_fix_attempts: Maximum repair iterations; defaults to
                ``self._config.max_fix_attempts``.
            task_ids: The tasks the change claims to fix. Checking these is what makes a
                failure informative — a random task the change does not touch passes on
                the first attempt, and the repair loop below never runs.

        Returns:
            bool: True if the build passes the smoke eval, False otherwise.

        Raises:
            RuntimeError: if no training tasks are available.

        """
        _max_fix_attempts = max_fix_attempts if max_fix_attempts is not None else self._config.max_fix_attempts

        all_tasks = list(dataset.list_tasks())
        if not all_tasks:
            raise RuntimeError("No tasks available for integration_check")
        # The tasks the Proposal named as evidence, so a build that does not repair what
        # it claimed fails here and gets the repair attempts below. Falling back to one
        # random task — seeded by the directory name so retries hit the same one — keeps
        # the old "does it run at all" check for a Proposal that names none.
        targeted = [task for task in all_tasks if task.id in set(task_ids or ())]
        tasks = targeted or [random.Random(workdir.name).choice(all_tasks)]
        for attempt in range(_max_fix_attempts + 1):
            evaluation = await self.run_smoke_eval(workdir, tasks, dataset, evaluator)
            if self._is_smoke_results_healthy(evaluation, tasks):
                return True
            if attempt < _max_fix_attempts:
                await self._fix_runtime_issues(workdir, evaluation)
        return False

    def _is_smoke_results_healthy(self, evaluation: EvaluationResult, tasks: Sequence[Task]) -> bool:
        """Return True if each selected task has a completed trial without evaluator error.

        Args:
            evaluation: Smoke evaluation result.
            tasks: Tasks included in the smoke evaluation.

        Returns:
            bool: True if all tasks produced at least one clean trial.

        """
        trials_by_task: dict[str, list[TrialResult]] = defaultdict(list)
        for trial in evaluation.trials:
            trials_by_task[trial.task_id].append(trial)

        for task in tasks:
            trials = trials_by_task.get(task.id, [])
            if not trials:
                return False
            if not any(self._trial_runtime_healthy(trial) for trial in trials):
                return False
        return True

    def _trial_runtime_healthy(self, trial: TrialResult) -> bool:
        """Return True if evaluator status/error indicates runtime success."""
        if trial.error is not None:
            return False
        return trial.status == "completed"

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=50, cell_timeout=3600.0)))
    async def _fix_runtime_issues(self, workdir: Path, evaluation: EvaluationResult) -> None:
        """Diagnose and repair runtime failures surfaced by the most recent smoke eval.

        Args:
        - workdir (Path): the agent directory being repaired. Every file you touch is under here.
        - evaluation (EvaluationResult): the smoke evaluation result. Use its
          trials, metrics, outputs, resources, errors, and metadata as failure evidence.

        The orchestrator (`integration_check`) just ran `run_smoke_eval` and the
        deterministic health check returned False. Your job is to read the failure
        evidence and edit the agent source so the next smoke eval will pass. Do NOT
        re-run smoke eval yourself — the orchestrator does it.

        ## Failure context (read in this order)

        For each trial in `evaluation.trials`:
        1. `trial.status` and `trial.error` — crash, timeout, or evaluator-owned
           failure details.
        2. `trial.outputs` — agent-visible output values or referenced output files.
        3. `trial.resources` — logs, traces, manifests, or artifact references made
           visible by the evaluator.
        4. `trial.metrics` — numeric evaluator results. Names are evaluator-defined.
        5. `trial.metadata` — adapter facts that help locate or interpret evidence.

        Read ResourceRef descriptions and metadata before opening referenced files.
        Do not assume any particular directory layout or artifact filename.

        ## What to fix

        Edit files in {workdir} — the same files `apply_change` and `wire_up_change`
        produced. Common runtime failure modes and their fixes:

        - **TypeError / wrong signature** — tool/function called with bad kwargs.
          Match the call to the definition.
        - **NameError / ImportError** — the change references a symbol that doesn't
          exist or removed an import. Add the import or revert the reference.
        - **KeyError on env var / config** — the change added a config read without
          a default or `.env` entry. Use `os.environ.get(..., default)` or revert.
        - **Evaluator failure, no exception** — the agent ran but failed task checks.
          Read visible outputs/resources to identify the missing file, wrong schema,
          bad side effect, service failure, or incorrect final value. Required
          artifact paths are task-defined; do not assume a generic `output.*`.
        - **Tool that no longer exists is being called** — a tool was removed but
          a call site wasn't updated. Update the call site.

        ## Constraints

        - Do NOT touch optimizer infrastructure: `init_structure.py`,
          `optimize_agent.py`, anything under `.claude/skills/`.
        - Do NOT call `run_smoke_eval` — the orchestrator re-runs it after you finish.
        - Make the minimum edit that fixes the failure. Don't introduce new behavior.
        """
        ...

    async def run_pyright(self, workdir: Path) -> str:
        """Run pyright on the agent's directory and return diagnostic lines.

        Args:
            workdir: The agent directory to check.

        Returns:
            str: newline-separated error and warning lines, or an empty string
                if pyright reports no issues.

        """
        r = await self.shell.run(f"pyright --outputjson {shlex.quote(str(workdir))} 2>/dev/null || true")
        output = (r.stdout or "").strip()
        try:
            data = json.loads(output)
            diags = data.get("generalDiagnostics", [])
            if not diags:
                return ""
            lines = []
            for d in diags:
                if d.get("severity") in ("error", "warning"):
                    file_ = d.get("file", "")
                    rule = d.get("rule", "")
                    msg = d.get("message", "")
                    rng = d.get("range", {}).get("start", {})
                    line_no = rng.get("line", "?")
                    lines.append(f"{file_}:{line_no}: {d['severity']}: {msg} ({rule})")
            return "\n".join(lines)
        except Exception:
            return output

    async def run_smoke_eval(
        self,
        workdir: Path,
        tasks: Sequence[Task],
        dataset: Dataset,
        evaluator: Evaluator,
    ) -> EvaluationResult:
        """Run smoke evaluation against a specific subset of tasks.

        Unrecorded by design: this measures work in progress, which is not yet a
        Candidate, so there is nothing for the run to associate a result with.

        Args:
            workdir: The agent directory to evaluate.
            tasks: Non-empty task objects to evaluate.
            dataset: The dataset to use for the evaluation.
            evaluator: The evaluator to use for the evaluation.

        Returns:
            EvaluationResult: smoke evaluation result.

        Raises:
            ValueError: if ``tasks`` is empty.

        """
        if not tasks:
            raise ValueError("run_smoke_eval requires at least one task")

        smoke_dataset = dataset.subset([task.id for task in tasks])
        base_options = evaluator.options
        # A configured job_name would otherwise be shared: builds run concurrently, and
        # `force_rerun` deletes the results directory that name points at, so one builder's
        # smoke check would clear another's mid-read. The workdir and the task subset are
        # what make this check distinct, so they name it.
        smoke_options = base_options.model_copy(
            update={"force_rerun": True, "job_name": f"smoke-{workdir.name}-{smoke_dataset.id}"}
        )
        return await evaluator.run(
            agent=workdir,
            dataset=smoke_dataset,
            options=smoke_options,
        )

    async def create_architecture_doc(
        self, workdir: Path, source_path: str | None = None, entrypoint: str | None = None
    ) -> None:
        """Update (or, only if absent, create) architecture.md for the given agent.

        Args:
            workdir: The agent directory to document.
            source_path: Directory holding the agent source, relative to the agent directory.
            entrypoint: File the evaluation harness invokes, relative to the agent directory.

        """
        # A @strategy config is fixed when the class is defined, but how many iterations
        # documenting an agent takes scales with the source this builder was handed.
        codeact = _architecture_doc_codeact(self._config.max_architecture_doc_iterations)
        await self._create_architecture_doc(
            workdir,
            source_path=source_path,
            entrypoint=entrypoint,
            _strategy=CodeActStrategy(config=codeact),  # ty: ignore[unknown-argument]
        )

    @strategy(
        CodeActStrategy(config=_architecture_doc_codeact(CodeEditBuilderConfig().max_architecture_doc_iterations)),
        llm=lambda self: self._architecture_model,
    )
    async def _create_architecture_doc(
        self, workdir: Path, source_path: str | None = None, entrypoint: str | None = None
    ) -> None:
        """Update (or, only if absent, create) architecture.md for the given agent.

        Args:
        - workdir (Path): the agent directory to document. Its architecture.md is at {workdir}/architecture.md
        - source_path (str | None): directory containing the agent source code, relative to the
          agent directory (e.g. "app/"). Read .py files from here when following imports.
          When absent, infer the source root from ETHOS.md or the directory structure.
        - entrypoint (str | None): the file that the evaluation harness invokes to run the agent,
          relative to the agent directory (e.g. "harbor/runner.py"). Start reading here to
          determine the call chain and identify the default execution path before reading
          source_path. When absent, infer from ETHOS.md or the directory structure.

        Load the agent-architecture skill and use it to document the agent.

        ## Edit in place — do NOT regenerate from scratch
        For any candidate with an ancestor, the framework has ALREADY seeded
        {workdir}/architecture.md with a
        byte-for-byte copy of the ancestor's architecture.md. Open the existing file and edit it
        in place to reflect the FINAL state of the candidate source (after all automated repairs),
        not just its originally declared change. Compare the final source against the ancestor and
        capture every resulting difference — nodes, edges, prompts, and type definitions — as a
        MINIMAL, targeted delta:

        - Preserve the existing document verbatim wherever the code did not change — same section
          order and headings, same Mermaid dialect and node-shape syntax, same wording, spacing,
          and type-annotation style. Do NOT restyle, reflow, re-title, re-render, or "improve" any
          part the change did not touch.
        - Edit ONLY the lines the final source differs from the ancestor: a node added/removed/
          modified, an edge rerouted, an affected custom type definition, or a prompt string that
          changed. If a node or type was already drawn a certain way in the ancestor's file, keep
          drawing it that way.
        - A reviewer diffing this file against the ancestor's must see only the real architectural
          delta. Spurious formatting churn (different Mermaid dialect, added prose, re-cased type
          names, reordered sections) hides the actual change and is a defect.

        ## Steps

        1. Read the entrypoint {entrypoint} to identify the default execution path — which feature
           flag value is default, which if/else branch runs in production. The entrypoint itself
           is infrastructure and must NOT appear in the diagram. Use it only to determine scope.
           Then read source files under {source_path}, following only imports reachable from the
           default path. If entrypoint is None, infer it from ETHOS.md or the directory.
           Apply the scope rules from the architecture skill:
           include only agent components (agents, stochastic methods, deterministic methods, tools),
           exclude the entrypoint/harness itself, evaluation code, test data, and infrastructure.

        2. Diff your understanding of the current source against the EXISTING architecture.md and
           apply the minimal edit that brings it up to date. Pay special attention to prompt strings
           (Section 3), which change frequently: copy them verbatim from source definitions, not
           call sites. A reader must understand what the LLM is asked to do without opening any
           source file.

        3. ONLY if architecture.md does not already exist (e.g. a baseline agent with no ancestor),
           author it from scratch using the node shapes and arrow conventions from the architecture
           skill exactly:
           - Agent hexagon {{}} for each agent
           - Stochastic method stadium ([]) with * suffix for every method that calls an LLM
           - Deterministic method rectangle [] for methods with real logic but no LLM
           - Dashed arrows -.-> for LLM-decided calls; solid arrows --> for code-enforced calls
           - Labeled solid arrows -->|param: Type| next showing data contracts
           - Section 2: Types — definitions of all non-primitive custom types used in the diagram
           - Section 3: Prompts — verbatim prompt strings the LLM will read (instruction sentences,
             role descriptions, template content). Copy from source definitions, not call sites.
        """
        ...
