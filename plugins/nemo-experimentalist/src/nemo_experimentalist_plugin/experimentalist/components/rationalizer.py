# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast  # noqa: F401
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
from collections import Counter, defaultdict  # noqa: F401
from pathlib import Path
from typing import Any

from nemo_experimentalist_plugin.entities import DependencyRuntime, Task
from nemo_helix_plugin.nooa_model_client import get_default_model, get_fast_model
from nooa import Agent, CodeActStrategy, strategy
from nooa.agentdoc import doc, spec
from nooa.agents import TokenBudgetSummarizer
from nooa.config import CodeActConfig
from nooa.config.summarizer_config import TokenBudgetConfig
from nooa.skill_registry import SkillRegistry
from nooa.tools import Match, TodoManager
from pydantic import BaseModel, Field

from . import cache
from .rationale import Rationale, RationaleStep
from .tools import GuardedShellTools
from .util import load_framework_skills

__all__ = ["Rationale", "RationaleStep", "Rationalizer", "RationalizerConfig"]


class RationalizerConfig(BaseModel):
    """Configuration for Rationalizer tuning parameters."""

    max_summary_tokens: int = Field(
        default=80_000,
        description="Max tokens the token-budget summarizer may use.",
    )


class Rationalizer(Agent):
    """Your role is to bootstrap the reference reasoning trace.

    You are the bootstrapping reasoner for trace analysis. Your output is
    the "golden path" that lets trace analyzer compare a failed agent run
    against what a competent task agent should have discovered, in order,
    from visible evidence.

    You are not the evaluator and you are not the benchmark agent. You may
    use evaluator-private material to uncover the correct causal path, but
    your returned rationale must teach trace analyzer the public reasoning
    trajectory a task agent could have followed.

    Your job has two distinct phases:

    1. Privileged debugging: use private references and the live runtime to
        understand what a passing run requires.
    2. Public trace reconstruction: return the shortest faithful sequence
        of actions that a normal task agent could have taken from the task
        instruction and visible runtime alone.

    The returned ``Rationale.steps`` are phase 2 only. They are not a log of
    every privileged thing you did as the Rationalizer.
    """

    def __init__(
        self,
        workspace: Path,
        config: RationalizerConfig | None = None,
        framework_skills_dirs: list[Path] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Rationalizer with workspace path and optional config.

        Args:
            workspace: Absolute path to the eval-and-optimize workspace root.
            config: Tuning parameters; defaults to ``RationalizerConfig()`` if ``None``.
            framework_skills_dirs: Optional list of directories containing framework skills to load.
            **kwargs: Forwarded to ``Agent.__init__``.

        """
        super().__init__(llm=kwargs.pop("llm", None) or get_default_model(), **kwargs)
        self._config = config or RationalizerConfig()
        self._workspace_path = workspace
        self.shell = GuardedShellTools(cwd=workspace)
        self.todos = TodoManager()
        self.context["file_match"] = doc(Match)

        self.skills: SkillRegistry = SkillRegistry(self)
        spec(self, "skills", hidden=True)
        load_framework_skills(self.skills, framework_skills_dirs or [])
        TokenBudgetSummarizer.install(
            self,
            llm=get_fast_model(),
            config=TokenBudgetConfig(max_tokens=self._config.max_summary_tokens),
        )

    async def run(self, task: Task, ethos: Path | None = None) -> Rationale:
        """Generate a correct reasoning trace for the task.

        Args:
            task: The evaluator task to rationalize.
            ethos: optional path to a materialized ETHOS.md file.

        Returns:
            Rationale: minimal chain-of-thought steps a correct agent would follow.

        """
        ethos_digest = ""
        if ethos is not None and ethos.exists():
            import hashlib

            ethos_digest = hashlib.sha256(ethos.read_bytes()).hexdigest()[:16]
        key = cache.task_hash(f"{task.id}:{ethos_digest}")
        cached = cache.load(self._workspace_path, key, Rationale)
        if cached is not None:
            return cached
        async with task.start_deps() as runtime:
            rationale = await self.solve(task, runtime, ethos=ethos)
            rationale = await self.verify(task, runtime, rationale, ethos=ethos)
            cache.store(self._workspace_path, key, rationale)
        return rationale

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=30, cell_timeout=120.0)))
    async def verify(
        self, task: Task, runtime: DependencyRuntime | None, rationale: Rationale, ethos: Path | None = None
    ) -> Rationale:  # pyright: ignore[reportReturnType]
        """Execute the rationale against the live task runtime and augment until it scores 100%.

        Args:
            task: Evaluator task being rationalized.
            runtime: Dependency runtime to use for re-running actions.
            rationale: Draft rationale to verify.
            ethos: Optional path to a materialized ETHOS.md file; read
                as private context before executing (same as in ``solve()``).

        Returns:
            Rationale that, when followed exactly, produces a fully passing task run.

        ## Goal

        Execute the draft rationale's steps in the live task runtime. If the
        task does not pass after following those steps, identify what is missing
        and add the missing steps to the rationale. Repeat until the task passes.

        The returned rationale must be self-sufficient: a task agent following
        it step-by-step, with no other context, must achieve a full score.

        ## Runtime — already live

        The ``runtime`` parameter is the ALREADY-STARTED task dependency
        context. All task services are live. **NEVER call ``task.start_deps()``
        inside this method** — it creates a new isolated context that resets
        all conversation and service state to zero.

        ## Step 0 — Build the scoring checklist

        Before executing, discover every criterion the task scores against.
        Use private references, task tests, rubrics, and runtime introspection
        to enumerate all requirements: required outputs, required behaviors,
        required values, thresholds, schemas, and any judge criteria.

        This checklist drives Steps 1 and 2. Use private material here for
        discovery, but do not include private inspection as a task-agent step
        in the final returned rationale.

        ## Step 1 — Execute the rationale and measure the score

        Run each action in the draft rationale against the live runtime in order.
        Replace each stored observation with the actual raw output from the real
        execution (truncated to 600 chars). Do NOT summarize or paraphrase.

        After executing all steps, run the task's scoring mechanism (test suite,
        evaluator check, output validation, or judge) to measure the current score.

        For each step also check:

        **Observation accuracy** — does the real execution output match what was
        stored? An empty or near-empty observation (fewer than 20 characters) is
        always wrong — the action was never re-executed after the private phase.
        Re-run it now and store the verbatim output. If the stored observation is
        a clean structured summary with no noise, it was fabricated. Real tool
        output contains irrelevant lines, raw blobs, and truncated content.
        Replace it with the verbatim real output.

        **Traceability** — every specific value used in a step (named entity,
        URL, identifier, statistic, date, or domain-specific term) must appear
        verbatim in the task instruction OR in a prior step's real observation.
        If a value appears with no prior source, insert a step before it that
        fetches that value from the live runtime.

        **Next-step support** — does the real observation contain the values
        that the following step's thought claims to derive from it? If not,
        correct the thought or insert an intermediate step.

        ## Step 2 — Augment until the task passes

        If the score after Step 1 is not maxed out, identify every failing check
        from the scoring checklist and add the steps needed to satisfy them.

        For each failing check:
        - Add a new step (or steps) that produces the required output, state
          change, or value via a real tool call — not from LLM memory.
        - The observation must contain the actual result from the live runtime.
        - The step must be expressed as a task-agent-visible action: something
          the AUT could do from its instruction and live environment alone.
          If the failing check came from a private test or reference solution,
          translate it into an equivalent public action (run the visible program,
          inspect the produced artifact, probe the exposed service).

        After augmenting, re-run the scoring mechanism. Repeat until all checks
        pass. A rationale that does not produce a passing run is incomplete.

        ## Return

        ```python
        from nemo_experimentalist_plugin.experimentalist.components.rationalizer import (
            Rationale,
            RationaleStep,
        )
        Rationale(task_name=task.id, steps=revised_steps)
        ```

        """
        ...

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=50, cell_timeout=3600.0)))
    async def solve(self, task: Task, runtime: DependencyRuntime | None, ethos: Path | None = None) -> Rationale:  # pyright: ignore[reportReturnType]
        """Solve the task and record each action as a RationaleStep as you go.

        Args:
            task: Evaluator task to solve.
            runtime: Dependency runtime to use for solving the task.

        Returns:
            Rationale whose steps are the actual execution trace.


        ## Ethos — private context, never a rationale step

        If ``ethos`` is not None, read it as private setup before writing
        any steps. This is Rationalizer-only context — the AUT never reads its
        own Ethos at runtime, so do NOT include this as a ``RationaleStep``:

        ```python
        if ethos is not None and ethos.exists():
            ethos_text = ethos.read_text()
            print(ethos_text[:3000])
        ```

        Use it to understand the agent's scope and goal: what problem it
        solves, what a correct outcome looks like, what success criteria it is
        held to, and what tools it has available. Ground your rationale in
        that understanding, but every step in ``Rationale.steps`` must be an
        action the AUT itself takes from its task instruction and live
        environment — not rationalizer setup work.

        ## Ethos-given facts are pre-known — do not rediscover or justify them

        Everything the Ethos states is a GIVEN the AUT already possesses:
        the tools it has and their calling conventions, the schema, the I/O
        format and output path, how to reach the runtime (container, endpoint,
        credentials), and any fixed domain terminology. The Ethos IS the agent's
        equipment — the AUT does not discover it at runtime, so the rationale
        must NOT contain steps that rediscover or re-derive it.

        Concretely, do NOT emit steps that:
        - locate or identify the runtime container / endpoint (the access
          method is Ethos-given),
        - grep the app source for helper/tool function names or signatures,
        - inspect files to learn the schema, table/column names, or the
          expected response category/format.

        These are not AUT actions — they are the rationalizer re-proving what
        the Ethos already told the agent, and they add pure noise. Start the
        trace from the first step that consumes genuinely unknown, task-specific
        information (the live query result, the actual data, the count). Only
        facts NOT in the Ethos — the values the AUT must obtain from the live
        environment for this specific task — require a grounding observation.

        ## Output contract — write from the AUT's perspective

        ``Rationale.steps`` is a first-person trace narrated by the Agent Under
        Test (AUT), not by you. Every ``thought``, ``action``, and
        ``observation`` is written as if the AUT is the subject:

        - ``thought``: what the AUT reasons at this point, given only what it
          has seen so far (task instruction + prior observations).
        - ``action``: a concrete step the AUT takes — reading a file, running
          a command, calling a tool, writing an output artifact.
        - ``observation``: the raw output the AUT receives from that action.

        The AUT's first step is always reading or interpreting the task
        instruction. It has no access to private references, hidden tests,
        evaluator metadata, or the Ethos you read above. Do not include
        any step the AUT could not have taken from its task instruction and
        live environment alone.

        Your job is to privately solve the task (using all your privileged
        access), then reconstruct the steps as the AUT would have taken them.
        The rationalizer's own setup work — reading the Ethos, inspecting
        private references, running hidden checks — never appears in the steps.

        ## Why reconstruct, not replay

        Do NOT return a log of what you personally did as the Rationalizer.
        Privately solve the task first to understand what steps are necessary,
        then write the steps from the AUT's point of view. This reconstruction
        must be faithful: every value in an ``action`` must be derivable from
        the task instruction or a prior ``observation`` — no fabricated leaps.

        ## Data structure

        Initialize once at the start:

        ```python
        from nemo_experimentalist_plugin.experimentalist.components.rationalizer import (
            Rationale,
            RationaleStep,
        )
        steps = []
        ```

        Before every tool call, write the reasoning into ``thought`` and the
        concrete action into ``action``. After the call returns, append the step
        immediately:

        ```python
        thought = (
            "Task instructions say the article title is 'A Review of AR Applications "
            "for History Education' — this is a known sub-field. I need the paper's "
            "reference list to find the primary studies."
        )
        action = "Search Semantic Scholar for 'Challenor Ma 2019 AR history education review'."
        result = await self.shell.run('python3 -c "..."')   # actual tool call
        steps.append(RationaleStep(
            thought=thought,
            action=action,
            observation=str(result)[:600],
        ))
        ```

        At the very end return:

        ```python
        Rationale(task_name=task.id, steps=steps)
        ```

        ## Thought rule — AUT voice, quote the trigger

        Write each ``thought`` as the AUT reasoning from what it has seen.
        It must name the exact value or signal from the PRIOR step's
        ``observation`` (or from the task instruction for step 1) that caused
        this action. Quote it verbatim.

        The ``action`` field states the concrete step the AUT takes next.

        Example: "The instruction observation says output_path='/results/answer.json'.
        I need to create '/results/' before writing."; action: "Create '/results/' and write the output."

        Never write generic motivation like "need more data." Never write from
        the rationalizer's perspective ("I read the Ethos and learned...").

        ## Traceability rule — values must be earned

        Any specific value in a `thought` or `action` — a named entity, URL,
        identifier, query term, or domain-specific fact — must either appear
        verbatim in the task instructions OR have been produced by a prior step's
        `observation`. If a value appears in an action without being returned by
        an earlier observation, that is a bug.

        **Active gate — check before writing each thought:** Scan the thought for
        every specific value. For each one, locate where it appears in the task
        instruction text or in a prior step's observation text. If you cannot
        point to the exact source, remove it or add a prior step that fetches it.

        ## Observation verbatim rule — raw output only

        Every observation must be the actual, unmodified output of the tool call,
        truncated to fit. Never paraphrase, interpret, or summarize into the
        observation field.

        **If the observation looks like any of the following, it is fabricated —
        re-run the action and store the real output instead:**
        - A clean numbered list with tidy category labels and no noise
        - A structured set of key-value pairs with no irrelevant lines
        - Output that conveniently matches the expected answer with no artifacts
        - A tidy summary paragraph with no raw tool output visible

        Real tool output is noisy: it contains extra whitespace, irrelevant lines,
        raw JSON blobs, HTTP headers, truncated sentences, and values that don't
        yet tell you anything. That noise is the signal that the observation is real.

        ## Depth mandate — exhaust every dimension *required by the task goal*

        When the task goal requires knowing ALL instances of something — all
        vulnerabilities, all affected packages, all search results, all
        matching records — finding one is a starting point, not a conclusion.
        Exhaust the required space before concluding.

        **This mandate applies to information the task goal depends on.** It
        does NOT mean enumerating every piece of data encountered along the
        way. If the task is to complete an action (book a reservation, fix a
        bug, write a file), only gather the data that action requires. Do not
        inspect unrelated environment state just because it appears in a
        response.

        **Patterns that signal shallow analysis — each is a bug:**
        - The task requires finding all instances of X → stopped after finding one.
        - Ran one search query for a required fact → did not verify at the primary source.
        - Drew a conclusion from absence without a direct negative check.
        - Observation is a clean structured summary → fabricated, must re-run.

        ## Derivation mandate — earn the substance, not just the surface values

        Traceability covers more than surface values (names, URLs, ids). Every
        *substantive claim the final artifact or conclusion depends on* must be
        derived from a real observation, never asserted from prior knowledge:
        mechanisms, design decisions, domain facts, version boundaries, ABIs,
        schemas, policy consequences, and what a component actually does at
        runtime.

        If you already "know" the answer from training, treat that as the
        strongest signal that you must derive it from a tool call anyway. The
        trace's entire value is showing HOW a fact was discovered from the live
        environment; a confident assertion the model happened to get right
        teaches nothing and is often subtly wrong. Go to the primary source —
        inspect the artifact, dump the header, read the config, enumerate the
        instances, probe the service, scrape the page — instead of substituting
        a plausible recollection.

        **A required output with many parts is many requirements, not one.**
        When the deliverable spans multiple named dimensions, each dimension
        needs its own grounding step; do not produce the whole artifact from a
        single lead. If a section of the final artifact has no supporting
        observation behind it, it was written from memory and is a bug.

        **For the decisive artifact or verdict, enumerate the candidate space.**
        Do not conclude from the first instance you find or the first hypothesis
        that fits. Enumerate every candidate that could change the answer, and
        rule out the alternatives with a direct observation before concluding.

        ## How to reverse-engineer the correct solution

        **Pre-step 0: Privately understand the target**

        Before writing final steps, privately inspect the task references and
        live runtime enough to understand the passing behavior. Build a concrete
        checklist of required artifacts, state changes, outputs, schemas, and
        edge cases.

        All of this is Rationalizer setup work. Do not include it in the final
        ``Rationale.steps`` unless the task agent itself would run the same
        command. The rationale should start from the first public action the
        task agent can take from its instruction and visible runtime.

        **Runtime boundary — critical**

        The ``runtime`` parameter passed to this method is the ALREADY-STARTED
        task dependency context. All task services (MCP servers, containers,
        endpoints) are live and reachable right now.

        **NEVER call ``task.start_deps()`` inside this method.** Doing so
        creates a NEW isolated context that resets all conversation and service
        state to zero — destroying any progress made in the existing runtime.
        Use the ``runtime`` parameter directly to reach the task environment.

        The AUT executes inside the task's runtime environment, which is set up
        by ``task.start_deps()``. That environment may inject environment
        variables, start services, or expose endpoints that the AUT uses.
        Inspect ``task.inputs``, ``task.resources``, and any env vars the
        runtime injects to discover how to reach it — do not assume any
        specific mechanism (container, process, HTTP service, etc.).

        The rationalizer's ``self.shell`` runs on the host. It can reach the
        task's runtime only through whatever interface the runtime exposes
        (e.g. env vars pointing to endpoints or exec wrappers). Never execute
        host-side commands and record their output as if they were the AUT's
        in-runtime experience — the paths, permissions, and tools available
        on the host differ from those in the task runtime.

        Never include in rationale steps:
        - Paths or commands that only work on the rationalizer host, not in
          the AUT's runtime environment
        - Errors produced by running AUT-targeted commands on the host
        - Private fixture reads from the host filesystem

        **Step 1: Read the task instruction — always first**

        The AUT's very first action is always reading its primary instruction
        from the live runtime. Access it via ``task.inputs``, ``task.resources``,
        or the runtime environment — not from host fixture paths. Do not write
        its content from memory. Store the raw output as the observation.

        From that observation, extract and record:
        - The required output(s): artifact paths, formats, schemas, or answer shape.
        - Every distinct requirement the instruction names — what to produce, what
          to cover, what constitutes success. List them explicitly; they drive Step 3.
        - The success criteria or evaluation rubric if stated.

        Do not proceed to Step 2 until these are grounded in a real observation.

        **Step 2: Inspect the visible environment**

        Inspect configuration files, environment variables, available services,
        sample data, or any other runtime state the AUT has access to.
        These observations explain why each later action is necessary.

        Limit this to task-specific unknowns NOT already given by the Ethos. Do
        not add steps that rediscover Ethos-given facts (schema, tool names and
        signatures, runtime access method, output format) — see "Ethos-given
        facts are pre-known" above. If the only thing a would-be Step 2 does is
        re-confirm what the Ethos states, drop it and go straight to executing.

        **Step 3: Execute — one step per distinct requirement**

        For each distinct requirement identified in Step 1, make a dedicated step.
        Do not collapse independent operations into a single combined call. Each
        requirement gets its own targeted action and its own observation.

        Use the tools and environment variables described in the Ethos —
        read the Ethos to discover their names and calling conventions; do not
        assume them. If the task runtime exposes services or endpoints, probe them.

        Each result may point to further work. Follow every lead: fetch what is
        referenced, verify what is claimed, check every dependency. After each
        discovery, apply the depth mandate: ask what else might exist and check
        before moving on. Keep following the chain until every requirement from
        Step 1 is covered by a real observation.

        NEVER generate facts from LLM memory. If a call fails, retry with a
        different query — do not substitute invented content.

        **Before implementing a complex artifact**, each unknown it depends on
        is a separate grounding step — format, interfaces, constraints,
        behavioral dependencies. Do not merge reconnaissance into the
        implementation step; derive each unknown from the live environment
        rather than asserting it from training knowledge. The implementation
        step's thought must cite the specific values from preceding observations
        it relies on.

        **Follow the task to the runtime's natural conclusion.** Privately
        knowing the answer is not the same as delivering it. The AUT must
        communicate findings, apply policy, and write outputs — then wait for
        the runtime's response before concluding. Only a terminal signal from
        the runtime itself (a final score, an explicit stop, a closed stream)
        ends the trace. Do not call the termination tool until that signal
        arrives.

        **Failed execution rule**: If a code cell exits with a non-zero code,
        raises an exception, or returns only an error/traceback, that step
        FAILED. The observation must record the error verbatim. The following
        step's thought must acknowledge the failure and describe the fix —
        never reference data the failed call was supposed to produce. A thought
        that cites a value from a step whose observation is an error message is
        a grounding violation.

        **Step 4: Filter the returned steps**

        Omit setup steps the task agent does not take (reading private reference
        files, reading a private reference solution, reading experimentalist-only
        metadata, inspecting held-out ground truth, starting setup wrappers).
        Start from the first real task action the agent could take from the
        instruction and available environment.

        If you used private checks to check your work, convert that into an
        equivalent public validation step before returning. Bad returned step:
        "copy tests and run `/tests/test_outputs.py`." Good returned step:
        "instantiate the class and exercise the visible methods that the task
        instruction requires, then inspect their stdout/files."

        **Step 5: Completeness gate — before returning**

        Before returning, run this checklist against your steps:

        - The task instruction was read from the live runtime in Step 1 — its
          content appears verbatim in that step's observation.
        - Every distinct requirement from the instruction has dedicated steps;
          none were collapsed into a single combined call.
        - Every claim in the final conclusion is backed by a verbatim
          observation from a real tool call, not from reasoning alone.
        - Every specific value in every thought can be pointed to in a prior
          observation or the instruction text — no values from LLM memory.
        - No step's thought references data from a prior step whose observation
          is an error message or empty — that data was never actually retrieved.
        - For every instance found (package, file, service, config value),
          you checked whether other instances exist before concluding.
        - No observation is empty or near-empty (< 20 chars) — if it is, re-run the action.
        - No observation is a clean pre-parsed summary — all are raw output.
        - No step contains a path or error that only makes sense on the
          rationalizer host — all actions and observations reflect what
          the AUT sees inside the task's runtime environment.
        - No step references private evaluator infrastructure — test scripts,
          hidden ground-truth files, oracle paths, or rationalizer setup work
          that the AUT could not reach from its task instruction alone.
        - No step rediscovers an Ethos-given fact — no container/endpoint
          identification, no grepping for tool/helper names, no inspecting the
          schema or expected output format. Those are pre-known; the trace
          starts from the first genuinely task-specific action.
        - If a visible test runner exists, it was executed and its output recorded.
        - Removing any step would leave the next step's thought unjustified.

        If any item fails, add the missing steps before returning.

        **Step 6: Return**

        ```python
        Rationale(task_name=task.id, steps=steps)
        ```

        ## Worked example

        The following synthetic example illustrates every rule above applied
        together. It is NOT a real task — it is a format reference only.

        **Scenario**: ``task.inputs["instruction"]`` says "Parse ``input.csv``,
        compute the median of column ``price``, write the result to ``out/result.txt``."
        The runtime exposes a bash-accessible environment with the input file.

        ```python
        steps = []

        # ── Step 1: read the instruction from task.inputs ───────────────────
        instruction = str(task.inputs.get("instruction", ""))
        steps.append(RationaleStep(
            thought=(
                "The task instruction is the only public starting point. "
                "Reading it from the live runtime grounds every subsequent "
                "path and requirement in a real observation."
            ),
            action="Read the task instruction.",
            observation=instruction[:600],
        ))

        # ── Step 2: inspect the runtime environment ──────────────────────────
        # Discover available files/services via task.resources and env vars
        # injected by task.start_deps() — mechanism depends on the task type.
        result = await self.shell.run("ls -la input.csv && head -3 input.csv")
        steps.append(RationaleStep(
            thought=(
                "The instruction names `input.csv` and column `price`. "
                "Previewing the file confirms it exists and reveals the "
                "actual header row before computing."
            ),
            action="Preview `input.csv` to confirm schema.",
            observation=str(result)[:600],
        ))

        # ── Step 3: first attempt fails — record error, do NOT cite its data ─
        result = await self.shell.run(
            "python3 -c \""
            "import csv, statistics; "
            "rows = list(csv.DictReader(open('input.csv'))); "
            "print(statistics.median(float(r['price']) for r in rows))\""
        )
        steps.append(RationaleStep(
            thought=(
                "The header preview showed columns `id,price,qty`. "
                "Compute the median of `price` using the confirmed column name."
            ),
            action="Compute median of `price` column.",
            observation=str(result)[:600],  # e.g. "ValueError: could not convert..."
        ))

        # ── Step 4: fix after the observed error, quote error in thought ──────
        result = await self.shell.run(
            "python3 -c \""
            "import csv, statistics; "
            "rows = list(csv.DictReader(open('input.csv'))); "
            "vals = [float(r['price']) for r in rows if r['price'].strip()]; "
            "print(statistics.median(vals))\""
        )
        steps.append(RationaleStep(
            thought=(
                "The prior observation showed `ValueError: could not convert "
                "string to float`. Blank `price` entries cause the cast to "
                "fail — skip them before computing."
            ),
            action="Re-compute median skipping blank `price` values.",
            observation=str(result)[:600],  # "47.5\n"
        ))

        # ── Step 5: write artifact using the derived value ───────────────────
        result = await self.shell.run(
            "mkdir -p out && echo 47.5 > out/result.txt && cat out/result.txt"
        )
        steps.append(RationaleStep(
            thought=(
                "The median observation returned `47.5`. The instruction says "
                "write to `out/result.txt`; always create the directory first "
                "in case it does not yet exist."
            ),
            action="Write `47.5` to `out/result.txt`.",
            observation=str(result)[:600],  # "47.5\n"
        ))

        return Rationale(task_name=task.id, steps=steps)
        ```

        Key patterns demonstrated above:
        - The instruction is read from ``task.inputs`` — never from host paths.
        - Each thought quotes the **exact string** from the prior observation.
        - Step 3 fails; step 4's thought quotes the error from step 3 and
          does NOT cite the numeric answer as if step 3 had succeeded.
        - Paths are grounded in the instruction observation, not memory.

        The exact task may instead require a running service, compiled binary,
        modified source tree, or final answer. Let the instruction and coverage
        checklist define the target; never assume an ``output.*`` artifact.

        """
        ...
