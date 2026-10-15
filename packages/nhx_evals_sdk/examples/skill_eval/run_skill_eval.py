# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A/B *skill evaluation* over the Fabric agent-eval runtime.

Runs one taskset twice through :class:`FabricAgentRuntime`:

* **baseline** — no skill;
* **treated** — the same runtime with an injected `agentskills.io <https://agentskills.io>`_ skill
  (``runtime.with_skill(skill)``), so the two arms differ in *exactly* the skill.

Each arm is a separate run (distinct ``run_id``), scored with each task's metrics — including
``SkillUsedMetric``, whose ``skill_present`` / ``skill_used`` outputs surface whether the agent
actually used the injected skill. A ``skill_present=True, skill_used=False`` row is a failure to use
the skill; comparing the other metrics baseline-vs-treated shows whether the skill helped.

Run as a module from the repository root. Needs the native NeMo Fabric stack and an ``NVIDIA_API_KEY``
whose account is provisioned for ``MODEL``. If ``python3`` on ``PATH`` is not the interpreter running
this (e.g. a venv on macOS/Homebrew), also set ``ADAPTER_PYTHON`` to it — the Fabric Hermes adapter
spawns a subprocess and otherwise falls back to bare ``python3`` (which lacks ``nemo_fabric_adapters``)::

    NVIDIA_API_KEY=... ADAPTER_PYTHON="$(pwd)/.venv/bin/python" \\
        python -m packages.nhx_evals_sdk.examples.skill_eval.run_skill_eval
"""

from __future__ import annotations

import ast
import asyncio
import logging
import re
from pathlib import Path

if __package__ in {None, ""}:
    raise SystemExit(
        "Run this example as a module from the repository root:\n"
        "  python -m packages.nhx_evals_sdk.examples.skill_eval.run_skill_eval"
    )

from nhx_evals_sdk.agent_eval.evaluator import AgentEvaluator
from nhx_evals_sdk.agent_eval.metrics import AgentPhaseSuccessMetric, SkillUsedMetric
from nhx_evals_sdk.agent_eval.results import AgentEvalResult
from nhx_evals_sdk.agent_eval.runtimes.fabric.runtime import FabricAgentRuntime
from nhx_evals_sdk.agent_eval.runtimes.fabric.skills import AgentSkill, SkillInjectionError
from nhx_evals_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrialStatus
from nhx_evals_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult

# Model under evaluation. A mid-size NVIDIA Nemotron is capable enough to follow the injected skill's
# guidelines (so the A/B shows signal) without the latency of a frontier model. Served via
# ``integrate.api.nvidia.com`` (provider ``nvidia``), so it needs ``NVIDIA_API_KEY``.
MODEL = "nvidia/nemotron-3-super-120b-a12b"


# Score off the *parsed function signature*, not the whole reply text, so a convention merely mentioned
# in prose doesn't count. Primary path: AST-parse the reply's fenced code blocks (robust to bracketed
# annotations, multi-line signatures, ``*args``). Fallback: a ``def`` regex for code that doesn't parse
# cleanly (truncated / pseudo-code).
_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)
_DEF_RE = re.compile(r"def\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)")
# French parameter names the gcd solution may legitimately use. The skill mandates French parameter
# names in general; gcd has no positional-index parameter, so its ``enieme`` rule does not apply here.
# Illustrative, not exhaustive — extend if a run uses a French word not listed.
_FRENCH_PARAMS = frozenset(
    {
        "premier",
        "premiere",
        "deuxieme",
        "second",
        "seconde",
        "nombre",
        "nombres",
        "entier",
        "entiers",
        "valeur",
        "valeurs",
        "numero",
        "chiffre",
        "terme",
    }
)


def _signatures_via_ast(text: str) -> list[tuple[str, list[str]]]:
    """``(name, [param, ...])`` for every function defined in the reply's code blocks, via ``ast``."""
    functions: list[tuple[str, list[str]]] = []
    for block in _CODE_BLOCK.findall(text) or [text]:
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                names = [arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
                if args.vararg:
                    names.append(args.vararg.arg)
                if args.kwarg:
                    names.append(args.kwarg.arg)
                functions.append((node.name, [name for name in names if name != "self"]))
    return functions


def _param_names(raw: str) -> list[str]:
    """Bare parameter names from a raw ``def`` parameter list — the regex fallback path."""
    params: list[str] = []
    for chunk in raw.split(","):
        # "enieme: int = 0" / "*args" -> "enieme" / "args"
        name = chunk.split(":", 1)[0].split("=", 1)[0].strip().lstrip("*").strip()
        if name and name != "self":
            params.append(name)
    return params


def _scored_signature(text: str) -> tuple[str, list[str]]:
    """Return the ``(name, [param, ...])`` of the function the guidelines govern.

    A reply often defines helper functions too, so scoring the first/last ``def`` is wrong. The skill
    mandates the answer be named ``supercool_...``, so prefer the first ``supercool_``-named definition;
    fall back to the first function defined, or ``("", [])`` if the reply defines none.
    """
    signatures = _signatures_via_ast(text)
    if not signatures:  # code didn't parse (truncated / pseudo-code) — best-effort regex
        signatures = [(m.group("name"), _param_names(m.group("params"))) for m in _DEF_RE.finditer(text)]
    if not signatures:
        return "", []
    for name, params in signatures:
        if name.startswith("supercool_"):
            return name, params
    return signatures[0]


class GuidelinesMetric:
    """Task-authored (positional-index tasks): does the solution follow the *Supercool Coding Guidelines*?

    Scored off the defined function's signature (so a convention only *mentioned* in prose does not
    count):

    * ``supercool_prefix`` — the function name starts with ``supercool_`` (weakly guessable, reported
      on its own);
    * ``enieme_param`` — the positional-index parameter is named ``enieme`` (French for "nth"), the
      unguessable skill-only signal;
    * ``follows_guidelines`` — both.

    Use for a task whose function takes a positional index (e.g. "nth digit of pi"). Expected to pass
    only in the treated (with-skill) arm.
    """

    @property
    def type(self) -> str:
        return "follows_guidelines"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [
            MetricOutputSpec.boolean("supercool_prefix"),
            MetricOutputSpec.boolean("enieme_param"),
            MetricOutputSpec.boolean("follows_guidelines"),
        ]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        name, params = _scored_signature(input.candidate.output_text or "")
        prefix = name.startswith("supercool_")
        enieme = "enieme" in params
        return MetricResult(
            outputs=[
                MetricOutput(name="supercool_prefix", value=prefix),
                MetricOutput(name="enieme_param", value=enieme),
                MetricOutput(name="follows_guidelines", value=prefix and enieme),
            ]
        )


class GcdGuidelinesMetric:
    """Task-authored (gcd): the *Supercool Coding Guidelines* for a task with **no** positional index.

    The skill's ``enieme`` rule is specific to a positional-index parameter, which gcd (two integers)
    does not have — so this checks the skill's *general* parameter rule instead: names must be French.

    * ``supercool_prefix`` — the function name starts with ``supercool_``;
    * ``french_params`` — every parameter name is a French word (:data:`_FRENCH_PARAMS`); baseline arms
      use English names (``a``/``b``), so this is the skill-only signal;
    * ``follows_guidelines`` — both.
    """

    @property
    def type(self) -> str:
        return "follows_guidelines"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [
            MetricOutputSpec.boolean("supercool_prefix"),
            MetricOutputSpec.boolean("french_params"),
            MetricOutputSpec.boolean("follows_guidelines"),
        ]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        name, params = _scored_signature(input.candidate.output_text or "")
        prefix = name.startswith("supercool_")
        french = bool(params) and all(param.lower() in _FRENCH_PARAMS for param in params)
        return MetricResult(
            outputs=[
                MetricOutput(name="supercool_prefix", value=prefix),
                MetricOutput(name="french_params", value=french),
                MetricOutput(name="follows_guidelines", value=prefix and french),
            ]
        )


def skill_eval_tasks() -> list[AgentEvalTask]:
    """Two "write a function following the Supercool Coding Guidelines" tasks.

    The guidelines (``supercool_`` prefix + French parameter names) live only in the injected
    ``supercool-guidelines`` skill and are not inferable from the prompt, so ``follows_guidelines``
    should pass only in the treated (with-skill) arm — a measured, skill-dependent difference. Each task
    gets the guidelines metric that matches its parameters: the index task checks for the ``enieme``
    positional-index name; gcd (no index param) checks that its parameters are French words.
    """

    def build(task_id: str, task: str, guidelines: GuidelinesMetric | GcdGuidelinesMetric) -> AgentEvalTask:
        intent = f"Write a Python function that {task}, following the Supercool Coding Guidelines."
        return AgentEvalTask(
            id=task_id,
            intent=intent,
            inputs={"instruction": f"{intent} Include the complete function in your reply."},
            metrics=[AgentPhaseSuccessMetric(), SkillUsedMetric(), guidelines],
        )

    return [
        build("pi-digit", "returns the nth digit of pi", GuidelinesMetric()),
        build("gcd", "returns the greatest common divisor of two integers", GcdGuidelinesMetric()),
    ]


async def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    tasks = skill_eval_tasks()
    current_dir = Path(__file__).resolve().parent
    output_dir = current_dir / "skill-eval-output"

    fabric_config = {
        "metadata": {"name": "skill-eval-hermes"},
        "harness": {
            "adapter_id": "nvidia.fabric.hermes",
            "resolution": "preinstalled",
        },
        "models": {"default": {"provider": "nvidia", "model": MODEL}},
        "runtime": {"input_schema": "chat", "output_schema": "message", "max_turns": 50},
    }

    baseline_runtime = FabricAgentRuntime(config=fabric_config)
    try:
        # Load the bundled skill inside the guarded block so a packaging mistake (e.g. a missing
        # SKILL.md) prints the friendly message instead of a traceback.
        skill = AgentSkill.from_directory(current_dir / "skills" / "supercool-guidelines")
        baseline = await AgentEvaluator().run(
            tasks=tasks,
            target=baseline_runtime,
            config=AgentEvalRunConfig(run_id="baseline", work_dir=output_dir / "baseline"),
        )
        baseline.persist(write_dashboard=False)
        treated = await AgentEvaluator().run(
            tasks=tasks,
            target=baseline_runtime.with_skill(
                skill
            ),  # We include the skill in the treated arm, so the two runs differ in *exactly* the skill.
            config=AgentEvalRunConfig(run_id="treated", work_dir=output_dir / "treated"),
        )
        treated.persist(write_dashboard=False)
    except SkillInjectionError as exc:
        print(f"skill eval failed to load the bundled skill: {exc}")
        return 1
    except RuntimeError as exc:
        print(f"skill eval failed: {exc}")
        print("This example needs the native NeMo Fabric stack and NVIDIA_API_KEY.")
        return 1

    # A failed trial produces no scorable output, so it would silently vanish from the tallies below and
    # make the A/B look empty-but-fine. Surface failures loudly and treat any as a non-zero exit — a
    # broken harness/model/credential must not read as "0/0, all good".
    def failed_trials(result: AgentEvalResult) -> list[tuple[str, str]]:
        failures: list[tuple[str, str]] = []
        for trial in result.trials:
            if trial.status == AgentEvalTrialStatus.FAILED:
                meta = trial.metadata or {}
                reason = str(meta.get("error") or meta.get("error_type") or "unknown error")
                failures.append((trial.task_id, reason))
        return failures

    all_failures = [("baseline", tid, err) for tid, err in failed_trials(baseline)]
    all_failures += [("treated", tid, err) for tid, err in failed_trials(treated)]
    if all_failures:
        print(f"\n⚠️  {len(all_failures)} trial(s) FAILED — the A/B numbers below are unreliable:")
        for arm, task_id, err in all_failures:
            print(f"  [{arm}] {task_id}: {err}")

    # Tally each boolean metric output (true/total) per arm, then print baseline vs. treated.
    def rates(result: AgentEvalResult) -> dict[str, tuple[int, int]]:
        counts: dict[str, tuple[int, int]] = {}
        for score in result.scores:
            for output in score.outputs:
                if isinstance(output.value, bool):
                    key = f"{score.metric_type}.{output.name}"
                    true_count, total = counts.get(key, (0, 0))
                    counts[key] = (true_count + int(output.value), total + 1)
        return counts

    base, treat = rates(baseline), rates(treated)
    print(
        f"\nHarness: {fabric_config['harness']['adapter_id']}   model: {MODEL}   tasks: {baseline.summary.task_count}"
    )
    print(f"runs: {baseline.run_id} (baseline) vs {treated.run_id} (treated)\n")
    width = max((len(key) for key in set(base) | set(treat)), default=len("metric.output"))
    print(f"  {'metric.output'.ljust(width)}   baseline   with-skill")
    for key in sorted(set(base) | set(treat)):
        bt, bn = base.get(key, (0, 0))
        tt, tn = treat.get(key, (0, 0))
        print(f"  {key.ljust(width)}   {bt}/{bn}        {tt}/{tn}")
    print(f"\noutput: {output_dir}")
    return 1 if all_failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
