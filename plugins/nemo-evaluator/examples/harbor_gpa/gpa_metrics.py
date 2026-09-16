# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent GPA metrics: nine LLM-judged scores over an agent's trajectory.

The dimensions follow TruLens's Agent GPA framework (Goal, Plan, Action). Every metric here is the
same small class, ``GpaMetric``, pointed at a different rubric. A metric:

1. reads the trial's ATIF trajectory from ``input.candidate.evidence``,
2. renders it as text a judge can read,
3. asks the judge for a 0-3 score and a written reason,
4. returns ``score`` (normalized to 0-1) and ``reason`` as its outputs.

That is the whole Evaluator ``Metric`` protocol: a ``type`` name, an ``output_spec()``, and an
async ``compute_scores(input)``. Nothing here is Harbor-specific.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence
from pathlib import Path

from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask, SemanticReducer, SemanticView, ViewSignal
from nemo_evaluator_sdk.metrics.protocol import (
    Metric,
    MetricDiagnostic,
    MetricInput,
    MetricOutput,
    MetricOutputSpec,
    MetricResult,
)
from nemo_evaluator_sdk.values.atif import Agent, Observation, ObservationResult, Step, ToolCall, Trajectory
from nemo_evaluator_sdk.values.evidence import EVIDENCE_TRACE, CandidateEvidence
from openai import AsyncOpenAI

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Super answers a judge prompt in ~2s on build.nvidia.com; Lightning takes 15-50s because it reasons at length.
DEFAULT_JUDGE_MODEL = "nvidia/nemotron-3-super-120b-a12b"

# What a perfect (3) and a failing (0) trajectory look like, per dimension. The judge sees the
# whole trial every time; the rubric tells it what to look at.
RUBRICS: dict[str, tuple[str, str]] = {
    # Goal
    "answer_correctness": (
        "The trace and final answer show the grader's goal fully reached.",
        "The goal is not reached, or the trace gives no evidence it was.",
    ),
    "answer_relevance": (
        "The final answer addresses exactly what the instruction asked.",
        "The final answer is missing, off-topic, or answers a different question.",
    ),
    "groundedness": (
        "Every claim in the final answer is supported by something the agent observed in a tool result.",
        "The final answer asserts things the agent never observed, or contradicts what it observed.",
    ),
    # Plan
    "plan_quality": (
        "The plan breaks the instruction into clear, necessary steps the available tools can carry out.",
        "The plan cannot reach the goal, omits critical steps, or includes irrelevant ones.",
    ),
    "plan_adherence": (
        "Every planned step was carried out in order; any deviation was explained by something observed.",
        "Planned steps were skipped, reordered, or replaced without explanation.",
    ),
    # Action
    "tool_selection": (
        "For every step the agent chose the most suitable tool, and used a tool only where one was needed.",
        "The agent chose ill-suited tools, ignored tools the task called for, or used tools needlessly.",
    ),
    "tool_calling": (
        "Every tool call had valid, complete arguments; results were read faithfully; errors were handled.",
        "Calls had missing or invalid arguments, results were ignored or misread, errors went unnoticed.",
    ),
    "execution_efficiency": (
        "Every action was executed once, in a sensible order, with no repetition or busywork.",
        "The run is dominated by loops, repeated calls, or steps that made no progress.",
    ),
    "logical_consistency": (
        "Every action and claim follows from what came before; nothing is fabricated or assumed.",
        "Statements are unsupported by prior context, or key facts and actions are invented.",
    ),
}
GPA_METRICS = tuple(RUBRICS)


# --- reading the trajectory ---------------------------------------------------------------------


async def read_trajectory(input: MetricInput) -> Trajectory | None:
    """The trial's trajectory, or ``None`` when the runner recorded none.

    The ATIF trace is the primary source. Relay's ATIF projection for deepagents currently
    collapses the whole run into one ``LangGraph`` step (nemo-relay 0.7), so when that is all the
    trace holds, the steps are rebuilt from the message list in Fabric's result file instead.
    """
    evidence = input.candidate.evidence
    if evidence is None:
        return None
    trajectory = None
    if evidence.get(EVIDENCE_TRACE) is not None:
        try:
            trajectory = await (await evidence.trace(EVIDENCE_TRACE, format="atif")).trace()
        except KeyError:
            pass
    if trajectory is None or len(trajectory.steps) <= 1:
        return trajectory_from_fabric_result(evidence) or trajectory
    return trajectory


def trajectory_from_fabric_result(evidence: CandidateEvidence) -> Trajectory | None:
    """Rebuild ATIF steps from the ``messages`` list in Fabric's ``fabric-result-*.json`` artifact."""
    name = next((n for n in evidence.names() if "fabric-result-" in n and n.endswith(".json")), None)
    descriptor = evidence.get(name) if name else None
    if descriptor is None or descriptor.ref is None:
        return None
    output = json.loads(Path(descriptor.ref).read_text()).get("output", {})
    steps: list[Step] = []
    for message in output.get("messages", []):
        role, content = message.get("role"), str(message.get("content") or "")
        if role == "human":
            steps.append(Step(source="user", message=content))
        elif role == "ai":
            calls = [
                ToolCall(tool_call_id=c.get("id"), function_name=c["name"], arguments=c.get("args"))
                for c in message.get("tool_calls") or []
            ]
            steps.append(Step(source="agent", message=content, tool_calls=calls or None))
        elif role == "tool" and steps:
            step = steps[-1]
            step.observation = step.observation or Observation()
            step.observation.results.append(ObservationResult(content=content))
    if not steps:
        return None
    for number, step in enumerate(steps, start=1):
        step.step_id = number
    return Trajectory(
        schema_version="ATIF-v1.7",
        agent=Agent(name="deepagents", version="fabric-result", model_name=output.get("model")),
        steps=steps,
    )


def render_trajectory(trajectory: Trajectory, max_chars: int = 600) -> str:
    """The trajectory as numbered steps: messages, tool calls, and tool results, each clipped."""

    def clip(value: object) -> str:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        return text if len(text) <= max_chars else text[:max_chars] + "... [truncated]"

    lines = []
    for number, step in enumerate(trajectory.steps, start=1):
        lines.append(f"[{number}] {step.source}: {clip(step.message)}")
        for call in step.tool_calls or []:
            lines.append(f"    tool_call {call.function_name}({clip(call.arguments or {})})")
        for result in step.observation.results if step.observation else []:
            lines.append(f"    result: {clip(result.content)}")
    return "\n".join(lines)


def describe_trial(input: MetricInput, trajectory: Trajectory) -> str:
    """Everything a judge may need, in one prompt: the instruction, goal, tools, plan, trace, and answer."""
    instruction = input.row.data.get("inputs", {}).get("instruction", "")
    goal = input.row.data.get("reference", {}).get("goal", "(no grader goal given)")

    agent = trajectory.agent
    tools = json.dumps(agent.tool_definitions, default=str) if agent and agent.tool_definitions else "(not recorded)"

    # deepagents states its plan through the write_todos tool; those calls are the plan.
    plans = [
        json.dumps(c.arguments)
        for s in trajectory.steps
        for c in s.tool_calls or []
        if c.function_name == "write_todos"
    ]
    plan = "\n".join(plans) if plans else "(no explicit plan; infer it from the trace)"

    answer = next((s.message for s in reversed(trajectory.steps) if s.source == "agent" and s.message.strip()), "")

    return (
        f"### Instruction\n{instruction}\n\n"
        f"### Grader's goal (never shown to the agent)\n{goal}\n\n"
        f"### Tools available\n{tools}\n\n"
        f"### Stated plan\n{plan}\n\n"
        f"### Trace\n{render_trajectory(trajectory)}\n\n"
        f"### Final answer\n{answer or '(none)'}"
    )


# --- the judge ----------------------------------------------------------------------------------


class Judge:
    """One chat call to an OpenAI-compatible endpoint that returns ``{"score": 0-3, "reason": ...}``."""

    def __init__(
        self, model: str = DEFAULT_JUDGE_MODEL, *, base_url: str = NVIDIA_BASE_URL, api_key_env: str = "NVIDIA_API_KEY"
    ):
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=os.environ[api_key_env])
        self.limit = asyncio.Semaphore(1)  # one judge call at a time: build.nvidia.com allows little concurrency

    async def score(self, best: str, worst: str, trial: str) -> tuple[float, str]:
        """Return the score normalized to 0-1 and the judge's reason."""
        system = (
            "You are auditing an autonomous agent's execution trace. Judge only from the material provided.\n"
            f"Score 3 means: {best}\nScore 0 means: {worst}\nUse 1 and 2 for partial satisfaction.\n"
            'Reply with JSON only: {"score": <0-3>, "reason": "<your reasoning, citing trace steps>"}'
        )
        async with self.limit:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": trial}],
                temperature=0.0,
                max_tokens=4096,
            )
        text = response.choices[0].message.content or ""
        start = text.find("{")
        if start == -1:
            raise ValueError(f"judge did not reply with JSON: {text[:200]!r}")
        verdict, _ = json.JSONDecoder().raw_decode(text, start)
        score = max(0.0, min(3.0, float(verdict["score"]))) / 3.0
        return score, str(verdict.get("reason", ""))


# --- the metric ---------------------------------------------------------------------------------


class GpaMetric:
    """One GPA dimension. Emits ``score`` and ``reason``; leaves a trial without a trajectory unmeasured."""

    def __init__(self, name: str, judge: Judge) -> None:
        self.name = name
        self.judge = judge

    @property
    def type(self) -> str:
        return self.name

    def output_spec(self) -> list[MetricOutputSpec]:
        # required=False: a missing output means "could not measure", not "scored zero".
        return [
            MetricOutputSpec.continuous_score("score", required=False),
            MetricOutputSpec.label("reason", required=False),
        ]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        trajectory = await read_trajectory(input)
        if trajectory is None:
            return MetricResult(outputs=[])
        best, worst = RUBRICS[self.name]
        try:
            score, reason = await self.judge.score(best, worst, describe_trial(input, trajectory))
        except Exception as error:  # a broken judge reply is recorded, not scored as zero
            return MetricResult(outputs=[], diagnostics=[MetricDiagnostic(message=str(error))])
        return MetricResult(
            outputs=[MetricOutput(name="score", value=score), MetricOutput(name="reason", value=reason)]
        )


def gpa_metrics(judge: Judge) -> list[GpaMetric]:
    """All nine GPA metrics, bound to one judge."""
    return [GpaMetric(name, judge) for name in GPA_METRICS]


# --- attaching to Harbor tasks -------------------------------------------------------------------


def attach_gpa(tasks: Sequence[AgentEvalTask], metrics: Sequence[Metric]) -> list[AgentEvalTask]:
    """Give each discovered Harbor task its ``gpa.json`` goal, the GPA metrics, and a ``gpa`` view.

    ``discover_harbor_tasks`` already attached ``HarborRewardMetric`` (the verifier's reward); it
    stays. ``view.gpa`` is the mean of the nine ``score`` outputs.
    """
    view = SemanticView(
        reducer=SemanticReducer.MEAN,
        signals=[ViewSignal(metric=metric.type, output="score") for metric in metrics],
    )
    attached = []
    for task in tasks:
        goal = json.loads((Path(task.metadata["harbor_task_dir"]) / "gpa.json").read_text())
        attached.append(
            AgentEvalTask(
                id=task.id,
                intent=task.intent,
                inputs=task.inputs,
                reference=goal,
                metrics=[*task.metrics, *metrics],
                views={"gpa": view},
                metadata=task.metadata,
            )
        )
    return attached
