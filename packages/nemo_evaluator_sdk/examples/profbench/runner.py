# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line runner for the ProfBench agent-eval example."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid
from datetime import datetime
from enum import StrEnum
from pathlib import Path

if __package__ in {None, ""}:
    raise SystemExit(
        "Run ProfBench as a module from the repository root:\n"
        "  python -m packages.nemo_evaluator_sdk.examples.profbench.runner"
    )

from nemo_evaluator_sdk.agent_eval.evaluator import AgentEvaluator
from nemo_evaluator_sdk.agent_eval.results import AgentEvalResult
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime import FabricAgentRuntime
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTarget, AgentEvalTrial
from nemo_evaluator_sdk.values import InferenceParams, Model, RunConfigOnlineModel, SecretRef

from .profbench import (
    PROFBENCH_DATASET_URL,
    PROFBENCH_METRIC_ID,
    PROFBENCH_METRIC_TYPE,
    ProfBenchModelJudge,
    load_profbench,
    write_example_dashboards,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "profbench-agent-eval-output"
DEFAULT_MODEL_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL_NAME = "nvidia/nemotron-3.5-lightning-30b-a3b"
DEFAULT_API_KEY_SECRET = os.getenv("NHX_EVALUATOR_DEFAULT_API_KEY_SECRET", "NVIDIA_API_KEY")
DEFAULT_FABRIC_CODEX_MODEL = "gpt-5.4"


class AgentChoice(StrEnum):
    MODEL = "model"
    FABRIC_CODEX = "fabric-codex"


class ProfBenchMode(StrEnum):
    BASELINE = "baseline"
    LIVE_JUDGE = "live-judge"
    LIVE_CANDIDATE = "live-candidate"


def configure_example_logging() -> None:
    """Enable SDK progress logs when this example file is executed directly."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("nemo_evaluator_sdk.inference").setLevel(logging.WARNING)


async def run_profbench_mode(
    mode: ProfBenchMode,
    *,
    limit: int | None,
    output_root: str | Path | None = None,
    run_instance_id: str | None = None,
    agent: AgentChoice = AgentChoice.MODEL,
    agent_model: str | None = None,
) -> None:
    """Run one ProfBench mode.

    - ``baseline``: score the dataset's recorded responses against their cached fulfilment labels.
    - ``live-judge``: re-score those recorded responses with a live LLM judge.
    - ``live-candidate``: generate fresh candidate responses, then score them with a live judge.
    """
    _print_example_separator(mode.value)

    output_root = _resolve_profbench_output_root(output_root)
    run_instance_id = run_instance_id or _new_profbench_run_instance_id()
    output_dir = _profbench_output_dir(output_root, run_instance_id, mode.value)

    judge = None if mode is ProfBenchMode.BASELINE else ProfBenchModelJudge(model=_judge_model())
    benchmark = load_profbench(
        _profbench_source(),
        limit=limit,
        judge=judge,
        evidence_dir=output_dir / "evidence",
        include_cached_fulfilments=mode is ProfBenchMode.BASELINE,
    )

    target: AgentEvalTarget | None = None
    trials: list[AgentEvalTrial] | None = None
    params: RunConfigOnlineModel | None = None
    benchmark_labels = {key: str(value) for key, value in benchmark.metadata.items()}
    tasks = benchmark.tasks
    if mode is ProfBenchMode.LIVE_CANDIDATE:
        target, params, score_source = _live_candidate_target(
            agent=agent, agent_model=agent_model, output_dir=output_dir
        )
        benchmark_labels["score_source"] = score_source
        if agent is AgentChoice.FABRIC_CODEX:
            tasks = [_as_candidate_task(task) for task in tasks]
    else:
        trials = benchmark.trials
        if mode is ProfBenchMode.LIVE_JUDGE:
            benchmark_labels["score_source"] = "live_judge"

    result = await AgentEvaluator().run(
        tasks=tasks,
        trials=trials,
        target=target,
        config=AgentEvalRunConfig(
            work_dir=output_dir,
            run_id=f"{run_instance_id}-{mode.value}",
            params=params,
            labels=benchmark_labels,
        ),
    )
    # This example renders its own dashboards below, so persistence skips the built-in one.
    result.persist(write_dashboard=False)
    sdk_dashboard_path, dashboard_path = write_example_dashboards(result, output_dir)

    overall = _profbench_overall(result)
    print(f"ProfBench tasks: {result.summary.task_count}")
    print(f"ProfBench trials: {result.summary.trial_count}")
    print(f"Overall score: {overall:.3f}" if overall is not None else "Overall score: n/a")
    print(f"Aggregated scores: {result.summary.scores.model_dump(mode='json')}")
    print(f"SDK dashboard: {sdk_dashboard_path}")
    print(f"Dashboard: {dashboard_path}")


async def run_examples(
    *,
    limit: int | None,
    run_live_judge: bool,
    run_live_candidate: bool,
    output_root: str | Path | None = None,
    run_instance_id: str | None = None,
    agent: AgentChoice = AgentChoice.MODEL,
    agent_model: str | None = None,
) -> None:
    """Execute the enabled ProfBench agent-eval modes under one shared run folder."""
    output_root = _resolve_profbench_output_root(output_root)
    run_instance_id = run_instance_id or _new_profbench_run_instance_id()
    run_output_dir = Path(output_root) / run_instance_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"ProfBench output root: {output_root}")
    print(f"ProfBench run instance: {run_instance_id}")

    await run_profbench_mode(
        ProfBenchMode.BASELINE,
        limit=limit,
        output_root=output_root,
        run_instance_id=run_instance_id,
    )

    if run_live_judge:
        await run_profbench_mode(
            ProfBenchMode.LIVE_JUDGE,
            limit=limit,
            output_root=output_root,
            run_instance_id=run_instance_id,
        )
    else:
        print("Skipping live ProfBench judge example. Remove --no-run-live-judge to run it.")

    if run_live_candidate:
        await run_profbench_mode(
            ProfBenchMode.LIVE_CANDIDATE,
            limit=limit,
            output_root=output_root,
            run_instance_id=run_instance_id,
            agent=agent,
            agent_model=agent_model,
        )
    else:
        print("Skipping live ProfBench candidate example. Remove --no-run-live-candidate to run it.")


def _profbench_source() -> str:
    return os.getenv("NEMO_EVALUATOR_PROFBENCH_SOURCE", PROFBENCH_DATASET_URL)


def _profbench_limit_from_args(limit: int) -> int | None:
    return None if limit == 0 else limit


def _profbench_overall(result: AgentEvalResult) -> float | None:
    """Return the mean ProfBench rubric score from the run summary, if present."""
    score_name = f"{PROFBENCH_METRIC_TYPE}.{PROFBENCH_METRIC_ID}"
    for score in result.summary.scores.scores:
        if score.name == score_name:
            return score.mean
    return None


def _resolve_profbench_output_root(output_dir: str | Path | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser()
    env_output_dir = os.getenv("NEMO_EVALUATOR_PROFBENCH_OUTPUT_DIR")
    if env_output_dir:
        return Path(env_output_dir).expanduser()
    return DEFAULT_OUTPUT_DIR


def _new_profbench_run_instance_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-1]
    return f"{timestamp}_{uuid.uuid4().hex[:6]}"


def _profbench_output_dir(output_root: str | Path, run_instance_id: str, mode: str) -> Path:
    return Path(output_root).expanduser() / run_instance_id / mode


def _evaluated_model(model_name: str | None = None) -> Model:
    return Model(
        url=DEFAULT_MODEL_URL,
        name=model_name or DEFAULT_MODEL_NAME,
        api_key_secret=SecretRef(root=DEFAULT_API_KEY_SECRET),
    )


def _judge_model() -> Model:
    return Model(
        url=DEFAULT_MODEL_URL,
        name=DEFAULT_MODEL_NAME,
        api_key_secret=SecretRef(root=DEFAULT_API_KEY_SECRET),
    )


#: Fabric config for the coding-agent candidate: the Codex CLI harness, run once per task as a
#: subprocess. ``sandbox: read-only`` because ProfBench grades a text answer and the agent has no
#: reason to write files. A plain mapping (rather than nemo_fabric's typed config) keeps this module
#: importable without the native Fabric stack installed.
PROFBENCH_FABRIC_CODEX_CONFIG = {
    "metadata": {"name": "profbench-candidate"},
    "harness": {"adapter_id": "nvidia.fabric.codex", "settings": {"sandbox": "read-only"}},
    # The Codex adapter refuses to start without a model provider, so the config carries a default
    # rather than deferring to the CLI's own; `--agent-model` overrides it.
    "models": {"default": {"provider": "openai", "model": DEFAULT_FABRIC_CODEX_MODEL}},
    "runtime": {"mode": "oneshot", "transport": "cli"},
}

#: ProfBench grades one block of answer text against a rubric, so tool logs and commentary read as a
#: worse answer. A chat model gives a bare answer already; a coding agent has to be told.
PROFBENCH_CANDIDATE_PREAMBLE = (
    "Answer the task below. Return only the final answer text; do not include analysis, "
    "markdown fences, tool logs, or commentary.\n\n"
)


def _as_candidate_task(task: AgentEvalTask) -> AgentEvalTask:
    """Prefix a task's instruction with the answer-only framing, leaving grading untouched.

    ``FabricAgentRuntime`` sends ``inputs['instruction']`` verbatim, so the framing has to live in
    the task. Only the agent arm gets it: the baseline and live-judge arms score recorded responses
    that were never prompted this way, and re-framing them would change what is being compared.
    """
    inputs = dict(task.inputs)
    # Read through ``agent_prompt`` rather than ``inputs`` directly: it rejects a task with no
    # instruction, and prefixing the preamble onto an empty one would make that value truthy and
    # silently run the agent on the preamble alone.
    inputs["instruction"] = PROFBENCH_CANDIDATE_PREAMBLE + task.agent_prompt()
    return task.model_copy(update={"inputs": inputs})


def _live_candidate_target(
    *, agent: AgentChoice, agent_model: str | None, output_dir: Path
) -> tuple[AgentEvalTarget, RunConfigOnlineModel | None, str]:
    if agent is AgentChoice.MODEL:
        return (
            _evaluated_model(agent_model),
            RunConfigOnlineModel(parallelism=2, inference=InferenceParams(temperature=0.0, max_tokens=32768)),
            "fresh_candidate_and_live_judge",
        )
    # Trajectory capture is off: it needs the nemo-relay gateway, and the rubric judge scores the
    # answer text, not the agent's steps.
    return (
        FabricAgentRuntime(
            config=PROFBENCH_FABRIC_CODEX_CONFIG,
            model=agent_model,
            work_root=output_dir / "evidence" / "fabric",
            capture_trajectory=False,
        ),
        None,
        "fabric_codex_candidate_and_live_judge",
    )


def _print_example_separator(name: str) -> None:
    edge = "====="
    middle_line = f"{edge} {name} {edge}"
    rule = "=" * len(middle_line)
    print(f"\n{rule}\n{middle_line}\n{rule}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ProfBench agent-eval examples.")
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Maximum number of ProfBench tasks to evaluate (0 = no limit). Default: 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Root directory for ProfBench outputs. "
            "Defaults to NEMO_EVALUATOR_PROFBENCH_OUTPUT_DIR or the example output directory."
        ),
    )
    parser.add_argument(
        "--run-live-judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score the recorded ProfBench responses with a live LLM judge after the baseline example.",
    )
    parser.add_argument(
        "--run-live-candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate fresh candidate responses from the configured model, then score them with a live LLM judge.",
    )
    parser.add_argument(
        "--agent",
        type=AgentChoice,
        choices=list(AgentChoice),
        default=AgentChoice.MODEL,
        help=(
            "Candidate for live-candidate mode. 'model' calls the chat-completions model directly; "
            "'fabric-codex' drives the Codex CLI through NeMo Fabric."
        ),
    )
    parser.add_argument(
        "--agent-model",
        default=None,
        help=(
            "Model for the live candidate. With --agent model it overrides the evaluated model name; "
            "with --agent fabric-codex it is a `provider/model` slug applied to the Fabric config "
            "(the harness default is used when omitted)."
        ),
    )
    args = parser.parse_args()
    configure_example_logging()

    asyncio.run(
        run_examples(
            limit=_profbench_limit_from_args(args.limit),
            run_live_judge=bool(args.run_live_judge),
            run_live_candidate=bool(args.run_live_candidate),
            output_root=args.output_dir,
            agent=args.agent,
            agent_model=args.agent_model,
        )
    )
