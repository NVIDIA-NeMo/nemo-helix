#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example setup helper: run Tau3 Airline and upload its traces to Intake.

This script supplies trace data for the walkthrough; Experimentalist does not require
agent repositories to include it.
"""

import argparse
import asyncio
import json
import os
import secrets
from datetime import UTC, datetime
from pathlib import Path

from nemo_experimentalist_plugin.client import make_client
from nemo_experimentalist_plugin.entities import TrialResult, local_path_from_uri
from nemo_experimentalist_plugin.experimentalist.components.evaluator.harbor import HarborDataset
from nemo_experimentalist_plugin.experimentalist.components.evaluator.harbor_native import (
    HarborEvaluatorConfig,
    HarborNativeOutcomeEvaluator,
)
from nemo_experimentalist_plugin.experimentalist.otlp import jsonl_to_protobuf, read_trace_id
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.client.errors import ConflictError, NotFoundError
from nemo_platform_plugin.intake.client import AsyncIntakeClient
from nemo_platform_plugin.intake.types import EvaluationCreateRequest, ExperimentCreateRequest
from nemo_platform_plugin.workspaces.client import AsyncWorkspacesClient
from nemo_platform_plugin.workspaces.types import CreateWorkspaceRequest

SCRIPT_DIR = Path(__file__).resolve().parent
PLUGIN_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATASET = PLUGIN_ROOT / "tmp" / "tau3-airline" / "insights"
DEFAULT_OUTPUT = PLUGIN_ROOT / "tmp" / "tau3-airline-insights"
DEFAULT_MODEL = "openai/openai/openai/gpt-5-mini"
DEFAULT_WORKSPACE = "tau3-airline"
DEFAULT_AGENT_NAME = "nemo-experimentalist-tau3-nooa"
DEFAULT_AGENT_VERSION = "1.0.0"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _evaluation_name() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"tau3-airline-{stamp}-{secrets.token_hex(2)}"


def _configure_models(*, model: str, user_model: str, api_base: str) -> None:
    api_key = os.environ.get("INFERENCE_API_KEY")
    if not api_key:
        raise RuntimeError("INFERENCE_API_KEY is required for the agent under test")

    normalized_base = api_base.rstrip("/")
    openai_base = normalized_base if normalized_base.endswith("/v1") else f"{normalized_base}/v1"
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = openai_base
    os.environ["AUT_MODEL_NAME"] = model
    os.environ["TAU2_USER_MODEL"] = user_model
    os.environ["TAU2_NL_ASSERTIONS_MODEL"] = user_model


def _resolve_harbor_output_dir(path: Path) -> tuple[Path, Path]:
    """Return the Harbor job directory and its enclosing run directory."""
    candidate = path.expanduser().resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"Harbor output directory not found: {candidate}")

    if any(child.is_dir() and (child / "result.json").is_file() for child in candidate.iterdir()):
        job_dir = candidate
        run_dir = candidate.parent.parent if candidate.parent.name == "results" else candidate
    else:
        jobs_dir = candidate / "results"
        job_dirs = (
            [
                child
                for child in jobs_dir.iterdir()
                if child.is_dir()
                and any(trial.is_dir() and (trial / "result.json").is_file() for trial in child.iterdir())
            ]
            if jobs_dir.is_dir()
            else []
        )
        if len(job_dirs) != 1:
            raise ValueError(f"{candidate} is not a Harbor output directory with exactly one job under results/")
        job_dir = job_dirs[0]
        run_dir = candidate

    if not any(child.is_dir() and (child / "result.json").is_file() for child in job_dir.iterdir()):
        raise ValueError(f"Harbor job directory contains no trial result files: {job_dir}")
    return job_dir, run_dir


def _write_upload_summary(
    run_dir: Path,
    *,
    evaluation_name: str,
    workspace: str,
    agent_name: str,
    agent_version: str,
    model: str,
    trials: list[TrialResult],
    trace_ids: dict[str, str],
) -> Path:
    summary_path = run_dir / "uploaded-traces.json"
    summary_path.write_text(
        json.dumps(
            {
                "evaluation_name": evaluation_name,
                "workspace": workspace,
                "agent_name": agent_name,
                "agent_version": agent_version,
                "model": model,
                "trace_count": len(trace_ids),
                "traces": [
                    {"trial_id": trial.id, "task_id": trial.task_id, "trace_id": trace_ids[trial.id]}
                    for trial in trials
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_path


async def _ensure_evaluation(client: AsyncNemoClient, *, workspace: str, name: str) -> None:
    """Register the Evaluation these spans name, or Intake drops them."""
    intake = AsyncIntakeClient.from_client(client)
    try:
        group = (await intake.create_experiment(workspace=workspace, body=ExperimentCreateRequest(name=name))).data()
    except ConflictError:
        group = (await intake.get_experiment(name=name, workspace=workspace)).data()
    try:
        await intake.create_evaluation(
            workspace=workspace,
            body=EvaluationCreateRequest(
                name=name,
                experiment_ids=[group.id],
                dataset_name=name,
                dataset_version="v1",
            ),
        )
    except ConflictError:
        pass


async def _upload_trials(
    client: AsyncNemoClient,
    trials: list[TrialResult],
    *,
    workspace: str,
    evaluation_name: str,
    agent_name: str,
    agent_version: str,
    model: str,
) -> dict[str, str]:
    trace_ids: dict[str, str] = {}
    intake = AsyncIntakeClient.from_client(client)

    for trial in trials:
        if trial.trace is None:
            raise RuntimeError(f"Trial {trial.id} did not produce an agent execution trace")
        trace_id = read_trace_id(trial.trace)
        if trace_id in trace_ids.values():
            raise RuntimeError(f"Trace {trace_id} was produced by more than one trial")

        attrs = {
            "nemo.evaluation.name": evaluation_name,
            "nemo.test_case.name": trial.task_id,
            "nemo.trial.id": trial.id,
            "gen_ai.agent.name": agent_name,
            "gen_ai.agent.version": agent_version,
            "gen_ai.request.model": model,
        }
        path = local_path_from_uri(trial.trace.uri, context="Agent execution trace")
        payloads = jsonl_to_protobuf(path, extra_resource_attrs=attrs)
        if not payloads:
            raise RuntimeError(f"Trial {trial.id} produced an empty agent execution trace")
        for payload in payloads:
            await intake.create_otlp_traces(content=payload, workspace=workspace)
        trace_ids[trial.id] = trace_id

    return trace_ids


async def _wait_for_traces(
    client: AsyncNemoClient,
    trace_ids: set[str],
    *,
    workspace: str,
    retries: int = 6,
) -> None:
    pending = set(trace_ids)
    intake = AsyncIntakeClient.from_client(client)
    delay = 1.0
    for attempt in range(retries):
        for trace_id in tuple(pending):
            try:
                await intake.get_trace(id=trace_id, workspace=workspace)
            except NotFoundError:
                continue
            pending.remove(trace_id)
        if not pending:
            return
        if attempt < retries - 1:
            await asyncio.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Uploaded traces did not become readable from Intake: {sorted(pending)}")


async def run(args: argparse.Namespace) -> Path:
    dataset_path = args.dataset.expanduser().resolve()
    agent_path = args.agent.expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Tau3 dataset not found: {dataset_path}")
    if not agent_path.is_dir():
        raise FileNotFoundError(f"Tau3 agent not found: {agent_path}")

    dataset = HarborDataset.from_path(dataset_path)
    if args.task_ids:
        dataset = dataset.subset(args.task_ids)
    if args.upload_dir is None and len(dataset.tasks) != args.expected_task_count:
        raise RuntimeError(
            f"Expected {args.expected_task_count} Tau3 tasks in {dataset_path}, found {len(dataset.tasks)}"
        )

    if args.upload_dir is not None:
        job_dir, run_dir = _resolve_harbor_output_dir(args.upload_dir)
        evaluator = HarborNativeOutcomeEvaluator(experiment_dir=run_dir)
        trials = list(await evaluator._trials_from_dir(job_dir, dataset.tasks))
        uploadable_trials = [trial for trial in trials if trial.status == "completed" and trial.trace is not None]
        if not uploadable_trials:
            raise RuntimeError(f"No completed Harbor trials with trace artifacts found in {job_dir}")

        evaluation_name = args.evaluation_name or run_dir.name
        client = make_client(args.base_url)
        workspaces = AsyncWorkspacesClient.from_client(client)
        try:
            (
                await workspaces.create_workspace(
                    exist_ok=True,
                    body=CreateWorkspaceRequest(
                        name=args.workspace, description="Tau3 Airline agent traces for Insights"
                    ),
                )
            ).data()
            await _ensure_evaluation(client, workspace=args.workspace, name=evaluation_name)
            trace_ids = await _upload_trials(
                client,
                uploadable_trials,
                workspace=args.workspace,
                evaluation_name=evaluation_name,
                agent_name=args.agent_name,
                agent_version=args.agent_version,
                model=args.model,
            )
            await _wait_for_traces(client, set(trace_ids.values()), workspace=args.workspace)
        finally:
            await client.close()

        summary_path = _write_upload_summary(
            run_dir,
            evaluation_name=evaluation_name,
            workspace=args.workspace,
            agent_name=args.agent_name,
            agent_version=args.agent_version,
            model=args.model,
            trials=uploadable_trials,
            trace_ids=trace_ids,
        )
        print(f"Uploaded and verified {len(trace_ids)} agent traces in workspace {args.workspace!r}.")
        print(summary_path)
        return summary_path

    evaluation_name = args.evaluation_name or _evaluation_name()
    run_dir = args.output.expanduser().resolve() / evaluation_name
    if run_dir.exists():
        raise FileExistsError(f"Output directory already exists: {run_dir}")

    _configure_models(model=args.model, user_model=args.user_model, api_base=args.api_base)
    client = make_client(args.base_url)
    workspaces = AsyncWorkspacesClient.from_client(client)
    try:
        (
            await workspaces.create_workspace(
                exist_ok=True,
                body=CreateWorkspaceRequest(name=args.workspace, description="Tau3 Airline agent traces for Insights"),
            )
        ).data()
        await _ensure_evaluation(client, workspace=args.workspace, name=evaluation_name)
        run_dir.mkdir(parents=True)

        options = HarborEvaluatorConfig(
            job_name="tau3-airline-insights",
            jobs_dir=Path("results"),
            n_attempts=1,
            n_concurrent_trials=args.concurrency,
            quiet=not args.verbose,
            agent_setup_timeout_multiplier=2.0,
            environment_build_timeout_multiplier=3.0,
        )
        result = await HarborNativeOutcomeEvaluator(experiment_dir=run_dir).run(
            agent=agent_path,
            dataset=dataset,
            options=options,
        )
        trials = list(result.trials)
        if len(trials) != args.expected_task_count:
            raise RuntimeError(
                f"Expected {args.expected_task_count} Tau3 trial results, found {len(trials)} in {run_dir}"
            )

        trace_ids = await _upload_trials(
            client,
            trials,
            workspace=args.workspace,
            evaluation_name=evaluation_name,
            agent_name=args.agent_name,
            agent_version=args.agent_version,
            model=args.model,
        )
        await _wait_for_traces(client, set(trace_ids.values()), workspace=args.workspace)
    finally:
        await client.close()

    summary_path = _write_upload_summary(
        run_dir,
        evaluation_name=evaluation_name,
        workspace=args.workspace,
        agent_name=args.agent_name,
        agent_version=args.agent_version,
        model=args.model,
        trials=trials,
        trace_ids=trace_ids,
    )
    print(f"Uploaded and verified {len(trace_ids)} agent traces in workspace {args.workspace!r}.")
    print(summary_path)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--agent", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--upload-dir",
        type=Path,
        help="Existing Harbor run directory or Harbor job-results directory to upload without rerunning trials.",
    )
    parser.add_argument("--base-url", default=os.environ.get("NMP_BASE_URL", "http://localhost:8080"))
    parser.add_argument(
        "--api-base",
        default=os.environ.get("INFERENCE_API_BASE", "https://inference-api.nvidia.com/v1"),
    )
    parser.add_argument("--model", default=os.environ.get("AUT_MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--user-model", default=os.environ.get("TAU2_USER_MODEL", DEFAULT_MODEL))
    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME)
    parser.add_argument("--agent-version", default=DEFAULT_AGENT_VERSION)
    parser.add_argument("--evaluation-name")
    parser.add_argument("--expected-task-count", type=_positive_int, default=20)
    parser.add_argument(
        "--task-id",
        dest="task_ids",
        action="append",
        help="Run only this task ID; repeat to select multiple tasks.",
    )
    parser.add_argument("--concurrency", type=_positive_int, default=4)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
