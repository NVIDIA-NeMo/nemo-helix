# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate one group's tasks and upload their traces to Intake.

Insight mode needs an Insight whose trace_refs resolve through the Platform
client, so a group's own tasks double as the trace source: evaluate them, then
ingest the trace each trial wrote.

The resource attributes attached here are not decoration. `gen_ai.agent.name` is
how the analyst later finds these traces, so the Insight generated from them is
only possible because they are set.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from nemo_experimentalist_plugin.client import make_client
from nemo_experimentalist_plugin.entities import TrialResult, local_path_from_uri
from nemo_experimentalist_plugin.experimentalist.components.evaluator.harbor_native import (
    HarborDataset,
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

AGENT_NAME = "smoke-agent"
AGENT_VERSION = "1.0.0"
POLL_ATTEMPTS = 30
POLL_DELAY_SECONDS = 2.0


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
    group: str,
) -> dict[str, str]:
    """Upload each trial's trace; return {task_id: trace_id}."""
    trace_ids: dict[str, str] = {}
    intake = AsyncIntakeClient.from_client(client)

    for trial in trials:
        if trial.trace is None:
            raise RuntimeError(f"Trial {trial.id} produced no trace; the agent must write /app/traces")
        trace_id = read_trace_id(trial.trace)
        if trace_id in trace_ids.values():
            raise RuntimeError(f"Trace {trace_id} was produced by more than one trial")

        attrs = {
            "nemo.experiment.id": group,
            "nemo.test_case.name": trial.task_id,
            "nemo.trial.id": trial.id,
            "gen_ai.agent.name": AGENT_NAME,
            "gen_ai.agent.version": AGENT_VERSION,
        }
        path = local_path_from_uri(trial.trace.uri, context="Agent execution trace")
        payloads = jsonl_to_protobuf(path, extra_resource_attrs=attrs)
        if not payloads:
            raise RuntimeError(f"Trial {trial.id} produced an empty trace")
        for payload in payloads:
            await intake.create_otlp_traces(content=payload, workspace=workspace)
        # Every recording run uses one attempt per task.  Keep the task id here
        # so the caller can put precisely the failing task traces in an Insight.
        trace_ids[trial.task_id] = trace_id

    return trace_ids


async def _wait_retrievable(client: AsyncNemoClient, workspace: str, trace_ids: set[str]) -> None:
    """Block until every trace id resolves, or raise once the budget is spent."""
    pending = set(trace_ids)
    intake = AsyncIntakeClient.from_client(client)
    for _ in range(POLL_ATTEMPTS):
        for trace_id in sorted(pending):
            try:
                await intake.get_trace(id=trace_id, workspace=workspace)
            except NotFoundError:
                continue
            pending.discard(trace_id)
        if not pending:
            return
        await asyncio.sleep(POLL_DELAY_SECONDS)
    raise TimeoutError(f"traces never became retrievable: {sorted(pending)}")


async def run(args: argparse.Namespace) -> dict[str, str]:
    """Evaluate the group's train split, then upload every trial's trace."""
    agent_path = args.agent.expanduser().resolve()
    dataset_path = (args.dataset_root / "groups" / args.group / args.split).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"group dataset not found: {dataset_path}")

    dataset = HarborDataset.from_path(dataset_path)
    run_dir = args.output.expanduser().resolve() / args.group / args.split
    if run_dir.exists():
        raise FileExistsError(f"output directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    client = make_client(args.base_url)
    try:
        (
            await AsyncWorkspacesClient.from_client(client).create_workspace(
                exist_ok=True,
                body=CreateWorkspaceRequest(name=args.workspace, description="smoke-agent traces for Insights"),
            )
        ).data()
        await _ensure_evaluation(client, workspace=args.workspace, name=args.group)
        options = HarborEvaluatorConfig(
            job_name=f"smoke-{args.group}-{args.split}-record",
            jobs_dir=Path("results"),
            n_attempts=1,
            n_concurrent_trials=args.concurrency,
            quiet=True,
        )
        result = await HarborNativeOutcomeEvaluator(experiment_dir=run_dir).run(
            agent=agent_path,
            dataset=dataset,
            options=options,
        )
        trials = list(result.trials)
        if not trials:
            raise RuntimeError(f"no trials produced under {run_dir}")

        trace_ids = await _upload_trials(client, trials, workspace=args.workspace, group=args.group)
        await _wait_retrievable(client, args.workspace, set(trace_ids.values()))
        return trace_ids
    finally:
        await client.close()


def main() -> None:
    """Record and ingest one group's traces."""
    example_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--workspace", default="smoke-agent")
    parser.add_argument("--agent", type=Path, default=example_dir)
    parser.add_argument("--dataset-root", type=Path, default=example_dir / "dataset")
    parser.add_argument("--output", type=Path, default=Path("tmp/smoke-record"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    for task_id, trace_id in asyncio.run(run(args)).items():
        print(f"{task_id} {trace_id}")


if __name__ == "__main__":
    main()
