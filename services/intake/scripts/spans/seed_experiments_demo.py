#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed local Intake with a curated multi-group evaluation dataset.

Designed for the Studio UI team to have realistic data while building out the
Evaluations surfaces. Seeds five experiment groups across multiple agents and
datasets, with varied evaluator scores, costs, and latencies — including a
``secondary-sort-fixtures`` group whose sessions share tied cost values so the
tables' secondary (tie-break) sort can be exercised. Every evaluation is
attached to a group; nothing is ungrouped.

Behavior:

* Re-running is safe. Groups that already exist by name are left alone — none
  of their evaluations, sessions, or rollups are touched.
* ``--wipe-and-seed`` deletes **every** experiment group and evaluation in the
  workspace (including ones this script didn't create) and then seeds from
  scratch. Note: this only removes entity-store rows. ClickHouse session data
  (spans, evaluator_results) is not deletable via the public API, so prior
  session telemetry tagged with a re-used ``evaluation.id`` will still feed
  the rollup after re-seeding.

Usage::

    uv run services/intake/scripts/spans/seed_experiments_demo.py \\
        --base-url http://127.0.0.1:8000

    # Wipe ALL experiment groups + evaluations in the workspace, then re-seed:
    uv run services/intake/scripts/spans/seed_experiments_demo.py \\
        --base-url http://127.0.0.1:8000 --wipe-and-seed

For the parameterized smoke test that exercises one evaluation + many sessions,
see ``seed_experiment_rollup_data.py`` in the same directory.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_WORKSPACE = "default"


# ---------------------------------------------------------------------------
# Demo dataset definitions
# ---------------------------------------------------------------------------


@dataclass
class EvaluationSpec:
    name: str
    description: str
    agent_name: str = "sample-agent"
    agent_version: str = "1.0.0"
    # Optional: sessions cycle through these instead of the scalar above, so a single
    # evaluation's rollup can surface multiple distinct agent names or versions (e.g.,
    # an A/B between agent versions, or a comparison of agents within one evaluation).
    agent_name_cycle: tuple[str, ...] | None = None
    agent_version_cycle: tuple[str, ...] | None = None
    model_name: str = "provider/sample-model"
    dataset_name: str = "sample-dataset"
    dataset_version: str = "v1"
    source_link: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    n_sessions: int = 50
    # Evaluator name -> mean score in [0, 1]. Per-session scores are drawn from a normal
    # distribution around this mean, clipped to [0, 1].
    evaluators: dict[str, float] = field(default_factory=dict)
    score_stddev: float = 0.08
    cost_mean_usd: float = 0.02
    cost_stddev_pct: float = 0.4
    # Optional: sessions cycle through these exact per-session costs (USD) instead of the
    # random draw above, guaranteeing many exact ties in cost_total_usd. Handy for exercising
    # the sessions table's secondary (tie-break) sort — sort by Cost, then watch a second column
    # order the tied rows. Two evaluations that share the same cycle and n_sessions also end up
    # with tied *aggregate* cost, covering the evaluations table's secondary sort at the group level.
    cost_usd_cycle: tuple[float, ...] | None = None
    latency_mean_ms: int = 1500
    latency_stddev_ms: int = 400
    prompt_tokens_mean: int = 600
    completion_tokens_mean: int = 200


@dataclass
class GroupSpec:
    name: str
    description: str
    evaluations: list[EvaluationSpec]


DEMO_GROUPS: list[GroupSpec] = [
    # ----- Group 1: progressive prompt + reranker iteration on one dataset -----
    GroupSpec(
        name="reranker-prompt-iteration",
        description="Iterating on the Support-Bench RAG agent's reranker and system prompt.",
        evaluations=[
            EvaluationSpec(
                name="reranker-main-baseline",
                description="Pre-reranker baseline on the production prompt.",
                agent_name="codex-cli",
                agent_version="1.2.3",
                model_name="openai/gpt-4o-mini",
                dataset_name="support-bench",
                dataset_version="v3",
                n_sessions=60,
                evaluators={"solved": 0.78, "helpful": 0.81, "groundedness": 0.85},
                cost_mean_usd=0.045,
                latency_mean_ms=2800,
            ),
            EvaluationSpec(
                name="reranker-add-cross-encoder",
                description="Add a cross-encoder reranker to the retrieval step.",
                agent_name="codex-cli",
                agent_version="1.2.3",
                model_name="openai/gpt-4o-mini",
                dataset_name="support-bench",
                dataset_version="v3",
                n_sessions=55,
                evaluators={"solved": 0.82, "helpful": 0.82, "groundedness": 0.88},
                cost_mean_usd=0.052,
                latency_mean_ms=3100,
            ),
            EvaluationSpec(
                name="reranker-tightened-prompt",
                description="Cross-encoder reranker + tightened system prompt.",
                agent_name="codex-cli",
                agent_version="1.2.3",
                model_name="openai/gpt-4o-mini",
                dataset_name="support-bench",
                dataset_version="v3",
                n_sessions=70,
                evaluators={"solved": 0.86, "helpful": 0.86, "groundedness": 0.92},
                cost_mean_usd=0.053,
                latency_mean_ms=3000,
            ),
            EvaluationSpec(
                name="reranker-top-k-8",
                description="Ablation: top_k = 8 instead of the default 4.",
                agent_name="codex-cli",
                agent_version="1.2.3",
                model_name="openai/gpt-4o-mini",
                dataset_name="support-bench",
                dataset_version="v3",
                n_sessions=50,
                evaluators={"solved": 0.82, "helpful": 0.84, "groundedness": 0.88},
                cost_mean_usd=0.075,
                latency_mean_ms=4100,
            ),
            EvaluationSpec(
                name="reranker-no-reranker",
                description="Ablation: prompt change without the reranker.",
                agent_name="codex-cli",
                agent_version="1.2.3",
                model_name="openai/gpt-4o-mini",
                dataset_name="support-bench",
                dataset_version="v3",
                n_sessions=45,
                evaluators={"solved": 0.74, "helpful": 0.80, "groundedness": 0.78},
                cost_mean_usd=0.038,
                latency_mean_ms=2400,
            ),
            EvaluationSpec(
                name="reranker-5x-averaged",
                description="Cross-encoder reranker, 5 trials per case, averaged for variance.",
                agent_name="codex-cli",
                agent_version="1.2.3",
                # Trials ran across a minor version bump mid-evaluation.
                agent_version_cycle=("1.2.3", "1.2.4"),
                model_name="openai/gpt-4o-mini",
                dataset_name="support-bench",
                dataset_version="v3",
                n_sessions=80,
                evaluators={"solved": 0.85, "helpful": 0.85, "groundedness": 0.91},
                score_stddev=0.04,
                cost_mean_usd=0.265,
                latency_mean_ms=15000,
            ),
        ],
    ),
    # ----- Group 2: multi-agent comparison on one dataset -----
    GroupSpec(
        name="coding-agent-showdown",
        description="Comparing three coding agents on terminal-bench-2.",
        evaluations=[
            EvaluationSpec(
                name="tb2-claude-code-opus",
                description="claude-code @ 0.125 with opus.",
                agent_name="claude-code",
                agent_version="0.125.0",
                model_name="anthropic/claude-opus-4-7",
                dataset_name="terminal-bench-2",
                dataset_version="v1",
                n_sessions=50,
                evaluators={"pass_at_1": 0.86, "trajectory_quality": 0.82, "safety": 0.97},
                cost_mean_usd=0.42,
                latency_mean_ms=8200,
            ),
            EvaluationSpec(
                name="tb2-codex-cli",
                description="codex-cli @ 1.4 with gpt-4o.",
                agent_name="codex-cli",
                agent_version="1.4.0",
                model_name="openai/gpt-4o",
                dataset_name="terminal-bench-2",
                dataset_version="v1",
                n_sessions=50,
                evaluators={"pass_at_1": 0.80, "trajectory_quality": 0.78, "safety": 0.95},
                cost_mean_usd=0.22,
                latency_mean_ms=5400,
            ),
            EvaluationSpec(
                name="tb2-cursor-agent",
                description="cursor-agent + cursor-cli mix @ 0.4 with claude-sonnet.",
                agent_name="cursor-agent",
                agent_version="0.4.1",
                # Mix of cursor's agent and CLI binary running the same task.
                agent_name_cycle=("cursor-agent", "cursor-cli"),
                model_name="anthropic/claude-sonnet-4-6",
                dataset_name="terminal-bench-2",
                dataset_version="v1",
                n_sessions=50,
                evaluators={"pass_at_1": 0.77, "trajectory_quality": 0.81, "safety": 0.96},
                cost_mean_usd=0.18,
                latency_mean_ms=4700,
            ),
        ],
    ),
    # ----- Group 3: cross-cutting (multi-agent × multi-dataset) -----
    GroupSpec(
        name="cross-agent-cross-dataset-sweep",
        description="Two agents × two datasets to see where each agent shines.",
        evaluations=[
            EvaluationSpec(
                name="claude-code-on-agentic-bench",
                description="claude-code on agentic-bench (long-horizon).",
                agent_name="claude-code",
                agent_version="0.125.0",
                model_name="anthropic/claude-opus-4-7",
                dataset_name="agentic-bench",
                dataset_version="2026-05",
                n_sessions=40,
                evaluators={"pass_at_1": 0.74, "goal_accuracy": 0.78, "trajectory_quality": 0.84},
                cost_mean_usd=0.55,
                latency_mean_ms=12000,
            ),
            EvaluationSpec(
                name="claude-code-on-support",
                description="claude-code on customer-support v2.",
                agent_name="claude-code",
                agent_version="0.125.0",
                model_name="anthropic/claude-opus-4-7",
                dataset_name="customer-support",
                dataset_version="v2",
                n_sessions=40,
                evaluators={"pass_at_1": 0.88, "goal_accuracy": 0.91, "trajectory_quality": 0.86},
                cost_mean_usd=0.12,
                latency_mean_ms=3100,
            ),
            EvaluationSpec(
                name="codex-cli-on-agentic-bench",
                description="codex-cli on agentic-bench (long-horizon).",
                agent_name="codex-cli",
                agent_version="1.4.0",
                model_name="openai/gpt-4o",
                dataset_name="agentic-bench",
                dataset_version="2026-05",
                n_sessions=40,
                evaluators={"pass_at_1": 0.69, "goal_accuracy": 0.72, "trajectory_quality": 0.75},
                cost_mean_usd=0.31,
                latency_mean_ms=8500,
            ),
            EvaluationSpec(
                name="codex-cli-on-support",
                description="codex-cli on customer-support v2.",
                agent_name="codex-cli",
                agent_version="1.4.0",
                model_name="openai/gpt-4o",
                dataset_name="customer-support",
                dataset_version="v2",
                n_sessions=40,
                evaluators={"pass_at_1": 0.84, "goal_accuracy": 0.86, "trajectory_quality": 0.81},
                cost_mean_usd=0.09,
                latency_mean_ms=2600,
            ),
        ],
    ),
    # ----- Group 4: model size sweep (one agent, one dataset, three models) -----
    GroupSpec(
        name="claude-model-size-sweep",
        description="Same agent + dataset, three claude model sizes.",
        evaluations=[
            EvaluationSpec(
                name="tau-claude-haiku",
                description="claude-code with claude-haiku-4-5.",
                agent_name="claude-code",
                agent_version="0.125.0",
                model_name="anthropic/claude-haiku-4-5",
                dataset_name="tau-bench",
                dataset_version="v0.4",
                n_sessions=80,
                evaluators={"pass_at_1": 0.68, "goal_accuracy": 0.70},
                cost_mean_usd=0.02,
                latency_mean_ms=1400,
            ),
            EvaluationSpec(
                name="tau-claude-sonnet",
                description="claude-code with claude-sonnet-4-6.",
                agent_name="claude-code",
                agent_version="0.125.0",
                model_name="anthropic/claude-sonnet-4-6",
                dataset_name="tau-bench",
                dataset_version="v0.4",
                n_sessions=60,
                evaluators={"pass_at_1": 0.78, "goal_accuracy": 0.81},
                cost_mean_usd=0.11,
                latency_mean_ms=3200,
            ),
            EvaluationSpec(
                name="tau-claude-opus",
                description="claude-code with claude-opus-4-7.",
                agent_name="claude-code",
                agent_version="0.125.0",
                model_name="anthropic/claude-opus-4-7",
                dataset_name="tau-bench",
                dataset_version="v0.4",
                n_sessions=40,
                evaluators={"pass_at_1": 0.86, "goal_accuracy": 0.88},
                cost_mean_usd=0.48,
                latency_mean_ms=7800,
            ),
        ],
    ),
    # ----- Group 5: fixtures for the tables' secondary (tie-break) sort -----
    # Both evaluations draw per-session cost from the SAME fixed cycle, so:
    #   * within an evaluation, many sessions share an identical cost_total_usd — sort the
    #     sessions table by Cost, then a second column orders the tied rows; and
    #   * because they also share n_sessions, their *aggregate* costs tie, exercising the
    #     evaluations table's secondary sort at the group level too.
    # Latency and evaluator scores still vary, so there's a meaningful second sort key.
    GroupSpec(
        name="secondary-sort-fixtures",
        description="Tied cost values for exercising the tables' secondary (tie-break) sort.",
        evaluations=[
            EvaluationSpec(
                name="tied-cost-fast-agent",
                description="Per-session cost snaps to four fixed buckets; low latency.",
                agent_name="claude-code",
                agent_version="0.125.0",
                model_name="anthropic/claude-haiku-4-5",
                dataset_name="tau-bench",
                dataset_version="v0.4",
                n_sessions=40,
                evaluators={"pass_at_1": 0.72, "goal_accuracy": 0.75},
                cost_usd_cycle=(0.05, 0.10, 0.15, 0.20),
                latency_mean_ms=1400,
                latency_stddev_ms=250,
            ),
            EvaluationSpec(
                name="tied-cost-slow-agent",
                description="Same four cost buckets as the fast agent (tied aggregate cost); higher latency.",
                agent_name="codex-cli",
                agent_version="1.4.0",
                model_name="openai/gpt-4o",
                dataset_name="tau-bench",
                dataset_version="v0.4",
                n_sessions=40,
                evaluators={"pass_at_1": 0.66, "goal_accuracy": 0.70},
                cost_usd_cycle=(0.05, 0.10, 0.15, 0.20),
                latency_mean_ms=5200,
                latency_stddev_ms=700,
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument(
        "--wipe-and-seed",
        action="store_true",
        help=(
            "DELETE every experiment group + evaluation in the workspace (including ones this "
            "script didn't create), then re-seed. Destructive — use carefully."
        ),
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    _preflight(base_url)

    with httpx.Client(timeout=10.0) as client:
        if args.wipe_and_seed:
            _wipe_workspace(client, base_url, args.workspace)
        seed(client, base_url, args.workspace)


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def seed(client: httpx.Client, base_url: str, workspace: str) -> None:
    """Seed the curated multi-group dataset. Skips groups that already exist."""
    print("=== Seeding demo data ===")
    # Anchor sessions ~6h in the past so timestamps look like "recent activity."
    base_started_at = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(hours=6)

    groups_created = 0
    groups_skipped = 0
    evaluations_created = 0
    sessions_seeded = 0

    for group_spec in DEMO_GROUPS:
        group_id, created = _create_group_if_missing(client, base_url, workspace, group_spec)
        if not created or group_id is None:
            print(f"\n[skip] group '{group_spec.name}' already exists; leaving it and its evaluations alone")
            groups_skipped += 1
            continue
        groups_created += 1
        print(f"\n[group] {group_spec.name}  ({len(group_spec.evaluations)} evaluations)")
        for exp_spec in group_spec.evaluations:
            print(f"  [evaluation] {exp_spec.name}  n_sessions={exp_spec.n_sessions}")
            _create_evaluation(client, base_url, workspace, exp_spec, group_id=group_id)
            _seed_sessions(client, base_url, workspace, exp_spec, base_started_at)
            evaluations_created += 1
            sessions_seeded += exp_spec.n_sessions

    print(
        f"\n=== Done. groups: {groups_created} created, {groups_skipped} skipped. "
        f"evaluations: {evaluations_created} created. sessions: {sessions_seeded} ingested. ==="
    )


def _create_group_if_missing(
    client: httpx.Client, base_url: str, workspace: str, spec: GroupSpec
) -> tuple[str | None, bool]:
    """Returns (group_id, created). On 409, returns (None, False) and leaves the existing group alone."""
    url = _intake_url(base_url, workspace, "/experiments")
    body = {"name": spec.name, "description": spec.description}
    response = client.post(url, json=body)
    if response.status_code == 409:
        return None, False
    response.raise_for_status()
    return response.json()["id"], True


def _create_evaluation(
    client: httpx.Client,
    base_url: str,
    workspace: str,
    spec: EvaluationSpec,
    *,
    group_id: str,
) -> None:
    """POST an evaluation. Errors on conflict — callers must guarantee the evaluation doesn't exist."""
    response = client.post(_intake_url(base_url, workspace, "/evaluations"), json=_evaluation_body(spec, group_id))
    response.raise_for_status()


def _evaluation_body(spec: EvaluationSpec, group_id: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": spec.name,
        "dataset_name": spec.dataset_name,
        "dataset_version": spec.dataset_version,
        "experiment_group_id": group_id,
        "description": spec.description,
        "metadata": {
            "seeded_by": "services/intake/scripts/spans/seed_experiments_demo.py",
            "model_name": spec.model_name,
            **spec.metadata,
        },
    }
    if spec.source_link:
        body["source_link"] = spec.source_link
    return body


def _seed_sessions(
    client: httpx.Client,
    base_url: str,
    workspace: str,
    spec: EvaluationSpec,
    base_started_at: datetime,
) -> None:
    """Ingest N sessions via ATIF + per-evaluator POST /evaluator-results."""
    # Deterministic per-evaluation so re-runs produce the same values.
    rng = random.Random(f"seed:{spec.name}")

    atif_url = _intake_url(base_url, workspace, "/ingest/atif")
    eval_url = _intake_url(base_url, workspace, "/evaluator-results")

    for i in range(spec.n_sessions):
        cost_usd = (
            spec.cost_usd_cycle[i % len(spec.cost_usd_cycle)]
            if spec.cost_usd_cycle
            else max(0.0005, rng.gauss(spec.cost_mean_usd, spec.cost_mean_usd * spec.cost_stddev_pct))
        )
        latency_ms = max(100, int(rng.gauss(spec.latency_mean_ms, spec.latency_stddev_ms)))
        prompt_tokens = max(10, int(rng.gauss(spec.prompt_tokens_mean, spec.prompt_tokens_mean * 0.25)))
        completion_tokens = max(5, int(rng.gauss(spec.completion_tokens_mean, spec.completion_tokens_mean * 0.3)))

        test_case_name = f"case-{i:04d}"
        run_id = f"run-{i // 25:02d}"
        # Spread sessions across the ~5.5h prior to "now" so the Studio timeline looks varied.
        offset_seconds = (i / max(1, spec.n_sessions)) * 5.5 * 3600

        session_agent_name = (
            spec.agent_name_cycle[i % len(spec.agent_name_cycle)] if spec.agent_name_cycle else spec.agent_name
        )
        session_agent_version = (
            spec.agent_version_cycle[i % len(spec.agent_version_cycle)]
            if spec.agent_version_cycle
            else spec.agent_version
        )
        atif_body = _demo_atif_body(
            base_started_at=base_started_at,
            evaluation_name=spec.name,
            run_id=run_id,
            test_case_name=test_case_name,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            offset_seconds=offset_seconds,
            agent_name=session_agent_name,
            agent_version=session_agent_version,
            model_name=spec.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        response = client.post(atif_url, json=atif_body)
        response.raise_for_status()

        session_id = atif_body["session_id"]
        # Loose-target span_id: evaluator_results joins by session_id; span_id isn't validated.
        synthetic_span_id = f"{session_id}-root"

        for eval_name, mean in spec.evaluators.items():
            score = _clip(rng.gauss(mean, spec.score_stddev))
            eval_response = client.post(
                eval_url,
                json={
                    "span_id": synthetic_span_id,
                    "session_id": session_id,
                    "name": eval_name,
                    "value": score,
                    "data_type": "NUMERIC",
                },
            )
            eval_response.raise_for_status()

        if (i + 1) % 25 == 0 or i + 1 == spec.n_sessions:
            print(f"      {i + 1}/{spec.n_sessions} sessions")


def _demo_atif_body(
    *,
    base_started_at: datetime,
    evaluation_name: str,
    run_id: str,
    test_case_name: str,
    cost_usd: float,
    latency_ms: int,
    offset_seconds: float,
    agent_name: str,
    agent_version: str,
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    session_started_at = base_started_at + timedelta(seconds=offset_seconds)
    finished_at = session_started_at + timedelta(milliseconds=latency_ms)
    session_id = f"{evaluation_name}-{run_id}-{test_case_name}"
    # `extra.verifier` carries the timing block (used by the rollup for session latency).
    # We omit `extra.verifier_result` so ATIF ingest doesn't auto-create a `harbor.verifier`
    # evaluator alongside our cleanly-named ones from POST /evaluator-results.
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": session_id,
        "evaluation_context": {
            "evaluation_name": evaluation_name,
            "test_case_name": test_case_name,
        },
        "extra": {
            "task_id": test_case_name,
            "task_name": test_case_name,
            "verifier": {
                "started_at": _iso(session_started_at),
                "finished_at": _iso(finished_at),
            },
        },
        "agent": {
            "name": agent_name,
            "version": agent_version,
            "model_name": model_name,
        },
        "steps": [
            {
                "step_id": 1,
                "timestamp": _iso(session_started_at),
                "source": "user",
                "message": f"test case: {test_case_name}",
            },
            {
                "step_id": 2,
                "timestamp": _iso(finished_at),
                "source": "agent",
                "model_name": model_name,
                "message": f"solved {test_case_name}",
                "metrics": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost_usd": cost_usd,
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# Wipe (--wipe-and-seed)
# ---------------------------------------------------------------------------


def _wipe_workspace(client: httpx.Client, base_url: str, workspace: str) -> None:
    """Delete every evaluation + experiment group in the workspace.

    Iterates the existing list endpoints with pagination and DELETEs each row.
    Evaluations are deleted before groups so the group-level UI doesn't briefly
    show empty groups. Only entity-store rows are removed; ClickHouse session
    data is untouched (no public API to delete it).
    """
    print(f"=== --wipe-and-seed: deleting every evaluation + group in workspace '{workspace}' ===")
    deleted_evaluations = 0
    for name in _list_all_names(client, base_url, workspace, "/evaluations"):
        if _delete(client, base_url, workspace, f"/evaluations/{name}"):
            deleted_evaluations += 1
    deleted_groups = 0
    for name in _list_all_names(client, base_url, workspace, "/experiments"):
        if _delete(client, base_url, workspace, f"/experiments/{name}"):
            deleted_groups += 1
    print(f"deleted {deleted_evaluations} evaluation(s) and {deleted_groups} group(s)\n")


def _list_all_names(client: httpx.Client, base_url: str, workspace: str, suffix: str) -> list[str]:
    """Paginate the given list endpoint and return every row's ``name``."""
    names: list[str] = []
    page = 1
    page_size = 1000  # max allowed
    while True:
        response = client.get(
            _intake_url(base_url, workspace, suffix),
            params={"page": page, "page_size": page_size},
        )
        response.raise_for_status()
        body = response.json()
        rows = body.get("data") or []
        names.extend(row["name"] for row in rows if "name" in row)
        pagination = body.get("pagination") or {}
        total_pages = pagination.get("total_pages") or 1
        if page >= total_pages or not rows:
            break
        page += 1
    return names


def _delete(client: httpx.Client, base_url: str, workspace: str, suffix: str) -> bool:
    response = client.delete(_intake_url(base_url, workspace, suffix))
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return True


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _preflight(base_url: str) -> None:
    try:
        response = httpx.get(_replace_path(base_url, "/openapi.json"), timeout=2.0)
        response.raise_for_status()
    except Exception as exc:
        raise SystemExit(f"Cannot reach NeMo Helix at {base_url}: {exc}") from exc


def _intake_url(base_url: str, workspace: str, suffix: str) -> str:
    return f"{base_url}/apis/intake/v2/workspaces/{workspace}{suffix}"


def _replace_path(base_url: str, path: str) -> str:
    parts = urlsplit(base_url)
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
