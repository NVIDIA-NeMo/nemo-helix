#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed local Intake with valid evaluation telemetry and verify API rollups.

A small, parameterized smoke test: seeds one evaluation (attached to one group)
and ingests sessions via ATIF, then polls the evaluation read endpoint until the
ClickHouse-hydrated rollup converges. Useful for verifying the rollup pipeline
at varying session counts.

For a curated multi-group dataset for the Studio UI team, see
``seed_experiments_demo.py`` in the same directory.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_WORKSPACE = "default"
DEFAULT_EVALUATION = "rollup-smoke-exp"
DEFAULT_GROUP = "rollup-smoke-group"
DATASET_NAME = "rollup-smoke-dataset"
AGENT_NAME = "sample-agent"
AGENT_VERSION = "1.0.0"

SAMPLE_ROWS = [
    ("run-1", "case-1a", 0.4, 0.05, 500),
    ("run-1", "case-1b", 0.8, 0.10, 1500),
    ("run-2", "case-2", 0.8, 0.20, 2000),
    ("run-3", "case-3", 1.0, 0.30, 3000),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--evaluation", default=DEFAULT_EVALUATION)
    parser.add_argument(
        "--runs", type=int, help="Generate this many synthetic evaluation runs instead of the smoke set."
    )
    parser.add_argument("--cases-per-run", type=int, default=1, help="Synthetic test cases per run when --runs is set.")
    args = parser.parse_args()
    if args.runs is not None and args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.cases_per_run < 1:
        raise SystemExit("--cases-per-run must be >= 1")

    sample_rows = _sample_rows(runs=args.runs, cases_per_run=args.cases_per_run)

    base_url = args.base_url.rstrip("/")
    _preflight(base_url)

    with httpx.Client(timeout=10.0) as client:
        group_id = _upsert_group(client, base_url, args.workspace)
        _upsert_evaluation(client, base_url, args.workspace, args.evaluation, group_id=group_id)
        started_at = datetime.now(timezone.utc).replace(microsecond=0)
        for index, (run_id, test_case_id, score, cost_usd, latency_ms) in enumerate(sample_rows):
            response = client.post(
                _intake_url(base_url, args.workspace, "/ingest/atif"),
                json=_atif_body(
                    started_at=started_at,
                    evaluation_id=args.evaluation,
                    run_id=run_id,
                    test_case_id=test_case_id,
                    score=score,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                    offset_seconds=index * 10,
                ),
            )
            response.raise_for_status()
            if (index + 1) % 100 == 0:
                print(f"seeded {index + 1} sessions")

        expected_session_count = len(sample_rows)
        evaluation = _wait_for_rollup(
            client,
            base_url,
            args.workspace,
            args.evaluation,
            expected_run_count=expected_session_count,
            expected_session_count=expected_session_count,
        )
        print(json.dumps(evaluation, indent=2, sort_keys=True))


def _preflight(base_url: str) -> None:
    try:
        response = httpx.get(_replace_path(base_url, "/openapi.json"), timeout=2.0)
        response.raise_for_status()
    except Exception as exc:
        raise SystemExit(f"Cannot reach NeMo Helix at {base_url}: {exc}") from exc


def _upsert_group(client: httpx.Client, base_url: str, workspace: str) -> str:
    body = {"name": DEFAULT_GROUP, "description": "Smoke-test group for the rollup script."}
    url = _intake_url(base_url, workspace, "/experiments")
    response = client.post(url, json=body)
    if response.status_code == 409:
        response = client.put(_intake_url(base_url, workspace, f"/experiments/{DEFAULT_GROUP}"), json=body)
    response.raise_for_status()
    return response.json()["id"]


def _upsert_evaluation(
    client: httpx.Client,
    base_url: str,
    workspace: str,
    evaluation: str,
    *,
    group_id: str,
) -> None:
    body = {
        "name": evaluation,
        "dataset_name": DATASET_NAME,
        "dataset_version": "v1",
        "experiment_group_id": group_id,
        "metadata": {"seeded_by": "services/intake/scripts/spans/seed_experiment_rollup_data.py"},
    }
    response = client.post(_intake_url(base_url, workspace, "/evaluations"), json=body)
    if response.status_code == 409:
        response = client.put(_intake_url(base_url, workspace, f"/evaluations/{evaluation}"), json=body)
    response.raise_for_status()


def _wait_for_rollup(
    client: httpx.Client,
    base_url: str,
    workspace: str,
    evaluation: str,
    *,
    expected_run_count: int,
    expected_session_count: int,
) -> dict[str, Any]:
    url = _intake_url(base_url, workspace, f"/evaluations/{evaluation}")
    last_response: httpx.Response | None = None
    for _ in range(20):
        response = client.get(url)
        last_response = response
        response.raise_for_status()
        payload = response.json()
        score = (payload.get("aggregate_scores") or {}).get("harbor.verifier") or {}
        if payload.get("run_count") == expected_run_count and score.get("count") == expected_session_count:
            return payload
        time.sleep(0.25)
    detail = last_response.text if last_response is not None else "<no response>"
    raise SystemExit(f"Evaluation rollup did not become visible at {url}: {detail}")


def _atif_body(
    *,
    started_at: datetime,
    evaluation_id: str,
    run_id: str,
    test_case_id: str,
    score: float,
    cost_usd: float,
    latency_ms: int,
    offset_seconds: int,
) -> dict[str, Any]:
    session_started_at = started_at + timedelta(seconds=offset_seconds)
    finished_at = session_started_at + timedelta(milliseconds=latency_ms)
    session_id = f"{evaluation_id}-{run_id}-{test_case_id}"
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": session_id,
        "evaluation_context": {
            "evaluation_name": evaluation_id,
            "test_case_name": test_case_id,
        },
        "extra": {
            "task_id": test_case_id,
            "task_name": test_case_id,
            "verifier": {
                "started_at": _iso(session_started_at),
                "finished_at": _iso(finished_at),
            },
            "verifier_result": {"rewards": {"reward": score}},
        },
        "agent": {
            "name": AGENT_NAME,
            "version": AGENT_VERSION,
            "model_name": "provider/sample-model",
        },
        "steps": [
            {
                "step_id": 1,
                "timestamp": _iso(session_started_at),
                "source": "agent",
                "model_name": "provider/sample-model",
                "message": f"solved {test_case_id}",
                "metrics": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "cost_usd": cost_usd,
                },
            }
        ],
    }


def _sample_rows(runs: int | None, cases_per_run: int) -> list[tuple[str, str, float, float, int]]:
    if runs is None:
        return SAMPLE_ROWS
    return [
        (
            f"run-{run_index}",
            f"case-{run_index}-{case_index}",
            _synthetic_score(run_index=run_index, case_index=case_index),
            _synthetic_cost(case_index=case_index),
            _synthetic_latency_ms(run_index=run_index, case_index=case_index),
        )
        for run_index in range(1, runs + 1)
        for case_index in range(1, cases_per_run + 1)
    ]


def _synthetic_score(*, run_index: int, case_index: int) -> float:
    return round(0.4 + (((run_index * 17 + case_index * 11) % 60) / 100), 3)


def _synthetic_cost(*, case_index: int) -> float:
    return round(0.001 * case_index, 6)


def _synthetic_latency_ms(*, run_index: int, case_index: int) -> int:
    return 250 + run_index + case_index * 10


def _intake_url(base_url: str, workspace: str, suffix: str) -> str:
    return f"{base_url}/apis/intake/v2/workspaces/{workspace}{suffix}"


def _replace_path(base_url: str, path: str) -> str:
    parts = urlsplit(base_url)
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
