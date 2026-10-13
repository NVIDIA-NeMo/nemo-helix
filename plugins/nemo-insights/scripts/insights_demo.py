#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed or clean a local NeMo Insights Studio demo through public HTTP APIs.

From the repository root::

    uv sync --group insights
    services/intake/scripts/spans/run_clickhouse.sh
    cd web && VITE_FF_OPTIMIZER_ENABLED=preview pnpm --filter nemo-studio-ui build:fastapi && cd ..
    NHX_INTAKE_CLICKHOUSE_URL=http://localhost:8123 uv run nemo services run \
      --service-group all --config packages/nhx_platform/config/local.yaml
    uv run plugins/nemo-insights/scripts/insights_demo.py seed

Open ``http://localhost:8080/studio/workspaces/insights-demo/optimizer``.
Clean with ``uv run plugins/nemo-insights/scripts/insights_demo.py clean``.
This Insights UI is separate from the existing Agents ``Suggestions`` feature.
The script deletes only the fixed ``insights-demo`` workspace. Intake has no
public telemetry-delete API, so deterministic session IDs keep reseeding stable
when inaccessible ClickHouse rows remain.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import httpx

DEMO_WORKSPACE = "insights-demo"
DEFAULT_BASE_URL = "http://localhost:8080"
CLICKHOUSE_RECOVERY_COMMAND = (
    "services/intake/scripts/spans/run_clickhouse.sh, then restart the platform with "
    "NHX_INTAKE_CLICKHOUSE_URL=http://localhost:8123"
)
INSIGHTS_INSTALL_COMMAND = "uv sync --group insights"
_BASE_TIME = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
_SOURCE_URL = "https://github.com/NVIDIA-NeMo/nemo-helix/commit"
_SUPPORT_AGENT = ("support-agent", "nvidia/nemotron-mini")
_RETRIEVAL_AGENT = ("retrieval-agent", "nvidia/llama-3.3-nemotron-super")


def _workspace_path(service: str, suffix: str = "") -> str:
    return f"/apis/{service}/v2/workspaces/{DEMO_WORKSPACE}{suffix}"


class DemoError(RuntimeError):
    """A concise, actionable demo failure."""


@dataclass(frozen=True)
class SessionSpec:
    session_id: str
    test_case_name: str
    started_at: datetime
    latency_ms: int
    cost_usd: float
    quality: float
    correctness: float


@dataclass(frozen=True)
class EvaluationSpec:
    name: str
    source_link: str
    agent: tuple[str, str]
    sessions: tuple[SessionSpec, ...]


@dataclass(frozen=True)
class GroupSpec:
    name: str
    description: str
    summary: str
    default_sort: str
    insight_key: str
    evaluations: tuple[EvaluationSpec, ...]


@dataclass(frozen=True)
class InsightSpec:
    key: str
    title: str
    description: str
    status: Literal["open", "resolved", "deleted"]
    trace_refs: tuple[str, ...]


@dataclass(frozen=True)
class DemoFixture:
    insights: tuple[InsightSpec, ...]
    groups: tuple[GroupSpec, ...]


def _evaluation(
    name: str,
    commit: str,
    count: int,
    *,
    start_index: int,
    latency_ms: int,
    cost_usd: float,
    quality: float,
    correctness: float,
    agent: tuple[str, str],
) -> EvaluationSpec:
    return EvaluationSpec(
        name=name,
        source_link=f"{_SOURCE_URL}/{commit}",
        agent=agent,
        sessions=tuple(
            SessionSpec(
                session_id=f"insights-demo-{name}-{index + 1:02d}",
                test_case_name=f"case-{index + 1:02d}",
                started_at=_BASE_TIME + timedelta(minutes=7 * (start_index + index)),
                latency_ms=latency_ms + index * 125,
                cost_usd=round(cost_usd + index * 0.002, 3),
                quality=round(min(1.0, quality + index * 0.02), 2),
                correctness=round(min(1.0, correctness + index * 0.01), 2),
            )
            for index in range(count)
        ),
    )


def build_fixture() -> DemoFixture:
    """Return the deterministic, compact demo fixture."""
    prompt_baseline = _evaluation(
        "prompt-baseline",
        "1111111",
        3,
        start_index=0,
        latency_ms=1850,
        cost_usd=0.032,
        quality=0.72,
        correctness=0.79,
        agent=_SUPPORT_AGENT,
    )
    prompt_compact = _evaluation(
        "prompt-compact-context",
        "2222222",
        3,
        start_index=3,
        latency_ms=1120,
        cost_usd=0.021,
        quality=0.84,
        correctness=0.88,
        agent=_SUPPORT_AGENT,
    )
    router_baseline = _evaluation(
        "router-baseline",
        "3333333",
        3,
        start_index=6,
        latency_ms=2300,
        cost_usd=0.041,
        quality=0.68,
        correctness=0.76,
        agent=_RETRIEVAL_AGENT,
    )
    router_streaming = _evaluation(
        "router-streaming-cache",
        "4444444",
        2,
        start_index=9,
        latency_ms=940,
        cost_usd=0.027,
        quality=0.89,
        correctness=0.91,
        agent=_RETRIEVAL_AGENT,
    )
    groups = (
        GroupSpec(
            name="prompt-response-time",
            description="Prompt experiments addressing slow support responses.",
            summary="Compact context improved quality while reducing average latency and cost.",
            default_sort="-evaluators.quality.mean,cost_usd.mean",
            insight_key="slow-responses",
            evaluations=(prompt_baseline, prompt_compact),
        ),
        GroupSpec(
            name="retrieval-routing",
            description="Routing experiments for retrieval-heavy support questions.",
            summary="Streaming cache-aware retrieval produced the strongest latency result.",
            default_sort="latency_ms.mean,-evaluators.correctness.mean",
            insight_key="slow-responses",
            evaluations=(router_baseline, router_streaming),
        ),
    )
    trace_refs = tuple(
        session.session_id for group in groups for evaluation in group.evaluations for session in evaluation.sessions
    )
    return DemoFixture(
        insights=(
            InsightSpec(
                key="slow-responses",
                title="Support responses are slow on retrieval-heavy requests",
                description=(
                    "Evidence traces show retrieval and oversized prompt context dominate response time. "
                    "Compare prompt compaction and cache-aware routing experiments."
                ),
                status="open",
                trace_refs=trace_refs,
            ),
            InsightSpec(
                key="stable-quality",
                title="Answer quality remains stable after prompt compaction",
                description=("Resolved after compact-context evaluations preserved correctness while lowering cost."),
                status="resolved",
                trace_refs=trace_refs[:2],
            ),
            InsightSpec(
                key="legacy-router",
                title="Legacy routing recommendation is no longer actionable",
                description="Deleted after cache-aware routing superseded the original recommendation.",
                status="deleted",
                trace_refs=trace_refs[-1:],
            ),
        ),
        groups=groups,
    )


class DemoAPI:
    """Small public-HTTP client for the APIs used by the demo."""

    def __init__(self, base_url: str, *, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=15.0)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> DemoAPI:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        expected: tuple[int, ...] = (200,),
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        try:
            response = self.client.request(
                method,
                f"{self.base_url}{path}",
                json=json,
                params=params,
            )
        except httpx.RequestError as error:
            raise DemoError(f"Platform is unavailable at {self.base_url}: {error}") from error
        if response.status_code not in expected:
            detail = response.text.strip()
            suffix = f": {detail}" if detail else ""
            raise DemoError(f"{method} {path} failed ({response.status_code}){suffix}")
        return response

    def preflight(self) -> None:
        self._request("GET", "/health/ready")
        openapi = self._request("GET", "/openapi.json").json()
        paths = openapi.get("paths", {})
        if "/apis/insights/v2/workspaces/{workspace}/insights" not in paths:
            raise DemoError(
                f"Insights service is unavailable. Install the optional plugin with: {INSIGHTS_INSTALL_COMMAND}"
            )
        response = self._request(
            "GET",
            "/apis/intake/v2/workspaces/default/traces",
            expected=(200, 404, 503),
            params={"page": 1, "page_size": 1, "mode": "summary"},
        )
        if response.status_code != 200:
            raise DemoError(f"Intake is unavailable. Start ClickHouse with: {CLICKHOUSE_RECOVERY_COMMAND}")

    def delete_workspace(self, *, sleep: Callable[[float], None]) -> None:
        path = _workspace_path("entities")
        self._request(
            "DELETE",
            path,
            expected=(200, 404),
        )
        for _ in range(20):
            response = self._request("GET", path, expected=(200, 404))
            if response.status_code == 404:
                return
            sleep(0.25)
        raise DemoError(f"Timed out deleting workspace '{DEMO_WORKSPACE}'")

    def create_workspace(self, *, sleep: Callable[[float], None]) -> None:
        for _ in range(40):
            response = self._request(
                "POST",
                "/apis/entities/v2/workspaces",
                expected=(200, 201, 409),
                json={
                    "name": DEMO_WORKSPACE,
                    "description": "Deterministic local NeMo Insights Studio demo.",
                },
            )
            if response.status_code != 409:
                return
            sleep(0.5)
        raise DemoError(f"Timed out recreating workspace '{DEMO_WORKSPACE}'")

    def create_insight(self, insight: InsightSpec) -> str:
        response = self._request(
            "POST",
            _workspace_path("insights", "/insights"),
            expected=(201,),
            json={
                "title": insight.title,
                "description": insight.description,
                "agent": "insights-demo-agent",
                "status": insight.status,
                "trace_refs": [],
            },
        )
        return str(response.json()["id"])

    def update_insight_traces(self, insight_id: str, trace_refs: tuple[str, ...]) -> None:
        self._request(
            "PATCH",
            _workspace_path("insights", f"/insights/{insight_id}"),
            json={"trace_refs": list(trace_refs)},
        )

    def create_group(self, group: GroupSpec, insight_id: str) -> str:
        response = self._request(
            "POST",
            _workspace_path("intake", "/experiment-groups"),
            expected=(201,),
            json={
                "name": group.name,
                "description": group.description,
                "summary": group.summary,
                "insight_id": insight_id,
                "default_sort": group.default_sort,
            },
        )
        return str(response.json()["id"])

    def create_evaluation(self, group_id: str, evaluation: EvaluationSpec) -> None:
        self._request(
            "POST",
            _workspace_path("intake", "/evaluations"),
            expected=(201,),
            json={
                "name": evaluation.name,
                "experiment_group_id": group_id,
                "dataset_name": "insights-demo-cases",
                "source_link": evaluation.source_link,
            },
        )

    def ingest_session(self, evaluation: EvaluationSpec, session: SessionSpec) -> None:
        finished_at = session.started_at + timedelta(milliseconds=session.latency_ms)
        agent_name, model_name = evaluation.agent
        self._request(
            "POST",
            _workspace_path("intake", "/ingest/atif"),
            expected=(201,),
            json={
                "schema_version": "ATIF-v1.7",
                "session_id": session.session_id,
                "evaluation_context": {
                    "evaluation_name": evaluation.name,
                    "test_case_name": session.test_case_name,
                },
                "extra": {
                    "verifier": {
                        "started_at": _iso(session.started_at),
                        "finished_at": _iso(finished_at),
                    },
                    "verifier_result": {
                        "rewards": {
                            "quality": session.quality,
                            "correctness": session.correctness,
                        }
                    },
                },
                "agent": {
                    "name": agent_name,
                    "version": "1.0.0",
                    "model_name": model_name,
                },
                "steps": [
                    {
                        "step_id": 1,
                        "timestamp": _iso(session.started_at),
                        "source": "user",
                        "message": f"Investigate support request {session.test_case_name}.",
                    },
                    {
                        "step_id": 2,
                        "timestamp": _iso(finished_at),
                        "source": "agent",
                        "model_name": model_name,
                        "message": "Retrieved current evidence and returned a grounded answer.",
                        "metrics": {
                            "prompt_tokens": 320,
                            "completion_tokens": 96,
                            "cost_usd": session.cost_usd,
                        },
                    },
                ],
            },
        )

    def evaluation_is_ready(self, evaluation: EvaluationSpec) -> bool:
        response = self._request(
            "GET",
            _workspace_path("intake", f"/evaluations/{evaluation.name}"),
        ).json()
        return (
            response.get("run_count") == len(evaluation.sessions)
            and bool(response.get("aggregate_scores"))
            and response.get("cost_usd") is not None
            and response.get("latency_ms") is not None
        )


def clean(
    api: DemoAPI,
    *,
    sleep: Callable[[float], None] = time.sleep,
    quiet: bool = False,
) -> None:
    """Delete only the dedicated demo workspace."""
    api.delete_workspace(sleep=sleep)
    if not quiet:
        print(f"Deleted workspace '{DEMO_WORKSPACE}'.")
        print("ClickHouse telemetry remains physically stored; public Intake APIs expose no delete operation.")


def seed(
    api: DemoAPI,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Recreate and seed the deterministic demo workspace."""
    api.preflight()
    clean(api, sleep=sleep, quiet=True)
    api.create_workspace(sleep=sleep)

    fixture = build_fixture()
    insight_ids = {insight.key: api.create_insight(insight) for insight in fixture.insights}
    evaluations: list[EvaluationSpec] = []
    for group in fixture.groups:
        group_id = api.create_group(group, insight_ids[group.insight_key])
        for evaluation in group.evaluations:
            api.create_evaluation(group_id, evaluation)
            evaluations.append(evaluation)
            for session in evaluation.sessions:
                api.ingest_session(evaluation, session)

    for insight in fixture.insights:
        api.update_insight_traces(insight_ids[insight.key], insight.trace_refs)

    pending = {evaluation.name: evaluation for evaluation in evaluations}
    for _ in range(30):
        pending = {name: evaluation for name, evaluation in pending.items() if not api.evaluation_is_ready(evaluation)}
        if not pending:
            break
        sleep(1.0)
    if pending:
        raise DemoError(f"Timed out waiting for evaluation rollups: {', '.join(sorted(pending))}")

    print(
        f"Seeded workspace '{DEMO_WORKSPACE}' with {len(fixture.insights)} insights, "
        f"{len(fixture.groups)} groups, {len(evaluations)} evaluations, and 11 traces."
    )
    print(f"Open {api.base_url}/studio/workspaces/{DEMO_WORKSPACE}/optimizer")


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seed", "clean"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("NHX_BASE_URL", DEFAULT_BASE_URL),
        help=f"Platform base URL (default: NHX_BASE_URL or {DEFAULT_BASE_URL}).",
    )
    args = parser.parse_args(argv)

    try:
        with DemoAPI(args.base_url) as api:
            if args.command == "seed":
                seed(api)
            else:
                clean(api)
    except DemoError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
