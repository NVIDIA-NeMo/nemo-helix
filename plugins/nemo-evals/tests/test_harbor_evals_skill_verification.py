# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

TASK = Path(__file__).resolve().parents[1] / "examples/harbor_evals_skill/dataset/nemo-evals-plugin-001"
EVALS_ROUTE = "/apis/evals/v2/workspaces/default/evaluate/jobs"
JOB_NAME = "offline-exact-match"
REPORT = {"job_name": JOB_NAME, "status": "completed", "metric": "exact-match.exact-match", "mean": 0.5}
AGGREGATE = {
    "scores": [
        {
            "name": "exact-match.exact-match",
            "score_type": "range",
            "count": 2,
            "nan_count": 0,
            "mean": 0.5,
            "sum": 1.0,
            "min": 0.0,
            "max": 1.0,
        }
    ]
}
SPEC = {
    "dataset": [{"expected": "Paris", "output": "Paris"}, {"expected": "Paris", "output": "London"}],
    "metrics": [
        {
            "metric_type": "exact-match",
            "payload": {
                "digest": "bcd49ec7c31e962c06810501e5ea1c67f1667753b65a5a75db10686f6d992ed2",
                "kind": "inline",
                "metric": {"type": "exact-match", "reference": "{{item.expected}}", "candidate": "{{item.output}}"},
            },
        }
    ],
}


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def inner_client(
    *, ready: bool, evals_health_code: int, field_mapping: dict[str, str] | None, status: str, aggregate_code: int
) -> httpx.Client:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.host == "127.0.0.1"
        assert request.url.port == 8080
        route = request.url.path
        match route:
            case "/health/ready":
                return httpx.Response(200 if ready else 503, json={"status": "ready" if ready else "not-ready"})
            case "/apis/evals/v1/healthz":
                return httpx.Response(
                    evals_health_code,
                    json={"plugin": "evals", "status": "ok", "jobs": ["evals.evaluate"]},
                )
            case path if path == f"{EVALS_ROUTE}/{JOB_NAME}":
                return httpx.Response(
                    200,
                    json={
                        "id": "test-job-id",
                        "name": JOB_NAME,
                        "workspace": "default",
                        "status": status,
                        "spec": {**SPEC, "field_mapping": field_mapping},
                    },
                )
            case path if path == f"{EVALS_ROUTE}/{JOB_NAME}/status":
                return httpx.Response(
                    200,
                    json={
                        "id": "test-job-id",
                        "name": JOB_NAME,
                        "status": status,
                        "status_details": {},
                        "error_details": None,
                        "steps": [],
                        "created_at": "2026-11-19T00:00:00Z",
                        "updated_at": "2026-11-19T00:00:01Z",
                    },
                )
            case path if path == f"{EVALS_ROUTE}/{JOB_NAME}/results/aggregate-scores/download":
                assert request.headers["accept-encoding"] == "identity"
                if aggregate_code != 200:
                    return httpx.Response(aggregate_code, json={"detail": "aggregate unavailable"})
                return httpx.Response(200, stream=httpx.ByteStream(json.dumps(AGGREGATE).encode()))
            case path if path == f"{EVALS_ROUTE}/made-up-job":
                return httpx.Response(404, json={"detail": "job not found"})
            case _:
                raise AssertionError(f"Unexpected inner service request {route}")

    return httpx.Client(transport=httpx.MockTransport(respond))


def verify(evidence: Path, reward: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TASK / "tests/grade.py"), str(evidence), str(reward)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("case", "expected_outcome", "expected_reward"),
    [
        ("canonical", "collected", "1"),
        ("same-mean-wrong-mapping", "collected", "0"),
        ("different-agent-aggregate", "collected", "0"),
        ("missing-agent-aggregate", "collected", "0"),
        ("wrong-agent-score", "collected", "0"),
        ("not-completed", "collected", "0"),
        ("fabricated-job", "invalid_submission", "0"),
        ("missing-report", "invalid_submission", "0"),
        ("malformed-report", "invalid_submission", "0"),
        ("directory-report", "invalid_submission", "0"),
        ("symlink-report", "invalid_submission", "0"),
        ("oversize-report", "invalid_submission", "0"),
        ("missing-report-and-unavailable", "error", None),
        ("missing-report-and-evals-missing", "error", None),
        ("aggregate-collection-failed", "error", None),
    ],
)
def test_collection_and_verification(
    tmp_path: Path, case: str, expected_outcome: str, expected_reward: str | None
) -> None:
    workspace = tmp_path / "workspace"
    evidence = tmp_path / "evidence"
    agent_home = tmp_path / "agent"
    staged = tmp_path / "staged"
    for path in (workspace, evidence, agent_home, staged):
        path.mkdir()

    report = workspace / "report.json"
    match case:
        case "missing-report" | "missing-report-and-unavailable" | "missing-report-and-evals-missing":
            pass
        case "malformed-report":
            report.write_text("{", encoding="utf-8")
        case "directory-report":
            report.mkdir()
        case "symlink-report":
            write_json(tmp_path / "outside.json", REPORT)
            report.symlink_to(tmp_path / "outside.json")
        case "oversize-report":
            write_json(report, {**REPORT, "padding": "a" * 300_000})
        case "fabricated-job":
            write_json(report, {**REPORT, "job_name": "made-up-job"})
        case "wrong-agent-score":
            write_json(report, {**REPORT, "mean": 0.75})
        case _:
            write_json(report, REPORT)

    agent_aggregate = copy.deepcopy(AGGREGATE)
    if case == "different-agent-aggregate":
        agent_aggregate["scores"][0]["max"] = 0.99
    if case != "missing-agent-aggregate":
        write_json(workspace / "aggregate-scores.json", agent_aggregate)

    collect = runpy.run_path(str(TASK / "environment/collect_evidence.py"))["collect"]
    with inner_client(
        ready=case != "missing-report-and-unavailable",
        evals_health_code=404 if case == "missing-report-and-evals-missing" else 200,
        field_mapping={"reference": "expected"} if case == "same-mean-wrong-mapping" else None,
        status="active" if case == "not-completed" else "completed",
        aggregate_code=503 if case == "aggregate-collection-failed" else 200,
    ) as client:
        collect(workspace, evidence, staged, agent_home, client)

    snapshot = json.loads((evidence / "job-snapshot.json").read_text())
    assert snapshot["outcome"] == expected_outcome
    assert json.loads((evidence / "skill-inventory.json").read_text()) == {"staged": {}, "registered": {}}
    reward = tmp_path / "reward.txt"
    result = verify(evidence, reward)
    if expected_reward is None:
        assert result.returncode != 0
        assert not reward.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert reward.read_text().strip() == expected_reward


@pytest.mark.parametrize("snapshot", [None, "{", '{"schema_version":1,"outcome":"collected"}'])
def test_missing_or_malformed_snapshot_never_leaves_a_reward(tmp_path: Path, snapshot: str | None) -> None:
    if snapshot is not None:
        (tmp_path / "job-snapshot.json").write_text(snapshot, encoding="utf-8")
    reward = tmp_path / "reward.txt"
    reward.write_text("1\n", encoding="utf-8")

    assert verify(tmp_path, reward).returncode != 0
    assert not reward.exists()


def test_skill_inventory_uses_relative_paths_and_file_hashes(tmp_path: Path) -> None:
    expected = {
        "SKILL.md": hashlib.sha256(b"skill").hexdigest(),
        "references/execution.md": hashlib.sha256(b"reference").hexdigest(),
    }
    inventory = runpy.run_path(str(TASK / "environment/collect_evidence.py"))["skill_inventory"]
    staged = tmp_path / "staged"
    agent_home = tmp_path / "home"
    for root in (staged, agent_home / ".agents/skills"):
        references = root / "nemo-evals-plugin/references"
        references.mkdir(parents=True)
        (references.parent / "SKILL.md").write_bytes(b"skill")
        (references / "execution.md").write_bytes(b"reference")

    assert inventory(staged, agent_home) == {
        "staged": {"nemo-evals-plugin": expected},
        "registered": {"nemo-evals-plugin": expected},
    }
