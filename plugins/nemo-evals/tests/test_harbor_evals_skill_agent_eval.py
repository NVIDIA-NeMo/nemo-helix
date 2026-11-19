# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
from pathlib import Path

import pytest

TASK = Path(__file__).resolve().parents[1] / "examples/harbor_evals_skill/dataset/nemo-evals-plugin-002"
ALIAS = "agent-smoke-supercool-7cf95d1e203ab864"
METRIC = "exact-match.exact-match"


@pytest.fixture
def snapshot() -> dict:
    task_pin = {"entity_name": "default/agent-smoke", "revision_digest": "a" * 64}
    return {
        "schema_version": 1,
        "outcome": "collected",
        "expected_task": task_pin,
        "report": {"job_name": "example", "run_alias": ALIAS, "status": "completed", "metric": METRIC, "mean": 1.0},
        "job": {
            "name": "example",
            "status": "completed",
            "run_alias": ALIAS,
            "target_matches": True,
            "task_provenance": [task_pin.copy()],
            "task_ids": ["agent-smoke"],
            "trials": [{"task_id": "agent-smoke", "status": "completed", "output": {"output_text": "ready"}}],
            "summary": {
                "trial_count": 1,
                "task_count": 1,
                "error_count": 0,
                "scores": {"scores": [{"name": METRIC, "count": 1, "nan_count": 0, "mean": 1.0}]},
            },
        },
    }


def _grade(root: Path, evidence: dict) -> tuple[int, dict | None]:
    grade = runpy.run_path(str(TASK / "tests/grade.py"))["grade"]
    (root / "job-snapshot.json").write_text(json.dumps(evidence))
    reward = root / "reward.json"
    status = grade(root, reward)
    return status, json.loads(reward.read_text()) if reward.exists() else None


@pytest.mark.parametrize("alias,expected", [(ALIAS, 1), ("agent-smoke", 0), ("wrong-" + ALIAS, 0)])
def test_verified_alias_distinguishes_valid_inner_jobs(tmp_path: Path, snapshot: dict, alias: str, expected: int):
    snapshot["job"]["run_alias"] = snapshot["report"]["run_alias"] = alias
    snapshot["report"]["note"] = "The smoke task completed."
    assert _grade(tmp_path, snapshot) == (0, {"reward": expected, "inner_eval_passed": 1, "suffix_applied": expected})


@pytest.mark.parametrize("invalid", ["report", "revision", "target", "trial"])
def test_invalid_agent_evaluation_earns_zero(tmp_path: Path, snapshot: dict, invalid: str):
    if invalid == "report":
        snapshot["job"]["run_alias"] = "agent-smoke"
    elif invalid == "revision":
        snapshot["job"]["task_provenance"][0]["revision_digest"] = "b" * 64
    elif invalid == "target":
        snapshot["job"]["target_matches"] = False
    else:
        snapshot["job"]["trials"][0]["output"]["output_text"] = "guessed"
    status, result = _grade(tmp_path, snapshot)
    assert status == 0 and result is not None
    assert result["reward"] == result["inner_eval_passed"] == 0


def test_collector_failure_leaves_no_reward(tmp_path: Path):
    snapshot = {"schema_version": 1, "outcome": "error", "reason": "collector could not reach Helix"}
    assert _grade(tmp_path, snapshot) == (1, None)
