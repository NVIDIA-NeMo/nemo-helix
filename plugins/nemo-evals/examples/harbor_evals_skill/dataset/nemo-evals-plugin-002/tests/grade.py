# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the collected server job and emit deterministic Harbor rewards."""

import json
import sys
from pathlib import Path
from typing import Any

METRIC = "exact-match.exact-match"
ALIAS = "agent-smoke-supercool-7cf95d1e203ab864"


def _valid_inner_eval(job: dict[str, Any], report: dict[str, Any], expected_task: dict[str, str]) -> bool:
    if job["status"] != "completed" or job["target_matches"] is not True or job["task_provenance"] != [expected_task]:
        return False
    task_ids, trials = job["task_ids"], job["trials"]
    if len(task_ids) != 1 or len(trials) != 1:
        return False
    (trial,) = trials
    if (
        trial["task_id"] != task_ids[0]
        or trial["status"] != "completed"
        or trial.get("error") is not None
        or trial["output"]["output_text"] != "ready"
    ):
        return False
    summary = job["summary"]
    metrics = [score for score in summary["scores"]["scores"] if score["name"] == METRIC]
    expected_report = {
        "job_name": job["name"],
        "run_alias": job["run_alias"],
        "status": "completed",
        "metric": METRIC,
        "mean": 1.0,
    }
    return (
        summary["trial_count"] == summary["task_count"] == 1
        and summary["error_count"] == 0
        and len(metrics) == 1
        and metrics[0]["count"] == metrics[0]["mean"] == 1
        and metrics[0]["nan_count"] == 0
        and all(key in report and report[key] == value for key, value in expected_report.items())
        and type(report["mean"]) in (int, float)
    )


def grade(evidence: Path, reward: Path) -> int:
    reward.unlink(missing_ok=True)
    try:
        snapshot = json.loads((evidence / "job-snapshot.json").read_bytes())
        if type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != 1:
            raise ValueError("unsupported job snapshot version")
        outcome = snapshot["outcome"]
        if outcome == "collected":
            job = snapshot["job"]
            passed = _valid_inner_eval(job, snapshot["report"], snapshot["expected_task"])
            suffix = job["run_alias"] == ALIAS
        elif outcome == "invalid_submission":
            print(f"Invalid submission: {snapshot['reason']}", file=sys.stderr)
            passed = suffix = False
        else:
            raise ValueError(f"collector reported {outcome}: {snapshot.get('reason')}")
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        print(f"Verifier error: {exc}", file=sys.stderr)
        return 1

    rewards = {"reward": int(passed and suffix), "inner_eval_passed": int(passed), "suffix_applied": int(suffix)}
    reward.parent.mkdir(parents=True, exist_ok=True)
    reward.write_text(json.dumps(rewards) + "\n", encoding="utf-8")
    print(f"Agent evaluation rewards: {rewards}")
    return 0


if __name__ == "__main__":
    sys.exit(grade(Path(sys.argv[1]), Path(sys.argv[2])))
