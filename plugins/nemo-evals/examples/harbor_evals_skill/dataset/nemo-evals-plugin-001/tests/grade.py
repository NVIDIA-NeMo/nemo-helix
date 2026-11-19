# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grade transferred evidence without the agent sandbox or a model."""

import json
import sys
from pathlib import Path
from typing import Any

METRIC = "exact-match.exact-match"
EXPECTED_MEAN = 0.5
FIXTURE = Path(__file__).with_name("expected-spec.json")


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    # Python's decoder also accepts non-finite floats, which cannot be scored.
    json.dumps(value, allow_nan=False)
    return value


def read_snapshot(path: Path) -> dict[str, Any]:
    snapshot = read_object(path)
    if type(snapshot.get("schema_version")) is not int or snapshot["schema_version"] != 1:
        raise ValueError("unsupported job snapshot version")
    outcome = snapshot["outcome"]
    reason = snapshot["reason"]
    if outcome == "error":
        raise ValueError(f"collector failed: {reason}")
    job = snapshot["job"]
    if outcome == "invalid_submission" and job is None and isinstance(reason, str) and reason:
        return snapshot
    if outcome != "collected" or reason is not None or not isinstance(job, dict):
        raise ValueError("malformed job snapshot outcome or job")
    if not all(isinstance(job.get(key), str) and job[key] for key in ("id", "name", "status")):
        raise ValueError("malformed job snapshot identity or status")
    if job["workspace"] != "default":
        raise ValueError("job snapshot is from a different workspace")

    spec = job["spec"]
    if (
        not isinstance(spec, dict)
        or spec.keys() != {"dataset", "metrics", "has_target", "field_mapping", "prompt_template"}
        or not isinstance(spec["dataset"], (list, str))
        or not isinstance(spec["metrics"], list)
        or any(
            not isinstance(metric, dict) or metric.keys() != {"metric_type", "kind", "type", "reference", "candidate"}
            for metric in spec["metrics"]
        )
        or not isinstance(spec["has_target"], bool)
    ):
        raise ValueError("malformed job snapshot evaluation spec")

    aggregate = job["aggregate"]
    if job["status"] == "completed":
        scores = aggregate.get("scores") if isinstance(aggregate, dict) else None
        if not isinstance(scores, list) or any(
            not isinstance(score, dict) or not isinstance(score.get("name"), str) for score in scores
        ):
            raise ValueError("malformed job snapshot aggregate")
    elif aggregate is not None:
        raise ValueError("unexpected aggregate for an incomplete job")
    return snapshot


def submission_error(evidence: Path, snapshot: dict[str, Any], expected_spec: dict[str, Any]) -> str | None:
    if snapshot["outcome"] == "invalid_submission":
        return snapshot["reason"]
    job = snapshot["job"]
    if job["status"] != "completed":
        return "the reported job did not complete"
    if job["spec"] != expected_spec:
        return "the service evaluation spec differs from the canonical offline exact-match fixture"

    scores = [score for score in job["aggregate"]["scores"] if score["name"] == METRIC]
    if (
        len(scores) != 1
        or type(scores[0].get("count")) is not int
        or scores[0]["count"] != 2
        or type(scores[0].get("nan_count")) is not int
        or scores[0]["nan_count"] != 0
        or scores[0].get("mean") != EXPECTED_MEAN
    ):
        return "the server aggregate does not have the expected exact-match statistics"

    try:
        report = read_object(evidence / "agent-report.json")
        aggregate = read_object(evidence / "agent-aggregate-scores.json")
    except (OSError, ValueError, RecursionError) as exc:
        return f"missing or malformed submission artifact: {exc}"
    expected_report = {"job_name": job["name"], "status": job["status"], "metric": METRIC, "mean": scores[0]["mean"]}
    if any(report.get(key) != value for key, value in expected_report.items()):
        return "the submitted report disagrees with the server job"
    if json.dumps(aggregate, sort_keys=True) != json.dumps(job["aggregate"], sort_keys=True):
        return "the submitted aggregate disagrees with the server aggregate"
    return None


def grade(evidence: Path, reward: Path) -> int:
    reward.unlink(missing_ok=True)
    try:
        snapshot = read_snapshot(evidence / "job-snapshot.json")
        expected_spec = read_object(FIXTURE)
    except (OSError, ValueError, KeyError, RecursionError) as exc:
        print(f"Verifier error: {exc}", file=sys.stderr)
        return 1

    error = submission_error(evidence, snapshot, expected_spec)
    if error is not None:
        print(f"Invalid submission: {error}", file=sys.stderr)
    reward.parent.mkdir(parents=True, exist_ok=True)
    reward.write_text("1\n" if error is None else "0\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(grade(Path(sys.argv[1]), Path(sys.argv[2])))
