#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu
export NHX_BASE_URL=http://127.0.0.1:8080 NEMO_BASE_URL=http://127.0.0.1:8080
export NEMO_TELEMETRY_ENABLED=false
cd /workspace

python - <<'PY'
import json
import subprocess
import time
from pathlib import Path


def read_json(*command):
    return json.loads(subprocess.check_output(command, timeout=30))


response = read_json(
    "nemo", "evals", "evaluate", "--spec-file", "/workspace/exact_match_metric.json", "--workspace", "default"
)
job = response["name"]
deadline = time.monotonic() + 300
while time.monotonic() < deadline:
    status = read_json("nemo", "jobs", "get-status", job, "--workspace", "default", "-f", "json")["status"]
    if status == "completed":
        break
    if status in {"error", "cancelled"}:
        raise SystemExit(f"{job} ended in {status}")
    time.sleep(2)
else:
    raise SystemExit(f"{job} did not complete")

aggregate = Path("/workspace/aggregate-scores.json")
subprocess.run(
    [
        "nemo", "jobs", "results", "download", "aggregate-scores",
        "--job", job, "--workspace", "default", "--output-file", str(aggregate),
    ],
    check=True, timeout=30,
)
(score,) = json.loads(aggregate.read_text())["scores"]
report = {"job_name": job, "status": status, "metric": score["name"], "mean": score["mean"]}
Path("/workspace/report.json").write_text(json.dumps(report) + "\n")
print(f"{job} completed with {score['name']} mean {score['mean']}")
PY
