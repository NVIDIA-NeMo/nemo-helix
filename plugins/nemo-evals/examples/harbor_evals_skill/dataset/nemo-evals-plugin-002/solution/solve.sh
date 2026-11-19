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
import tarfile
import time
from pathlib import Path


def read_json(*command):
    return json.loads(subprocess.check_output(command, timeout=30))


spec = Path("/workspace/oracle-spec.json")
payload = json.loads(Path("/workspace/agent-evaluate-spec.json").read_text())
alias = payload["labels"]["run_alias"] + "-supercool-7cf95d1e203ab864"
payload["labels"]["run_alias"] = alias
spec.write_text(json.dumps(payload))
response = read_json("nemo", "evals", "agent-evaluate", "--spec-file", str(spec), "--workspace", "default")
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

archive = Path("/workspace/agent-eval-results.tar.gz")
subprocess.run(
    [
        "nemo", "jobs", "results", "download", "agent-eval-results",
        "--job", job, "--workspace", "default", "--output-file", str(archive),
    ],
    check=True, timeout=30,
)
with tarfile.open(archive) as bundle:
    member = next(member for member in bundle if Path(member.name).name == "summary.json")
    summary = json.load(bundle.extractfile(member))
score = next(score for score in summary["scores"]["scores"] if score["name"] == "exact-match.exact-match")
report = {"job_name": job, "run_alias": alias, "status": status, "metric": score["name"], "mean": score["mean"]}
Path("/workspace/report.json").write_text(json.dumps(report) + "\n")
print(f"{job} completed with {score['name']} mean {score['mean']}")
PY
