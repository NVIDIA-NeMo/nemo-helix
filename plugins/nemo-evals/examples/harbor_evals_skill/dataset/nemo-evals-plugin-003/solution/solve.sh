#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu
export NHX_BASE_URL=http://127.0.0.1:8080 NEMO_BASE_URL=http://127.0.0.1:8080
export NEMO_TELEMETRY_ENABLED=false

solution=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd /workspace
cp -R "$solution/pi-digit" .
cp "$solution/pi_eval.py" .
chmod +x pi_eval.py

receipt=$(mktemp /workspace/.taskset.XXXXXX)
trap 'rm -f "$receipt"' EXIT
nemo evals tasksets create --harbor-dir /workspace/pi-digit --workspace default --json > "$receipt"
python - "$receipt" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

from nemo_evals.sdk.resources import Evaluator
from nemo_helix_plugin.client.client import NemoClient


def split_pin(value):
    match = re.fullmatch(r"default/([a-z0-9-]+)#([0-9a-f]{64})", value)
    if match is None:
        raise ValueError(f"expected a qualified content digest in the default workspace: {value!r}")
    return match.groups()


receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
(task_ref,) = receipt["tasks"]
task_name, task_digest = split_pin(task_ref)
taskset_name, taskset_digest = split_pin(receipt["taskset_ref"])
with NemoClient(base_url=os.environ["NHX_BASE_URL"], workspace="default", timeout=30) as client:
    evaluator = Evaluator.from_client(client)
    task = evaluator.tasks.retrieve(task_name, revision=task_digest, workspace="default")
    taskset = evaluator.tasksets.retrieve(taskset_name, revision=taskset_digest, workspace="default")
if task.spec.kind != "harbor" or task.spec.native_task_id != "pi-digit":
    raise ValueError(f"unexpected task at {task_ref}")
if [ref.root for ref in taskset.tasks] != [task_ref]:
    raise ValueError(f"{receipt['taskset_ref']} does not contain exactly {task_ref}")
PY
mv "$receipt" /workspace/taskset.json
