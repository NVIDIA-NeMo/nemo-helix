# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the Kubernetes guide's request and command flags on the release CLI contract.

These checks are offline: they do not submit Jobs, build images, or contact a provider.
"""

import json
import re
from pathlib import Path

import typer
from click import Group
from nemo_evals.jobs.agent_spec import HarborRunnerTarget
from nemo_evals.runs_cli import add_runs_commands, run_request
from typer.main import get_command

ROOT = Path(__file__).resolve().parents[3]
GUIDE = ROOT / "docs/evals/agent-eval/kubernetes-taskset-runs.mdx"


def test_guide_request_is_valid_after_dataset_selection(tmp_path):
    blocks = re.findall(r"```json\n(.*?)\n```", GUIDE.read_text(), re.DOTALL)
    assert len(blocks) == 1
    body = json.loads(blocks[0])
    assert "tasks" not in body["spec"]  # --harbor-dir supplies the pinned selection.
    spec_file = tmp_path / "harbor-run.json"
    spec_file.write_text(blocks[0])
    request = run_request(
        "default/hello-world#" + "a" * 64,
        spec_file=spec_file,
        agent="ignored",
        dispatcher_profile=None,
        idempotency_key="hello-world-spec-1",
    )
    assert isinstance(request.spec.target, HarborRunnerTarget)
    assert request.execution.max_concurrent_task_jobs == 2
    assert request.execution.sandbox is not None
    assert request.execution.sandbox.cpus == 1
    assert request.execution.sandbox.memory_mb == 2048
    assert request.execution.sandbox.storage_mb == 10240
    assert request.timeout_seconds == 3600
    assert request.idempotency_key == "hello-world-spec-1"


def test_guide_uses_supported_run_commands_and_flags():
    def no_client():
        raise AssertionError("Documentation contract checks must not contact a deployment")

    app = typer.Typer()
    add_runs_commands(app, no_client)
    root = get_command(app)
    assert isinstance(root, Group)
    runs = root.commands["runs"]
    assert isinstance(runs, Group)
    text = GUIDE.read_text()
    shell_blocks = re.findall(r"```bash\n(.*?)\n```", text, re.DOTALL)
    commands = "\n".join(shell_blocks).replace("\\\n", " ")
    for line in commands.splitlines():
        if not line.startswith("nemo evals runs "):
            continue
        command, _, arguments = line.removeprefix("nemo evals runs ").partition(" ")
        assert command in runs.commands
        options = {opt for param in runs.commands[command].params for opt in param.opts}
        used_options = set(re.findall(r"--[a-z][a-z-]*", arguments))
        assert used_options <= options, f"Unsupported {command} options: {used_options - options}"
    assert "--taskset" not in text
    assert "tasksets create" not in text
    assert "tasksets build-images" not in text
    assert (ROOT / "packages/nhx_evals_sdk/examples/harbor/hello_world_dataset/hello-world/task.toml").is_file()
