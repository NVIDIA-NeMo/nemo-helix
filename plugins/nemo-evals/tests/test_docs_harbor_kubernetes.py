# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the Kubernetes Harbor guide and the skill's recipe on the CLI and request contract.

Offline: nothing here submits a run, builds an image or contacts a deployment.
"""

import json
import re
from pathlib import Path

import pytest
import typer
from click import Group
from nemo_evals.api.taskset_run_schemas import TasksetRunInput
from nemo_evals.jobs.agent_spec import HarborRunnerTarget
from nemo_evals.runs_cli import add_runs_commands, run_request
from nemo_evals.tasksets_cli import add_tasksets_commands
from typer.main import get_command

ROOT = Path(__file__).resolve().parents[3]
GUIDE = ROOT / "docs/evals/agent-eval/harbor-kubernetes.mdx"
SKILL = ROOT / "skills/nemo-evals-plugin"
RECIPE = SKILL / "references/harbor-kubernetes.md"
OPENHANDS_SPEC = SKILL / "assets/specs/harbor_taskset_run_openhands.json"
PROFILES_GUIDE = ROOT / "docs/set-up/helm/harbor-sandbox-profiles.mdx"
TASKSET_REF = "default/example-taskset#" + "0" * 64


def _blocks(path: Path, language: str) -> list[str]:
    """Fenced blocks of ``language``, dedented, including ones nested in list items."""
    blocks = re.findall(rf"^([ \t]*)```{language}\n(.*?)^\1```", path.read_text(), re.DOTALL | re.MULTILINE)
    return ["\n".join(line.removeprefix(indent) for line in body.splitlines()) for indent, body in blocks]


def _evals_commands(path: Path) -> list[tuple[str, str, str]]:
    """Every ``nemo evals runs|tasksets <command> <arguments>`` in the file's bash blocks."""
    commands = "\n".join(_blocks(path, "bash")).replace("\\\n", " ")
    return re.findall(r"nemo evals (runs|tasksets) ([a-z-]+)([^|\n]*)", commands)


def _cli() -> Group:
    def no_client():
        raise AssertionError("documentation contract checks must not contact a deployment")

    app = typer.Typer()
    add_runs_commands(app, no_client)
    add_tasksets_commands(app, no_client)
    root = get_command(app)
    assert isinstance(root, Group)
    return root


@pytest.mark.parametrize("path", [GUIDE, RECIPE, PROFILES_GUIDE], ids=["guide", "skill-recipe", "profiles-guide"])
def test_commands_use_supported_subcommands_and_flags(path: Path) -> None:
    root = _cli()
    used = _evals_commands(path)
    assert used, f"{path.name} shows no nemo evals runs/tasksets commands"
    for group, command, arguments in used:
        commands = root.commands[group]
        assert isinstance(commands, Group)
        assert command in commands.commands, f"{path.name}: nemo evals {group} {command} doesn't exist"
        options = {opt for param in commands.commands[command].params for opt in param.opts}
        unsupported = set(re.findall(r"(?<!\S)--[a-z][a-z-]*", arguments)) - options
        assert not unsupported, f"{path.name}: nemo evals {group} {command} has no {sorted(unsupported)}"


def test_guide_openhands_request_is_the_skill_asset() -> None:
    requests = [json.loads(block) for block in _blocks(GUIDE, "json") if '"spec"' in block]
    assert requests == [json.loads(OPENHANDS_SPEC.read_text())]


def test_openhands_request_is_valid_once_the_task_source_supplies_the_taskset(tmp_path: Path) -> None:
    spec_file = tmp_path / "openhands-run.json"
    spec_file.write_text(OPENHANDS_SPEC.read_text())
    request = run_request(
        TASKSET_REF, spec_file=spec_file, agent="ignored", dispatcher_profile=None, idempotency_key="docs-1"
    )
    assert isinstance(request.spec.target, HarborRunnerTarget)
    assert request.spec.target.source.model_dump(exclude_none=True) == {
        "name": "openhands",
        "model": {"ref": "default/nemotron-super-closed-book"},
    }
    assert request.spec.target.model_dump(mode="json")["env_secrets"] == {"LLM_API_KEY": "evals-inference-key"}


def test_guide_sizing_fragment_is_valid_in_a_request() -> None:
    fragments = [json.loads(block) for block in _blocks(GUIDE, "json") if '"spec"' not in block]
    assert len(fragments) == 1
    openhands = json.loads(OPENHANDS_SPEC.read_text())
    request = TasksetRunInput.model_validate(
        {**openhands, **fragments[0], "spec": {**openhands["spec"], "tasks": TASKSET_REF}}
    )
    assert request.execution.sandbox is not None
    assert request.execution.max_concurrent_task_jobs == 5
    assert request.timeout_seconds == 7200


def test_quickstart_suite_exists_with_one_task() -> None:
    suite = ROOT / "packages/nhx_evals_sdk/examples/harbor/hello_world_dataset"
    assert [task.parent.name for task in suite.glob("*/task.toml")] == ["hello-world"]
