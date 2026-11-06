# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the Kubernetes Harbor guide and the skill's recipe on the CLI and request contract.

Offline: nothing here submits a run, builds an image or contacts a deployment.
"""

import json
import os
import re
import shlex
import subprocess
import sys
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


def _json_blocks(kind: str) -> list[dict]:
    """The guide's JSON blocks of one kind, told apart by their keys.

    ``request``: a whole request file (``"spec"``); ``closed_book``: the closed-book target fragment; ``sizing``: the
    request fragment that sizes sandboxes and limits concurrency.
    """
    blocks = _blocks(GUIDE, "json")
    kinds = {
        "request": lambda block: '"spec"' in block and '"closed_book"' not in block,
        "closed_book": lambda block: '"closed_book"' in block,
        "sizing": lambda block: '"spec"' not in block and '"closed_book"' not in block,
    }
    return [json.loads(block) for block in blocks if kinds[kind](block)]


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
    assert _json_blocks("request") == [json.loads(OPENHANDS_SPEC.read_text())]


def test_openhands_request_is_valid_once_the_task_source_supplies_the_taskset(tmp_path: Path) -> None:
    spec_file = tmp_path / "openhands-run.json"
    spec_file.write_text(OPENHANDS_SPEC.read_text())
    request = run_request(
        TASKSET_REF, spec_file=spec_file, agent="ignored", dispatcher_profile=None, idempotency_key="docs-1"
    )
    assert isinstance(request.spec.target, HarborRunnerTarget)
    assert request.spec.target.source.model_dump(exclude_none=True) == {
        "name": "openhands",
        "model": {"ref": "default/nvidia-nemotron-3-5-super-vl-120b-a12b"},
    }
    assert request.spec.target.model_dump(mode="json")["env_secrets"] == {"LLM_API_KEY": "evals-inference-key"}


def test_guide_sizing_fragment_is_valid_in_a_request() -> None:
    fragments = _json_blocks("sizing")
    assert len(fragments) == 1
    openhands = json.loads(OPENHANDS_SPEC.read_text())
    request = TasksetRunInput.model_validate(
        {**openhands, **fragments[0], "spec": {**openhands["spec"], "tasks": TASKSET_REF}}
    )
    assert request.execution.sandbox is not None
    assert request.execution.max_concurrent_task_jobs == 5
    assert request.timeout_seconds == 7200


def test_guide_closed_book_fragment_makes_the_openhands_request_closed_book(tmp_path: Path) -> None:
    """The opt-in fragment merges into the default request's target: labelled closed-book, on the closed-book model."""
    (fragment,) = _json_blocks("closed_book")
    openhands = json.loads(OPENHANDS_SPEC.read_text())
    openhands["spec"]["target"].update(fragment)
    spec_file = tmp_path / "openhands-closed-book.json"
    spec_file.write_text(json.dumps(openhands))

    request = run_request(
        TASKSET_REF, spec_file=spec_file, agent="ignored", dispatcher_profile=None, idempotency_key="docs-cb"
    )

    assert isinstance(request.spec.target, HarborRunnerTarget)
    assert request.spec.target.closed_book is True
    assert request.spec.target.source.model_dump(exclude_none=True) == {
        "name": "openhands",
        "model": {"ref": "default/nemotron-super-closed-book"},
    }


def test_quickstart_suite_exists_with_one_task() -> None:
    suite = ROOT / "packages/nhx_evals_sdk/examples/harbor/hello_world_dataset"
    assert [task.parent.name for task in suite.glob("*/task.toml")] == ["hello-world"]


@pytest.mark.parametrize("command", ["results", "tasks"])
def test_troubleshooting_diagnostics_name_the_run_and_workspace(command: str) -> None:
    """The copyable diagnostics must parse as a lookup of the user's run in their workspace."""
    troubleshooting = (SKILL / "references/troubleshooting.md").read_text()
    example = re.findall(rf"\x60(nemo evals runs {command}[^\x60]*)\x60", troubleshooting)[-1]
    arguments = shlex.split(example.replace("$RUN_ID", "run-0123456789abcdef").replace("$WS", "team"))
    runs = _cli().commands["runs"]
    assert isinstance(runs, Group)
    with runs.commands[command].make_context(command, arguments[4:]) as context:
        assert context.params["run_id"] == "run-0123456789abcdef"
        assert context.params["workspace"] == "team"
        if command == "tasks":
            assert context.params["phase"] == "failed"


@pytest.mark.parametrize("path", [GUIDE, RECIPE], ids=["guide", "skill-recipe"])
def test_publish_once_examples_use_the_created_taskset_and_source_label(path: Path, tmp_path: Path) -> None:
    """Execute the shell example so stale variables and placeholder source versions cannot pass."""
    blocks = _blocks(path, "bash")
    start = max(
        index
        for index, block in enumerate(blocks)
        if "nemo evals tasksets create" in block and "datacurve/deep-swe" in block
    )
    end = next(index for index in range(start, len(blocks)) if "nemo evals runs submit" in blocks[index])
    snippet = "\n".join(blocks[start : end + 1])
    published_ref = "team/example-taskset#" + "0" * 64
    source_label = "datacurve/deep-swe@sha256:" + "a" * 64
    created = {
        "taskset_ref": published_ref,
        "tasks": ["team/example-task#" + "b" * 64],
        "total": 3,
        "intake_dataset": source_label,
    }
    calls_path = tmp_path / "calls.jsonl"
    fake_cli = tmp_path / "nemo"
    fake_cli.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"created = {created!r}\n"
        "arguments = sys.argv[1:]\n"
        "with open(os.environ['TEST_NEMO_CALLS'], 'a') as calls:\n"
        "    calls.write(json.dumps(arguments) + '\\n')\n"
        "if arguments[:3] == ['evals', 'tasksets', 'create']:\n"
        "    if '--json' in arguments:\n"
        "        print(json.dumps(created))\n"
        "    else:\n"
        "        print('Intake label: --intake-dataset ' + created['intake_dataset'])\n"
        "        print(created['taskset_ref'])\n"
        "elif arguments[:3] not in [['evals', 'tasksets', 'build-images'], ['evals', 'runs', 'submit']]:\n"
        "    raise SystemExit('unexpected CLI command in publish-once example')\n"
    )
    fake_cli.chmod(0o755)
    subprocess.run(
        ["bash", "-eu", "-c", snippet],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "TEST_NEMO_CALLS": str(calls_path),
            "WS": "team",
            "TASKSET": "stale/taskset",
            "INTAKE_DATASET": "stale/dataset@old",
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    (submission,) = [call for call in calls if call[:3] == ["evals", "runs", "submit"]]
    runs = _cli().commands["runs"]
    assert isinstance(runs, Group)
    with runs.commands["submit"].make_context("submit", submission[3:]) as context:
        assert context.params["workspace"] == "team"
        assert context.params["taskset"] == published_ref
        assert context.params["intake_dataset"] == source_label
