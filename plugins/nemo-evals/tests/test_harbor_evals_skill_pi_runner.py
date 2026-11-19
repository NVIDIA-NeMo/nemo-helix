# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Pi example against the NeMo Helix 0.10.0 CLI response shapes."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
from nemo_evals.api.taskset_run_schemas import TasksetRunInput
from nemo_evals.sdk.task_preparation import describe_harbor_task
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import discover_harbor_tasks

SOLUTION = Path(__file__).parents[1] / "examples/harbor_evals_skill/dataset/nemo-evals-plugin-003/solution"
TASKSET = "demo/pi-digit-abcdef#" + "a" * 64
RUN_ID = "run-0123456789abcdef"
READY_RUN = {"run_id": RUN_ID, "taskset_ref": TASKSET, "phase": "completed", "result_state": "ready"}


@pytest.fixture
def pi_eval():
    spec = importlib.util.spec_from_file_location("harbor_pi_eval", SOLUTION / "pi_eval.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result_with_reward(reward):
    return {
        "run": READY_RUN.copy(),
        "tasks": [
            {
                "state": "succeeded",
                "result": {
                    "summary": {"trial_count": 1, "error_count": 0},
                    "scores": {
                        "scores": [{"name": "harbor_reward.reward", "count": 1, "nan_count": 0, "mean": reward}]
                    },
                },
            }
        ],
    }


@pytest.mark.parametrize("reward", [0, 1.0])
def test_runner_waits_for_ready_and_outputs_one_binary_reward(pi_eval, monkeypatch, capsys, reward):
    command_calls = []
    statuses = iter([READY_RUN | {"result_state": "pending"}, READY_RUN])

    def nemo_output(command, *, timeout):
        assert timeout > 0
        assert command[:4] == ["nemo", "--context", "dev", "evals"]
        args = command[4:]
        command_calls.append(args[:2])
        if args[:2] == ["tasksets", "build-images"]:
            assert args[2:] == [TASKSET, "--wait"]
            return "1 image ready"
        if args[:2] == ["runs", "submit"]:
            assert args[2:6] == ["--taskset", TASKSET, "--workspace", "demo"]
            payload = json.loads(Path(args[args.index("--spec-file") + 1]).read_text())
            TasksetRunInput.model_validate(payload)
            assert payload["idempotency_key"] == "pi-smoke-a"
            assert payload["spec"]["tasks"] == TASKSET
            assert payload["spec"]["target"] == {
                "kind": "harbor",
                "source": {"name": "codex", "model": {"ref": "demo/gateway"}},
                "agent_kwargs": {"version": "0.162.1", "disable_web_search": True},
                "closed_book": True,
                "n_attempts": 1,
                "env_secrets": {"OPENAI_API_KEY": "demo/inference"},
            }
            assert payload["execution"]["sandbox"] == {"profile": "harbor-standard"}
            if reward == 0:
                return f"Run {RUN_ID} accepted: 1 tasks, queued"
            return f"Run {RUN_ID} was already submitted with this key and is running; nothing new started"
        assert args[2:] == [RUN_ID, "--workspace", "demo", "--json"]
        if args[1] == "status":
            return json.dumps({"run": next(statuses)})
        assert args[1] == "results"
        return json.dumps(result_with_reward(reward))

    monkeypatch.setattr(pi_eval, "nemo_output", nemo_output)
    monkeypatch.setattr(pi_eval, "POLL_SECONDS", 0)
    assert (
        pi_eval.main(
            [
                "--taskset",
                TASKSET,
                "--model-ref",
                "demo/gateway",
                "--context",
                "dev",
                "--inference-key-secret",
                "demo/inference",
                "--idempotency-key",
                "pi-smoke-a",
            ]
        )
        == 0
    )
    assert command_calls == [
        ["tasksets", "build-images"],
        ["runs", "submit"],
        ["runs", "status"],
        ["runs", "status"],
        ["runs", "results"],
    ]
    assert json.loads(capsys.readouterr().out) == {"run_id": RUN_ID, "taskset_ref": TASKSET, "reward": reward}


@pytest.mark.parametrize(
    "section, field, value",
    [
        ("run", "taskset_ref", "other/taskset#" + "b" * 64),
        ("run", "phase", "failed"),
        ("run", "result_state", "pending"),
        ("task", "state", "failed"),
        ("task", "result", None),
        ("summary", "trial_count", 2),
        ("summary", "error_count", 1),
        ("summary", "error_count", None),
        ("score", "name", "harbor_reward.reward.pass@1"),
        ("score", "count", 0),
        ("score", "nan_count", 1),
        ("score", "mean", None),
        ("score", "mean", 0.5),
        ("score", "mean", True),
    ],
)
def test_result_rejects_incomplete_or_wrong_trials(pi_eval, section, field, value):
    response = result_with_reward(0.0)
    task = response["tasks"][0]
    result = task["result"]
    sections = {
        "run": response["run"],
        "task": task,
        "summary": result["summary"],
        "score": result["scores"]["scores"][0],
    }
    sections[section][field] = value
    with pytest.raises(ValueError):
        pi_eval.measured_reward(response, run_id=RUN_ID, taskset_ref=TASKSET)


def test_release_result_with_a_completed_job_status(pi_eval):
    response = result_with_reward(1.0)
    task = response["tasks"][0]
    del task["state"]
    task["status"] = "completed"
    assert pi_eval.measured_reward(response, run_id=RUN_ID, taskset_ref=TASKSET) == 1


@pytest.mark.parametrize(
    "answer, reward", [(b"9", "1\n"), (b"9\n", "1\n"), (b"8", "0\n"), (b"99", "0\n"), (None, "0\n")]
)
def test_inner_task_packages_and_scores_the_requested_digit(tmp_path, answer, reward):
    (task,) = discover_harbor_tasks(SOLUTION / "pi-digit")
    spec = describe_harbor_task(task, workspace="default").spec
    assert spec.kind == "harbor"
    assert spec.native_task_id == "pi-digit"
    assert spec.config["environment"]["network_mode"] == "no-network"

    answer_path = tmp_path / "pi_digit.txt"
    reward_path = tmp_path / "reward.txt"
    if answer is not None:
        answer_path.write_bytes(answer)
    subprocess.run(["sh", str(task.source_dir / "tests/test.sh"), str(answer_path), str(reward_path)], check=True)
    assert reward_path.read_text() == reward
