# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_LAUNCHER = Path(__file__).resolve().parents[1] / "examples" / "harbor_skill_injection" / "run_codex_demo.sh"
_TASKSET = "demo/evals-taskset#" + "a" * 64
_SKILL = "demo/nemo-evals-plugin#" + "b" * 64
_BASELINE_ID = "run-0000000000000001"
_TREATMENT_ID = "run-0000000000000002"


@pytest.fixture
def launcher_env(tmp_path: Path) -> dict[str, str]:
    """Run the real Bash launcher with a platform CLI double on PATH."""
    for executable in ("bash", "jq", "python3"):
        if shutil.which(executable) is None:
            pytest.skip(f"Launcher uses {executable}")
    stub = tmp_path / "nemo"
    stub.write_text(
        """#!/usr/bin/env python3
import io
import json
import os
import sys
import tarfile
from pathlib import Path

args = sys.argv[1:]
assert args[:2] == ["--context", "demo-context"], args
args = args[2:]
with Path(os.environ["CALLS"]).open("a") as calls:
    calls.write(json.dumps(args) + "\\n")

def option(name):
    return args[args.index(name) + 1]

def out(data):
    print(json.dumps(data))

mode = os.environ.get("STUB_MODE", "ev")
if args[:3] == ["evals", "tasksets", "create"]:
    out({"taskset_ref": "demo/supercool-taskset#" + "c" * 64})
elif args[:3] == ["evals", "tasksets", "build-images"]:
    assert args[3] == "demo/supercool-taskset#" + "c" * 64 and "--wait" in args, args
elif args[:3] == ["agents", "skills", "publish"]:
    assert args[3] == "supercool-guidelines", args
    out({"content_hash": "d" * 64})
elif args[:3] == ["evals", "runs", "submit"]:
    spec_file = Path(option("--spec-file"))
    spec = json.loads(spec_file.read_text())
    treatment = bool(spec["spec"]["target"]["skills"])
    run_id = "run-0000000000000002" if treatment else "run-0000000000000001"
    with Path(os.environ["SUBMISSIONS"]).open("a") as submissions:
        submissions.write(json.dumps({"spec": spec, "taskset": option("--taskset")}) + "\\n")
    if treatment and mode == "ambiguous":
        print("The run may already be accepted", file=sys.stderr)
        sys.exit(1)
    if treatment and mode == "replay":
        print(f"Run {run_id} was already submitted with this key and is completed; nothing new started")
    else:
        print(f"Run {run_id} accepted: 1 tasks, queued")
elif args[:3] == ["evals", "runs", "status"]:
    assert "--watch" in args, args
elif args[:3] == ["evals", "runs", "tasks"]:
    out({"data": [{"task": "task-0", "task_index": 0, "job": args[3] + "-task-0"}]})
elif args[:2] == ["jobs", "get"]:
    second = args[2] == "run-0000000000000002-task-0"
    agent = "registry.invalid/task@sha256:" + ("3" if second and mode == "agent-drift" else "1") * 64
    verifier = "registry.invalid/verifier@sha256:" + ("4" if second and mode == "verifier-drift" else "2") * 64
    if mode == "supercool":
        agent = "registry.invalid/task:supercool"
    out({"name": args[2], "spec": {
        "environment": {
            "agent": {"image_ref": agent},
            "verifier": None if mode == "supercool" else {"image_ref": verifier},
            "agent_preinstalled": mode == "supercool" or (second and mode == "layer-fallback"),
        },
        "sandbox_policy": {"image_reference_mode": "tag" if mode in {"supercool", "pinned"} else "digest"},
    }})
elif args[:4] == ["jobs", "results", "download", "agent-eval-results"]:
    with tarfile.open(option("--output-file"), "w:gz") as tar:
        entry = tarfile.TarInfo("agent-eval-results/trials.jsonl")
        data = b"{}\\n"
        entry.size = len(data)
        tar.addfile(entry, io.BytesIO(data))
elif args[:3] == ["evals", "runs", "results"]:
    second = args[3] == "run-0000000000000002"
    scores = [] if second and mode == "missing-score" else [
        {"name": "harbor_reward.reward", "mean": 0, "count": 1, "nan_count": 0}
    ]
    out({"run": {"phase": "completed", "result_state": "ready"}, "tasks": [{"result": {
        "summary": {"trial_count": 1, "error_count": int(second and mode == "incomplete")},
        "scores": {"scores": scores},
    }}]})
else:
    raise AssertionError(args)
""",
        encoding="utf-8",
    )
    stub.chmod(0o700)
    return {
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "CALLS": str(tmp_path / "calls.jsonl"),
        "SUBMISSIONS": str(tmp_path / "submissions.jsonl"),
        "RUN_DIR": str(tmp_path / "pair"),
        "CONTEXT": "demo-context",
        "WORKSPACE": "demo",
        "MODEL_REF": "demo/codex",
        "SANDBOX_PROFILE": "harbor-standard",
        "INFERENCE_KEY_SECRET": "inference-key",
        "TASKSET_REF": _TASKSET,
        "SKILL_REF": _SKILL,
    }


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", _LAUNCHER], env=env, capture_output=True, text=True, timeout=20, check=False)


def _read_lines(path: str) -> list:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("mode", ["ev", "pinned"])
def test_pinned_pair_saves_two_scored_bundles(launcher_env: dict[str, str], mode: str) -> None:
    result = _run(launcher_env | {"STUB_MODE": mode})
    assert result.returncode == 0, result.stderr
    calls = _read_lines(launcher_env["CALLS"])
    assert not any(call[:2] in (["agents", "skills"], ["evals", "tasksets"]) for call in calls)
    baseline, treatment = _read_lines(launcher_env["SUBMISSIONS"])
    assert baseline["taskset"] == treatment["taskset"] == _TASKSET
    assert baseline["spec"]["idempotency_key"] != treatment["spec"]["idempotency_key"]
    for submission, skills in ((baseline, []), (treatment, [_SKILL])):
        spec = submission["spec"]
        assert spec["execution"]["sandbox"] == {"profile": "harbor-standard"}
        assert spec["spec"]["target"]["env_secrets"] == {"OPENAI_API_KEY": "inference-key"}
        assert spec["spec"]["target"]["skills"] == skills

    run_dir = Path(launcher_env["RUN_DIR"])
    pair = json.loads((run_dir / "comparison.json").read_text(encoding="utf-8"))
    assert (pair["taskset_ref"], pair["skill_ref"], pair["workspace"], pair["context"]) == (
        _TASKSET,
        _SKILL,
        "demo",
        "demo-context",
    )
    for arm, run_id in (("baseline", _BASELINE_ID), ("with-skill", _TREATMENT_ID)):
        assert pair["arms"][arm] == {
            "run_id": run_id,
            "job_name": f"{run_id}-task-0",
            "reward": 0,
            "agent_image_ref": "registry.invalid/task@sha256:" + "1" * 64,
            "verifier_image_ref": "registry.invalid/verifier@sha256:" + "2" * 64,
            "bundle": f"{arm}/agent-eval-results",
        }
        assert (run_dir / pair["arms"][arm]["bundle"] / "trials.jsonl").is_file()


def test_single_override_fails_before_platform_calls(launcher_env: dict[str, str]) -> None:
    launcher_env.pop("SKILL_REF")
    result = _run(launcher_env)
    assert result.returncode != 0
    assert "TASKSET_REF" in result.stderr and "SKILL_REF" in result.stderr
    assert not Path(launcher_env["CALLS"]).exists()


def test_ambiguous_submission_reuses_saved_spec_and_key(launcher_env: dict[str, str]) -> None:
    assert _run(launcher_env | {"STUB_MODE": "ambiguous"}).returncode != 0
    result = _run(launcher_env | {"STUB_MODE": "replay"})
    assert result.returncode == 0, result.stderr
    baseline, first_attempt, replay = _read_lines(launcher_env["SUBMISSIONS"])
    assert baseline["spec"]["spec"]["target"]["skills"] == []
    assert first_attempt == replay
    assert first_attempt["spec"]["idempotency_key"]


@pytest.mark.parametrize("mode", ["agent-drift", "verifier-drift", "layer-fallback", "incomplete", "missing-score"])
def test_unusable_pair_keeps_the_task_bundles(launcher_env: dict[str, str], mode: str) -> None:
    result = _run(launcher_env | {"STUB_MODE": mode})
    assert result.returncode != 0
    run_dir = Path(launcher_env["RUN_DIR"])
    assert not (run_dir / "comparison.json").exists()
    for arm in ("baseline", "with-skill"):
        assert (run_dir / arm / "agent-eval-results" / "trials.jsonl").is_file(), result.stderr


def test_default_supercool_pins_publication_and_accepts_shared_verifier(launcher_env: dict[str, str]) -> None:
    launcher_env.pop("TASKSET_REF")
    launcher_env.pop("SKILL_REF")
    result = _run(launcher_env | {"STUB_MODE": "supercool"})
    assert result.returncode == 0, result.stderr
    baseline, treatment = _read_lines(launcher_env["SUBMISSIONS"])
    assert baseline["spec"]["spec"]["target"]["skills"] == []
    assert treatment["spec"]["spec"]["target"]["skills"] == ["demo/supercool-guidelines#" + "d" * 64]
    pair = json.loads((Path(launcher_env["RUN_DIR"]) / "comparison.json").read_text(encoding="utf-8"))
    assert pair["arms"]["baseline"]["agent_image_ref"] == "registry.invalid/task:supercool"
