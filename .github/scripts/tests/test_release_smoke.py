# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise failure evidence and cleanup without installing or launching Helix."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("release_smoke", SCRIPTS / "release_smoke.py")
smoke_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke_module)


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"error": {"message": "backend unavailable"}},
        {"choices": []},
        {"choices": [{"message": {"content": " "}}]},
        {"choices": [{"message": {"content": "text"}, "finish_reason": "error"}]},
        {"choices": [{"message": {"content": "hello"}}, {"error": "bad"}]},
        "502 Could not connect to agent",
    ],
)
def test_protocol_errors_and_empty_responses_fail(response):
    with pytest.raises(ValueError):
        smoke_module.require_completion(response)


@pytest.mark.parametrize("content", ["hi", "ANALYSIS\nVERDICT: benign", "A changed sample response"])
def test_success_does_not_depend_on_sample_wording(content):
    smoke_module.require_completion({"choices": [{"message": {"content": content}}]})


@pytest.fixture
def smoke(tmp_path, monkeypatch):
    for name, value in {
        "SMOKE_ROOT": str(tmp_path / "private"),
        "SMOKE_RESULTS": str(tmp_path / "results"),
        "RELEASE_VERSION": "0.8.0",
        "KIND_CLUSTER_NAME": "release-smoke",
        "NVIDIA_INFERENCE_KEY": "private-test-key",
        "NVIDIA_INFERENCE_URL": "https://private.test/v1",
    }.items():
        monkeypatch.setenv(name, value)
    return smoke_module.Smoke()


def test_intentional_command_failure_records_stage_and_diagnostics(smoke):
    smoke.stage("agent-invoke")
    with pytest.raises(RuntimeError, match="exit 7"):
        smoke.command([sys.executable, "-c", "print('intentional failure'); raise SystemExit(7)"])
    assert json.loads(smoke.state_file.read_text())["stage"] == "agent-invoke"
    assert "intentional failure" in (smoke.results / "agent-invoke.log").read_text()


def test_timeout_is_a_failure(smoke):
    with patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("request", 120)):
        with pytest.raises(subprocess.TimeoutExpired):
            smoke.command(["request"], timeout=120)


def test_cleanup_runs_when_diagnostics_fail(smoke):
    smoke.stage("agent-invoke")
    commands = []

    def command(args, **kwargs):
        commands.append(args)
        if args[0] == "bash":
            raise subprocess.TimeoutExpired(args, 120)
        return ""

    with patch.object(smoke, "command", side_effect=command):
        smoke.finalize("kubernetes")
    assert ["kind", "delete", "cluster", "--name", "release-smoke"] in commands
    report = json.loads(smoke.state_file.read_text())
    assert report["release"] == "0.8.0"
    assert report["stage"] == "agent-invoke"
    assert report["cleanup_errors"]


def test_pod_logs_are_collected_before_deleting_agent(smoke):
    smoke.stage("agent-invoke")
    Path(smoke.nemo).parent.mkdir(parents=True, exist_ok=True)
    Path(smoke.nemo).touch()
    with patch.object(smoke, "command") as command:
        smoke.finalize("kubernetes")
    commands = [call.args[0] for call in command.call_args_list]
    assert commands[0][0] == "bash"
    assert commands[1][1:4] == ["agents", "deployments", "delete"]
    assert commands[-1] == ["kind", "delete", "cluster", "--name", "release-smoke"]


def test_json_response_does_not_include_stderr_warnings(smoke):
    response = smoke.command(
        [sys.executable, "-c", "import sys; print('warning', file=sys.stderr); print('{}')"], capture=True
    )
    assert json.loads(response) == {}
    assert "warning" in (smoke.results / "setup.log").read_text()


def test_install_is_version_pinned_and_separate_from_source(smoke):
    smoke.env.update({"WHEEL_VERSION": "0.8.0", "WHEEL_INDEX": "https://pypi.org/simple"})
    with patch.object(smoke, "command") as command:
        smoke.install("all")
        assert "nemo-helix[all]==0.8.0" in command.call_args_list[0].args[0]
        assert command.call_args_list[1].args[0][0] == smoke.python
    with patch.object(smoke, "command") as command:
        smoke.install("all,nemo-agents-plugin-deepagents")
        assert "nemo-helix[all,nemo-agents-plugin-deepagents]==0.8.0" in command.call_args_list[0].args[0]
