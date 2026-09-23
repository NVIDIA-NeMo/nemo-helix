# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end validation of sandboxed Gym execution.

Two levels, because they fail for different reasons and only one of them needs a cluster.

:func:`test_a_sandboxed_run_provisions_a_host_and_returns_attributed_trials` runs everywhere. It
drives the real `SessionBackedGymRunner` through a real `SandboxedGymOrchestrator`, which starts a
real episode broker on a real socket, and substitutes only the *job host provider* -- the one piece
that would otherwise call OpenSandbox. The fake provider serves a genuine HTTP endpoint, so the
rollout request crosses a real network boundary. Everything between the target and the trials is
production code.

:func:`test_a_real_opensandbox_host_serves_rollouts` is the same path with nothing substituted, and
needs a cluster. It is opt-in and skipped by default; it is what proves the OpenSandbox calls
themselves, which no amount of local wiring can.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
from nemo_evaluator.config import EvaluatorConfig
from nemo_evaluator.jobs.agent_spec import GymRunnerTarget
from nemo_evaluator.jobs.gym_sandbox import SessionBackedGymRunner, resolve_sandbox_plan
from nemo_evaluator_sdk.agent_eval.runtimes.gym import discover_gym_tasks
from nemo_evaluator_sdk.agent_eval.runtimes.gym.records import NG_ROLLOUT_INDEX, NG_TASK_INDEX
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig
from sandboxed_gym.runtime.gym_host_runtime import GYM_GLOBAL_CONFIG_ENV_KEY

pytestmark = pytest.mark.integration

RUNTIME_IMAGE = "registry.example.com/nhx-gym-runtime:test"


def _tasks(tmp_path: Path) -> list:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps({"responses_create_params": {"input": "What is 2 + 2?"}})
        + "\n"
        + json.dumps({"responses_create_params": {"input": "Capital of France?"}})
        + "\n",
        encoding="utf-8",
    )
    return discover_gym_tasks(dataset)


def _target() -> GymRunnerTarget:
    return GymRunnerTarget(
        agent="simple_agent",
        agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
        resources_server="mcqa",
    )


def _config(**overrides: Any) -> EvaluatorConfig:
    fields: dict[str, Any] = {
        "sandboxed_gym_default": True,
        "sandbox_cluster_capable": True,
        "sandbox_runtime_image": RUNTIME_IMAGE,
        "sandbox_job_storage_pvc_claim": "job-storage",
        # Episodes are the nested tier, which this evaluation path does not use -- only SWE-style
        # environments create them. The in-memory backend keeps the broker real without needing a
        # cluster to provision episodes nothing asks for.
        "sandbox_episode_backend": "memory",
        "sandbox_allow_insecure_memory_backend": True,
        "sandbox_policy_base_urls": ("https://integrate.api.nvidia.com/v1",),
    }
    fields.update(overrides)
    return EvaluatorConfig(**fields)


def _plan(**overrides: Any):
    """The plan the compiler resolves service-side; the job only ever reads one."""
    plan = resolve_sandbox_plan(_config(**overrides), _target())
    assert plan is not None
    return plan


# --------------------------------------------------------------------------------------------
# Level 1: everything but the OpenSandbox calls, no cluster required
# --------------------------------------------------------------------------------------------


class _StubGymHostHandler(BaseHTTPRequestHandler):
    """A Gym host as far as the rollout client can tell: `/health` and `/rollouts/run`."""

    #: Set by the fixture; the spec the provider was asked to create a host for.
    received_specs: list[Any] = []

    def log_message(self, *args: Any) -> None:  # keep pytest output readable
        return

    def do_GET(self) -> None:
        body = json.dumps({"status": "ready"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode())
        # Echo the caller's own index back, which is what a real Gym host does with a stamped row.
        results = [
            {
                NG_TASK_INDEX: example[NG_TASK_INDEX],
                NG_ROLLOUT_INDEX: 0,
                "reward": float(example[NG_TASK_INDEX]),
                "response": f"answer-{example[NG_TASK_INDEX]}",
            }
            for example in payload["examples"]
        ]
        body = json.dumps({"results": results}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _StubHostProvider:
    """A job-host provider that serves HTTP locally instead of calling OpenSandbox.

    Deliberately the *only* substitution: the broker below it is real, the orchestrator that wires
    them is real, and the rollout request really crosses a socket.
    """

    name = "stub"

    def __init__(self) -> None:
        self.created: list[Any] = []
        self.destroyed: list[str] = []
        self._server: HTTPServer | None = None

    async def create_host(self, spec: Any) -> Any:
        from sandboxed_gym.host.models import GymHostHandle

        self.created.append(spec)
        self._server = HTTPServer(("127.0.0.1", 0), _StubGymHostHandler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{self._server.server_address[1]}"
        return GymHostHandle(
            host_id="stub-host",
            health_url=f"{base}/health",
            rollout_url=f"{base}/rollouts/run",
            headers={},
            provider=self,
        )

    async def wait_ready(self, handle: Any, timeout_s: float) -> None:
        return

    async def destroy_host(self, handle: Any) -> None:
        self.destroyed.append(handle.host_id)
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


@pytest.fixture
def stub_provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[_StubHostProvider]:
    provider = _StubHostProvider()
    # Patched where the orchestrator looks it up, so the orchestrator itself stays untouched.
    monkeypatch.setattr("sandboxed_gym.orchestrator.get_host_provider", lambda *a, **k: provider)
    yield provider


async def test_a_sandboxed_run_provisions_a_host_and_returns_attributed_trials(
    stub_provider: _StubHostProvider, tmp_path: Path
) -> None:
    """The whole path: target -> serve config -> broker + host -> rollouts -> attributed trials."""
    tasks = _tasks(tmp_path)
    runner = SessionBackedGymRunner(target=_target(), plan=_plan(), job_id="eval-job-1")

    trials = await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    rewards = {trial.task_id: trial.metadata["reward"] for trial in trials}
    assert rewards[tasks[0].id] == 0.0
    assert rewards[tasks[1].id] == 1.0, "each trial must carry its own task's reward, not a neighbour's"


async def test_the_host_is_created_from_the_deployment_config_and_the_targets_selection(
    stub_provider: _StubHostProvider, tmp_path: Path
) -> None:
    # Proves the two halves of the serve config actually reach the provider, rather than the run
    # succeeding on defaults.
    runner = SessionBackedGymRunner(target=_target(), plan=_plan(), job_id="eval-job-2")

    await runner.run_tasks(_tasks(tmp_path), AgentEvalRunConfig(work_dir=tmp_path))

    (spec,) = stub_provider.created
    assert spec.runtime_image == RUNTIME_IMAGE
    assert spec.job_id == "eval-job-2"

    # The broker is real, listening, and reachable from the host: its URL and per-run token are in
    # the host's bootstrap environment. Without this the run could pass with no broker at all, which
    # is the whole trusted-side mechanism.
    from sandboxed_gym.wire import BROKER_TOKEN_ENV, BROKER_URL_ENV

    assert spec.bootstrap_env[BROKER_URL_ENV].startswith("http")
    assert spec.bootstrap_env[BROKER_TOKEN_ENV], "the host must receive a broker token"

    # ...and the target's environment selection reached Gym as config paths it will resolve itself.
    global_config = json.loads(spec.bootstrap_env[GYM_GLOBAL_CONFIG_ENV_KEY])
    assert "resources_servers/mcqa/configs/mcqa.yaml" in global_config["config_paths"]


async def test_the_host_is_destroyed_when_the_run_finishes(stub_provider: _StubHostProvider, tmp_path: Path) -> None:
    runner = SessionBackedGymRunner(target=_target(), plan=_plan(), job_id="eval-job-3")

    await runner.run_tasks(_tasks(tmp_path), AgentEvalRunConfig(work_dir=tmp_path))

    assert stub_provider.destroyed == ["stub-host"], "a host outliving its run is a leaked pod holding a PVC"


async def test_the_host_is_destroyed_when_the_run_fails(
    stub_provider: _StubHostProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case teardown-in-`finally` exists for; a failed run must not strand the host."""
    runner = SessionBackedGymRunner(target=_target(), plan=_plan(), job_id="eval-job-4")
    monkeypatch.setattr(
        "nemo_evaluator_sdk.agent_eval.runtimes.gym.sandboxed.SandboxedGymAgentTaskRunner.run_tasks",
        _raise_boom,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await runner.run_tasks(_tasks(tmp_path), AgentEvalRunConfig(work_dir=tmp_path))

    assert stub_provider.destroyed == ["stub-host"]


async def _raise_boom(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("boom")


# --------------------------------------------------------------------------------------------
# Level 2: the real thing, which needs a cluster
# --------------------------------------------------------------------------------------------

_LIVE_ENV = ("RUN_SANDBOXED_GYM_LIVE", "OPENSANDBOX_DOMAIN", "OPENSANDBOX_API_KEY", "NHX_GYM_RUNTIME_IMAGE")


@pytest.mark.skipif(
    any(os.environ.get(name) is None for name in _LIVE_ENV),
    reason=f"needs a real OpenSandbox cluster; set {', '.join(_LIVE_ENV)} to run",
)
async def test_a_real_opensandbox_host_serves_rollouts(tmp_path: Path) -> None:
    """Provision a real sandboxed Gym host and collect from it.

    This is the only test that exercises the OpenSandbox calls, the PVC mounts, the egress policy
    and the readiness probe. The level-1 tests above deliberately cannot: they replace the provider
    that makes those calls.

    Requires a `NHX_GYM_RUNTIME_IMAGE` carrying NeMo-Gym and the host runtime, and a PVC holding an
    environment the target selects. Run with::

        RUN_SANDBOXED_GYM_LIVE=1 OPENSANDBOX_DOMAIN=... OPENSANDBOX_API_KEY=... \\
        NHX_GYM_RUNTIME_IMAGE=... NHX_GYM_PVC_CLAIM=... \\
        uv run pytest plugins/nemo-evaluator/tests/integration/test_sandboxed_gym_execution.py -k real -v
    """
    config = _config(
        sandbox_runtime_image=os.environ["NHX_GYM_RUNTIME_IMAGE"],
        sandbox_job_storage_pvc_claim=os.environ.get("NHX_GYM_PVC_CLAIM", "job-storage"),
    )
    runner = SessionBackedGymRunner(
        target=_target(), plan=resolve_sandbox_plan(config, _target()), job_id="eval-live-1"
    )

    trials = await runner.run_tasks(_tasks(tmp_path), AgentEvalRunConfig(work_dir=tmp_path))

    assert trials, "a live host returned no trials"
    assert {trial.task_id for trial in trials} == {task.id for task in _tasks(tmp_path)}
