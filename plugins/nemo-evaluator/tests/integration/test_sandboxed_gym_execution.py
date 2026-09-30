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

import asyncio
import json
import os
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


def _target(**overrides: Any) -> GymRunnerTarget:
    return GymRunnerTarget(
        agent="simple_agent",
        agent_config="responses_api_agents/simple_agent/configs/simple_agent.yaml",
        resources_server="mcqa",
        **overrides,
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
    received_payloads: list[dict[str, Any]] = []
    reply: tuple[int, dict[str, Any]] | None = None
    #: Held per POST, so concurrent POSTs overlap long enough to be counted.
    delay_s = 0.0
    #: When set, each POST blocks until this event is set, holding the request in flight.
    gate: threading.Event | None = None
    post_received = threading.Event()
    in_flight = 0
    peak_in_flight = 0
    _lock = threading.Lock()

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
        cls = type(self)
        with cls._lock:
            cls.received_payloads.append(payload)
            cls.in_flight += 1
            cls.peak_in_flight = max(cls.peak_in_flight, cls.in_flight)
        cls.post_received.set()
        if cls.gate is not None:
            cls.gate.wait(timeout=30)
        time.sleep(cls.delay_s)
        with cls._lock:
            cls.in_flight -= 1
        if cls.reply is not None:
            status, reply = cls.reply
            self._send(status, reply)
            return
        # Echo the caller's own indices back, which is what a real Gym host does with a stamped row.
        results = [
            {
                NG_TASK_INDEX: example[NG_TASK_INDEX],
                NG_ROLLOUT_INDEX: example[NG_ROLLOUT_INDEX],
                "reward": float(example[NG_TASK_INDEX]),
                "response": f"answer-{example[NG_TASK_INDEX]}",
            }
            for example in payload["examples"]
        ]
        self._send(200, {"results": results})

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
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
        self._server: ThreadingHTTPServer | None = None

    async def create_host(self, spec: Any) -> Any:
        from sandboxed_gym.host.models import GymHostHandle

        self.created.append(spec)
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _StubGymHostHandler)
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
    _StubGymHostHandler.received_payloads = []
    _StubGymHostHandler.reply = None
    _StubGymHostHandler.delay_s = 0.0
    _StubGymHostHandler.gate = None
    _StubGymHostHandler.post_received = threading.Event()
    _StubGymHostHandler.in_flight = _StubGymHostHandler.peak_in_flight = 0
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


async def test_repeats_reach_the_host_in_chunks_sized_by_concurrency(
    stub_provider: _StubHostProvider, tmp_path: Path
) -> None:
    """The run is split across POSTs, so no one request carries, or waits on, all of it."""
    tasks = _tasks(tmp_path)
    runner = SessionBackedGymRunner(
        target=_target(num_repeats=5, concurrency=16), plan=_plan(), job_id="eval-job-repeats"
    )

    trials = await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    payloads = _StubGymHostHandler.received_payloads
    assert sorted(len(payload["examples"]) for payload in payloads) == [2] * 5, "concurrency 16 is 8 chunks of 2"
    assert all(list(payload) == ["examples"] for payload in payloads)
    assert len(trials) == 10
    for task in tasks:
        assert sorted(t.metadata[NG_ROLLOUT_INDEX] for t in trials if t.task_id == task.id) == [0, 1, 2, 3, 4]
    info = runner.runner_info().config
    assert (info["num_repeats"], info["rollout_chunk_size"], info["rollout_max_in_flight"]) == (5, 2, 8)
    assert "timeout_s" not in info, "the SDK runner's own POST timeout does not apply to session collection"


async def test_an_explicit_concurrency_bounds_the_rollouts_in_flight(
    stub_provider: _StubHostProvider, tmp_path: Path
) -> None:
    _StubGymHostHandler.delay_s = 0.1
    runner = SessionBackedGymRunner(
        target=_target(num_repeats=3, concurrency=2), plan=_plan(), job_id="eval-job-concurrency"
    )

    trials = await runner.run_tasks(_tasks(tmp_path), AgentEvalRunConfig(work_dir=tmp_path))

    assert len(trials) == 6
    assert [len(payload["examples"]) for payload in _StubGymHostHandler.received_payloads] == [1] * 6
    assert _StubGymHostHandler.peak_in_flight == 2
    info = runner.runner_info().config
    assert (info["rollout_chunk_size"], info["rollout_max_in_flight"]) == (1, 2)


async def test_a_host_bootstrap_failure_fails_the_run_with_the_hosts_output(
    stub_provider: _StubHostProvider, tmp_path: Path
) -> None:
    _StubGymHostHandler.reply = (
        503,
        {
            "error": {
                "code": "bootstrap_failed",
                "message": "Gym host failed to start",
                "host_output_tail": ["ModuleNotFoundError: No module named 'mcqa'"],
            }
        },
    )
    runner = SessionBackedGymRunner(target=_target(), plan=_plan(), job_id="eval-job-bootstrap")

    with pytest.raises(RuntimeError) as excinfo:
        await runner.run_tasks(_tasks(tmp_path), AgentEvalRunConfig(work_dir=tmp_path))

    message = str(excinfo.value)
    assert "bootstrap_failed" in message
    assert "--- gym host output (1 lines) ---\nModuleNotFoundError: No module named 'mcqa'" in message
    assert len(_StubGymHostHandler.received_payloads) == 2, "one POST per chunk: a host-reported failure is not retried"
    assert stub_provider.destroyed == ["stub-host"]


async def test_a_rollout_error_reported_on_a_200_fails_the_run_with_the_hosts_output(
    stub_provider: _StubHostProvider, tmp_path: Path
) -> None:
    _StubGymHostHandler.reply = (
        200,
        {"error": {"code": "internal", "message": "KeyError: 'agent_ref'", "host_output_tail": ["Traceback ..."]}},
    )
    runner = SessionBackedGymRunner(target=_target(), plan=_plan(), job_id="eval-job-rollout-error")

    with pytest.raises(RuntimeError) as excinfo:
        await runner.run_tasks(_tasks(tmp_path), AgentEvalRunConfig(work_dir=tmp_path))

    message = str(excinfo.value)
    assert "KeyError: 'agent_ref'" in message
    assert "--- gym host output (1 lines) ---\nTraceback ..." in message


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


async def test_cancelling_a_run_stops_it_sending_the_rollouts_still_queued(
    stub_provider: _StubHostProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled job must not go on posting its remaining chunks, and retrying them, to a host
    its own teardown has already destroyed."""
    from sandboxed_gym.orchestrator import SandboxedGymSession

    attempts: list[list[dict[str, Any]]] = []
    first_attempt_returned = threading.Event()
    post_chunk = SandboxedGymSession._post_chunk

    def _counted_post_chunk(self: SandboxedGymSession, chunk: list[dict[str, Any]]) -> list[Any]:
        attempts.append(chunk)
        try:
            return post_chunk(self, chunk)
        finally:
            first_attempt_returned.set()

    monkeypatch.setattr(SandboxedGymSession, "_post_chunk", _counted_post_chunk)
    release = _StubGymHostHandler.gate = threading.Event()
    # Six single-example chunks, one in flight at a time: five are still queued at cancellation.
    runner = SessionBackedGymRunner(
        target=_target(num_repeats=3, concurrency=1), plan=_plan(), job_id="eval-job-cancelled"
    )
    run = asyncio.create_task(runner.run_tasks(_tasks(tmp_path), AgentEvalRunConfig(work_dir=tmp_path)))
    try:
        assert await asyncio.to_thread(_StubGymHostHandler.post_received.wait, 10), "the first chunk never arrived"

        run.cancel()
        done, _ = await asyncio.wait({run}, timeout=10)

        assert run in done, "cancellation must not wait for the chunk in flight to come back"
        assert run.cancelled()
        assert stub_provider.destroyed == ["stub-host"]
    finally:
        release.set()
    assert await asyncio.to_thread(first_attempt_returned.wait, 10)
    # A leaked dispatcher sends the next chunk within milliseconds of the first returning.
    await asyncio.sleep(0.5)
    assert len(attempts) == 1, f"{len(attempts) - 1} chunk POST(s) attempted after the run was cancelled"


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
