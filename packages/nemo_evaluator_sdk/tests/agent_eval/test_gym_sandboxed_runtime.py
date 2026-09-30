# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for collecting Gym rollouts from a sandboxed host rather than the local ``gym`` CLI.

The host is faked at the HTTP boundary; everything on this side of it is the real path -- real
tasks from :func:`discover_gym_tasks`, the real dataset materialization, and the real rollout
parser. That is deliberate: the runner's whole claim is that only the *collection* step differs
from the CLI runner, so the test exercises the parts that are supposed to be shared.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from nemo_evaluator_sdk.agent_eval.runtimes.gym import discover_gym_tasks, sandboxed
from nemo_evaluator_sdk.agent_eval.runtimes.gym.records import NG_ROLLOUT_INDEX, NG_TASK_INDEX
from nemo_evaluator_sdk.agent_eval.runtimes.gym.sandboxed import (
    MODEL_CALLS_RESULT_KEY,
    PROXY_AUTH_HEADER,
    SandboxedGymAgentTaskRunner,
    SandboxedGymRuntimeConfig,
    _stamp_rollout_indices,
)
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig
from nemo_evaluator_sdk.values.evidence import EVIDENCE_FORMAT_OTLP, EVIDENCE_TRACE

ROLLOUT_URL = "http://gym-host.example/rollouts/run"


@pytest.fixture
def tasks(tmp_path: Path) -> list:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps({"responses_create_params": {"input": "What is 2 + 2?"}})
        + "\n"
        + json.dumps({"responses_create_params": {"input": "Capital of France?"}})
        + "\n",
        encoding="utf-8",
    )
    return discover_gym_tasks(dataset)


class _FakeHost:
    """Records what was posted and answers with rollout records keyed by the caller's own index."""

    def __init__(
        self,
        *,
        status: int = 200,
        body: Any = None,
        content: bytes | None = None,
        rewards: dict[int, float] | None = None,
        model_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self.posted: list[dict[str, Any]] = []
        self._status = status
        self._body = body
        self._content = content
        self._rewards = rewards
        self._model_calls = model_calls

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            self.posted = json.loads(request.content.decode())["examples"]
            if self._content is not None:
                return httpx.Response(self._status, content=self._content)
            if self._body is not None or self._status >= 400:
                return httpx.Response(self._status, json=self._body if self._body is not None else {"error": "boom"})
            rewards = self._rewards if self._rewards is not None else {}
            results = [
                {
                    NG_TASK_INDEX: example[NG_TASK_INDEX],
                    # Echoed, never invented: Gym's `run_examples` copies the row's identity onto
                    # the result and assigns nothing of its own. A fake that supplies an index the
                    # caller failed to send would hide exactly the bug this mirrors.
                    NG_ROLLOUT_INDEX: example.get(NG_ROLLOUT_INDEX),
                    "reward": rewards.get(example[NG_TASK_INDEX], 1.0),
                    "response": f"answer-{example[NG_TASK_INDEX]}",
                    **({MODEL_CALLS_RESULT_KEY: self._model_calls} if self._model_calls else {}),
                }
                for example in self.posted
            ]
            return httpx.Response(200, json={"results": results})

        return httpx.MockTransport(handle)


def bind_http_transport(monkeypatch: pytest.MonkeyPatch, transport: httpx.MockTransport) -> None:
    """Point ``httpx.AsyncClient`` at ``transport``. Typed kwargs so ty can check the call."""
    original = httpx.AsyncClient

    def bound(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", bound)


def runner_against(host: _FakeHost, monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> SandboxedGymAgentTaskRunner:
    """A runner whose HTTP client is bound to ``host``."""
    bind_http_transport(monkeypatch, host.transport())
    return SandboxedGymAgentTaskRunner(config=SandboxedGymRuntimeConfig(rollout_url=ROLLOUT_URL, **overrides))


async def test_rollouts_from_the_host_are_attributed_to_the_right_tasks(tasks, tmp_path, monkeypatch) -> None:
    # The property the whole design turns on: identity assigned here survives the hop and comes
    # back joinable. Distinct rewards make a swapped attribution visible rather than plausible.
    host = _FakeHost(rewards={0: 1.0, 1: 0.0})
    runner = runner_against(host, monkeypatch)

    trials = await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    rewards = {trial.task_id: trial.metadata["reward"] for trial in trials}
    assert rewards[tasks[0].id] == 1.0
    assert rewards[tasks[1].id] == 0.0


async def test_the_examples_posted_carry_the_index_we_stamped(tasks, tmp_path, monkeypatch) -> None:
    # The host cannot attribute anything we did not stamp, so this is the precondition for the
    # test above rather than a restatement of it.
    host = _FakeHost()
    runner = runner_against(host, monkeypatch)

    await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    assert [example[NG_TASK_INDEX] for example in host.posted] == [0, 1]
    assert all("responses_create_params" in example for example in host.posted)


async def test_the_auth_token_is_sent_as_the_proxy_header(tasks, tmp_path, monkeypatch) -> None:
    host = _FakeHost()
    runner = runner_against(host, monkeypatch, auth_token="tok-123", headers={"X-Extra": "kept"})

    await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    sent = host.requests[0].headers
    assert sent[PROXY_AUTH_HEADER] == "tok-123"
    assert sent["X-Extra"] == "kept", "caller-supplied headers must survive alongside the token"


async def test_no_auth_header_is_sent_when_no_token_is_configured(tasks, tmp_path, monkeypatch) -> None:
    # An unauthenticated proxy is a valid configuration; sending an empty token would fail closed
    # against it for no reason.
    host = _FakeHost()
    runner = runner_against(host, monkeypatch)

    await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    assert PROXY_AUTH_HEADER not in host.requests[0].headers


async def test_an_http_error_names_the_host_and_carries_its_body(tasks, tmp_path, monkeypatch) -> None:
    """The body says which example or server failed; the status code alone does not."""
    host = _FakeHost(status=502, body={"error": {"message": "resources_server crashed"}})
    runner = runner_against(host, monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    message = str(excinfo.value)
    assert "502" in message
    assert ROLLOUT_URL in message
    assert "resources_server crashed" in message


async def test_an_error_carried_on_a_200_is_still_a_failure(tasks, tmp_path, monkeypatch) -> None:
    """The host commits its 200 before the batch finishes, so late failures ride the body.

    It answers early on purpose: the sandbox proxy gives up on a request whose first byte has not
    arrived, and a batch outlasts that. Reading only the status would call a deadline or a crashed
    environment a success and then blame the run for having no results.
    """
    host = _FakeHost(body={"error": {"code": "deadline_exceeded", "message": "abandoned after 1800s"}})
    runner = runner_against(host, monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    message = str(excinfo.value)
    assert "deadline_exceeded" in message
    assert "abandoned after 1800s" in message
    assert ROLLOUT_URL in message


async def test_a_heartbeat_only_200_names_the_host_as_the_failure(tasks, tmp_path, monkeypatch) -> None:
    """OOMKill after the host committed 200 leaves heartbeats and no envelope.

    The orchestrator classifies that as the sandbox dying. This runner is a second client of the
    same host; it must not surface a JSON decode error that looks like a bug in the evaluator.
    """
    host = _FakeHost(content=b"     ")
    runner = runner_against(host, monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    message = str(excinfo.value)
    assert ROLLOUT_URL in message
    assert "JSON" not in message
    assert "error" in message.lower() or "results" in message.lower()


async def test_a_body_free_200_names_both_causes(tasks, tmp_path, monkeypatch) -> None:
    """Zero bytes, not even heartbeat padding.

    A host that died before writing and a silent host whose request the proxy truncated arrive
    identically, because without a Content-Length the body ends at the close. The runner must
    offer both and the check for each, not pick one.
    """
    host = _FakeHost(content=b"")
    runner = runner_against(host, monkeypatch)
    # The runner times its own POST; a MockTransport answers instantly, so stand in for the wait.
    monkeypatch.setattr(
        "nemo_evaluator_sdk.agent_eval.runtimes.gym.sandboxed.time.monotonic",
        _clock_advancing_by(180.0),
    )

    with pytest.raises(RuntimeError) as excinfo:
        await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    message = str(excinfo.value)
    assert ROLLOUT_URL in message
    assert "OOMKill" in message
    assert "predates the rollout heartbeat" in message


async def test_a_body_free_200_that_arrived_fast_blames_the_sandbox_alone(tasks, tmp_path, monkeypatch) -> None:
    host = _FakeHost(content=b"")
    runner = runner_against(host, monkeypatch)
    monkeypatch.setattr(
        "nemo_evaluator_sdk.agent_eval.runtimes.gym.sandboxed.time.monotonic",
        _clock_advancing_by(2.0),
    )

    with pytest.raises(RuntimeError) as excinfo:
        await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    message = str(excinfo.value)
    assert "OOMKilled" in message
    assert "heartbeat" not in message


def _clock_advancing_by(delta: float):
    """A monotonic clock whose second reading is ``delta`` later than its first."""
    readings = iter((0.0, delta))

    def _clock() -> float:
        return next(readings, delta)

    return _clock


async def test_a_non_json_200_names_the_host_rather_than_the_decoder(tasks, tmp_path, monkeypatch) -> None:
    """A non-empty body that is not JSON is a broken response contract, not an evaluator bug.

    `json.JSONDecodeError` is a `ValueError`, so leaving it unwrapped escapes every caller that
    guards this runner for `RuntimeError`.
    """
    host = _FakeHost(content=b"   <html>502 Bad Gateway</html>")
    runner = runner_against(host, monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    message = str(excinfo.value)
    assert ROLLOUT_URL in message
    assert "502 Bad Gateway" in message  # the undecodable body itself, so the reader can place it


async def test_a_non_utf8_200_names_the_host_rather_than_the_decoder(tasks, tmp_path, monkeypatch) -> None:
    # Decoded strictly: errors="replace" would hand the parser U+FFFD where the host wrote data,
    # and a corrupt batch would score rather than fail.
    host = _FakeHost(content=b'{"results": [\xff\xfe]}')
    runner = runner_against(host, monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    assert ROLLOUT_URL in str(excinfo.value)


async def test_a_response_without_a_results_list_is_refused(tasks, tmp_path, monkeypatch) -> None:
    # Reaching the parser with nothing collected would surface as "no rollouts", blaming the run
    # for what is a malformed reply.
    host = _FakeHost(body={"unexpected": True})
    runner = runner_against(host, monkeypatch)

    with pytest.raises(RuntimeError, match="no `results` list"):
        await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))


async def test_a_task_the_host_never_answered_fails_the_run(tasks, tmp_path, monkeypatch) -> None:
    """Partial collection must not read as a completed evaluation with fewer tasks."""

    def handle(request: httpx.Request) -> httpx.Response:
        example = json.loads(request.content.decode())["examples"][0]
        return httpx.Response(
            200,
            json={"results": [{NG_TASK_INDEX: example[NG_TASK_INDEX], NG_ROLLOUT_INDEX: 0, "reward": 1.0}]},
        )

    bind_http_transport(monkeypatch, httpx.MockTransport(handle))
    runner = SandboxedGymAgentTaskRunner(config=SandboxedGymRuntimeConfig(rollout_url=ROLLOUT_URL))

    with pytest.raises(RuntimeError, match=r"no rollout for 1 of 2 requested task") as excinfo:
        await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    # The check that fires is the CLI runner's own, reached because this runner writes the records
    # where that parser reads them -- which is the reuse claim, stated as a test.
    assert "will not be scored" in str(excinfo.value)


def test_runner_info_records_the_host_but_not_the_token() -> None:
    runner = SandboxedGymAgentTaskRunner(
        config=SandboxedGymRuntimeConfig(rollout_url=ROLLOUT_URL, auth_token="sk-secret-value")
    )

    info = runner.runner_info()

    assert info.name == "gym"
    assert info.config["mode"] == "sandboxed"
    assert info.config["rollout_url"] == ROLLOUT_URL
    assert "sk-secret-value" not in json.dumps(info.config), "the token must not reach the run bundle"


def _model_call(call_id: str, started_at: float) -> dict[str, Any]:
    return {
        "model_call_id": call_id,
        "model_ref": {"name": "policy_model"},
        "status_code": 200,
        "started_at": started_at,
        "completed_at": started_at + 0.25,
        "latency_ttft_ms": 12.5,
        "response": {"usage": {"input_tokens": 5, "output_tokens": 2}},
    }


async def test_captures_from_the_host_become_timed_per_call_spans(tasks, tmp_path, monkeypatch) -> None:
    # The point of the whole hop: without the capture a trial's trace is one AGENT span and no
    # timing, which is what this runner produced before.
    host = _FakeHost(model_calls=[_model_call("c0", 1788534870.5), _model_call("c1", 1788534871.0)])
    runner = runner_against(host, monkeypatch)

    trials = await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    trial = trials[0]
    assert trial.evidence is not None
    resource_spans = await (await trial.evidence.trace(EVIDENCE_TRACE, format=EVIDENCE_FORMAT_OTLP)).resource_spans()
    llm = [
        span
        for resource_span in resource_spans
        for scope_spans in resource_span.scope_spans
        for span in scope_spans.spans
        for attribute in span.attributes
        if attribute.key == "openinference.span.kind" and attribute.value.string_value == "LLM"
    ]
    assert len(llm) == 2
    assert all(span.start_time_unix_nano > 0 and span.end_time_unix_nano > span.start_time_unix_nano for span in llm)


async def test_the_raw_capture_is_kept_as_its_own_evidence(tasks, tmp_path, monkeypatch) -> None:
    # The OTLP view is a projection of the capture, and a projection is not a reason to lose what it
    # was projected from.
    host = _FakeHost(model_calls=[_model_call("c0", 1788534870.5)])
    runner = runner_against(host, monkeypatch)

    trials = await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    assert trials[0].evidence is not None
    assert trials[0].evidence.get("ng_trajectory") is not None


async def test_the_transport_key_never_reaches_the_rollouts_file(tasks, tmp_path, monkeypatch) -> None:
    # `rollouts.jsonl` is read by the same parser as the CLI runner's output, so it has to be the
    # shape Gym itself would have written -- this key is ours, not Gym's.
    host = _FakeHost(model_calls=[_model_call("c0", 1788534870.5)])
    runner = runner_against(host, monkeypatch)

    await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    written = (tmp_path / "gym_run" / "rollouts.jsonl").read_text(encoding="utf-8")
    assert MODEL_CALLS_RESULT_KEY not in written


async def test_a_host_that_returns_no_captures_still_produces_trials(tasks, tmp_path, monkeypatch) -> None:
    # An older host, or a run where the work path was unwritable. The trial's reward and output
    # stand on their own; only the per-call timing is missing.
    runner = runner_against(_FakeHost(), monkeypatch)

    trials = await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    assert len(trials) == len(tasks)
    assert not (tmp_path / "gym_run" / "model_calls").exists()


async def test_examples_carry_a_rollout_index_so_gym_will_capture(tasks, tmp_path, monkeypatch) -> None:
    # `run_examples` assigns no `_ng_rollout_index` -- only `gym eval run`'s preprocessing does.
    # Without it Gym's `maybe_rollout_id_from_run_body` returns None and it writes *no capture at
    # all*. Found by running a real sandboxed rollout, whose capture directory stayed empty.
    host = _FakeHost()
    runner = runner_against(host, monkeypatch)

    await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    assert all(example.get(NG_ROLLOUT_INDEX) is not None for example in host.posted)


def test_rollout_indices_number_repeats_within_each_task() -> None:
    # Per task, not across the batch: the index distinguishes repeats of one task, and a global
    # counter would make every rollout look like a different task's first attempt.
    examples = [{NG_TASK_INDEX: 0}, {NG_TASK_INDEX: 1}, {NG_TASK_INDEX: 0}]

    _stamp_rollout_indices(examples)

    assert [example[NG_ROLLOUT_INDEX] for example in examples] == [0, 0, 1]


def test_a_caller_that_numbers_its_own_repeats_keeps_its_numbering() -> None:
    examples = [{NG_TASK_INDEX: 0, NG_ROLLOUT_INDEX: 7}, {NG_TASK_INDEX: 0}]

    _stamp_rollout_indices(examples)

    assert [example[NG_ROLLOUT_INDEX] for example in examples] == [7, 0]


def test_a_host_failure_surfaces_the_host_output_not_just_a_loopback_500() -> None:
    """The component servers report an upstream refusal as a 500 from 127.0.0.1.

    Without the host's own output the caller is left with that address and a Gym traceback, and
    the sandbox holding the real cause has already been destroyed.
    """
    message = sandboxed._host_error_message(
        "http://sandbox/proxy/8080/rollouts/run",
        {
            "code": "internal",
            "message": "ClientResponseError: 500, url='http://127.0.0.1:5902/run'",
            "host_output_tail": ["(policy_model) upstream returned 401 Unauthorized"],
        },
    )

    assert "(policy_model) upstream returned 401 Unauthorized" in message
    assert "gym host output" in message, "the tail must be labelled, not spliced into the summary"
    assert "ClientResponseError" in message, "the host's own error still has to survive"


def test_a_host_failure_without_output_reads_as_before() -> None:
    message = sandboxed._host_error_message("http://sandbox/run", {"code": "internal", "message": "boom"})

    assert "gym host output" not in message
    assert "boom" in message


def test_a_non_mapping_error_is_still_rendered() -> None:
    """Older hosts send a bare string; it must not become a stack trace in the caller."""
    assert "plain failure" in sandboxed._host_error_message("http://sandbox/run", "plain failure")


async def test_a_503_bootstrap_failure_renders_the_envelope_it_carried(tasks, tmp_path, monkeypatch) -> None:
    """A host that failed to bootstrap answers 503 with the same envelope a 200 would carry.

    Truncating the raw body instead drops the output tail, which is ordered oldest first, so the
    cut lands on the traceback that says why the host never started.
    """
    tail = [f"bootstrap line {index}" for index in range(80)]
    host = _FakeHost(
        status=503,
        body={
            "error": {
                "code": "bootstrap_failed",
                "message": "PolicyCredentialRejected: the policy endpoint rejected the configured credential",
                "host_output_tail": tail,
            }
        },
    )
    runner = runner_against(host, monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        await runner.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))

    message = str(excinfo.value)
    assert "rejected the configured credential" in message
    assert "gym host output" in message, "the tail must be labelled, not left as raw JSON"
    assert "bootstrap line 0" in message, "the oldest line is where the traceback starts"
    assert "bootstrap line 79" in message
