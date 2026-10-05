# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a rollout batch reaches the Gym host and how its failures are classified.

Two hops can refuse a rollout and they say so differently: the Gym host wraps its own failures as
a nested ``{"error": {...}}``, while the sandbox proxy answers with a flat ``{"code", "message"}``
-- and does so when its read timeout fires, where the message is empty. Confusing the two turns a
proxy giving up into "the environment failed", which sends the reader to the wrong system.
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from email.message import Message
from typing import Any
from unittest.mock import MagicMock

import pytest
from sandboxed_gym.config import BrokerEndpoint
from sandboxed_gym.host.models import GymHostHandle
from sandboxed_gym.orchestrator import RolloutTransportError, SandboxedGymSession
from sandboxed_gym.serve_config import SandboxedGymServeConfig

ROLLOUT_URL = "http://host/rollouts/run"


def _session(**sandbox_overrides: Any) -> SandboxedGymSession:
    cfg = SandboxedGymServeConfig.model_validate(
        {
            "job_id": "job-1",
            "sandbox": {
                "image": "runtime:dev",
                "network_policy": {"egress_allow": []},
                "environment_pvc_claim": "env",
                "workspace_pvc_claim": "work",
                "rollout_retry_backoff_s": 0,
                **sandbox_overrides,
            },
        }
    )
    return SandboxedGymSession(
        cfg=cfg,
        broker_server=MagicMock(),
        broker=BrokerEndpoint(url="http://broker:1", host="broker", port=1, token="t"),
        host_provider=MagicMock(),
        host=GymHostHandle(host_id="h1", health_url="http://host/health", rollout_url=ROLLOUT_URL, headers={}),
    )


def _http_error(status: int, body: dict[str, Any]) -> urllib.error.HTTPError:
    payload = io.BytesIO(json.dumps(body).encode())
    return urllib.error.HTTPError(ROLLOUT_URL, status, "err", Message(), payload)


def _responds(monkeypatch: pytest.MonkeyPatch, results: list[Any]) -> list[list[dict]]:
    """Record each chunk posted and answer it with ``results``."""
    posted: list[list[dict]] = []

    def _post(self: SandboxedGymSession, chunk: list[dict]) -> list[Any]:
        posted.append(chunk)
        return results

    monkeypatch.setattr(SandboxedGymSession, "_post_chunk", _post)
    return posted


# --------------------------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------------------------


def test_a_batch_is_split_into_bounded_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Request duration and response size have to track the chunk, not the batch.

    A whole batch in one POST is what puts the request over the proxy's cap and the response over
    max_response_bytes at the same time, with nothing to lower but the batch itself.
    """
    posted = _responds(monkeypatch, [{"reward": 1.0}])
    session = _session(rollout_chunk_size=2)

    results = session.run_rollouts([{"id": index} for index in range(5)])

    # Sizes sorted, not in dispatch order: chunks go out concurrently, so which one records itself
    # first is thread scheduling. What is guaranteed is the partition -- every example sent once.
    assert sorted(len(chunk) for chunk in posted) == [1, 2, 2]
    assert sorted(row["id"] for chunk in posted for row in chunk) == list(range(5))
    assert len(results) == 3  # one stub result per chunk


def test_an_empty_batch_is_refused() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _session().run_rollouts([])


def test_concurrency_is_bounded_by_max_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunks must run concurrently, but only so many at once.

    Asserts the peak is exactly the limit, not merely under it: a stub that returned instantly
    would never overlap, and the test would pass just as well against unbounded fan-out.
    """
    lock = threading.Lock()
    in_flight = 0
    peak = 0

    def _post(self: SandboxedGymSession, chunk: list[dict]) -> list[Any]:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1
        return [{"reward": 1.0}]

    monkeypatch.setattr(SandboxedGymSession, "_post_chunk", _post)
    _session(rollout_chunk_size=1, rollout_max_in_flight=2).run_rollouts([{"id": i} for i in range(8)])

    assert peak == 2


# --------------------------------------------------------------------------------------------
# What a running batch says about itself
# --------------------------------------------------------------------------------------------


def test_a_chunk_announces_itself_before_it_goes_out(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A chunk that never comes back is the case worth logging, and only a line written
    before the POST can cover it.

    Asserted through a chunk that fails rather than one that hangs, because both reach the
    completion log the same way -- they don't. A test that let the POST succeed would pass
    against a log line written afterwards.
    """

    def _post(self: SandboxedGymSession, chunk: list[dict]) -> list[Any]:
        raise RolloutTransportError("host is gone", retryable=False, origin="sandbox")

    monkeypatch.setattr(SandboxedGymSession, "_post_chunk", _post)
    caplog.set_level(logging.INFO, logger="sandboxed_gym.orchestrator")

    with pytest.raises(RolloutTransportError):
        _session(rollout_chunk_size=2).run_rollouts([{"id": i} for i in range(2)])

    assert "rollout chunk 1/1: POST 2 example(s)" in caplog.text
    # The chunk never returned, so nothing should claim it did.
    assert "result(s) in" not in caplog.text


def test_a_retried_chunk_names_the_attempt(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Two identical POST lines for one chunk would read as two chunks."""
    attempts = 0

    def _post(self: SandboxedGymSession, chunk: list[dict]) -> list[Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RolloutTransportError("dropped", retryable=True, origin="proxy")
        return [{"reward": 1.0}]

    monkeypatch.setattr(SandboxedGymSession, "_post_chunk", _post)
    caplog.set_level(logging.INFO, logger="sandboxed_gym.orchestrator")

    _session(rollout_chunk_size=1, rollout_max_attempts=2).run_rollouts([{"id": 0}])

    assert "rollout chunk 1/1: POST 1 example(s)\n" in caplog.text + "\n"
    assert "(attempt 2/2)" in caplog.text


# --------------------------------------------------------------------------------------------
# Who refused the rollout
# --------------------------------------------------------------------------------------------


def test_a_host_error_is_attributed_to_the_sandbox_and_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host answered, so the environment failed: deterministic, and a retry only costs time.

    Drives the real ``_post_chunk`` by failing the HTTP call, so the classification under test is
    the one production runs rather than a re-implementation of it.
    """
    attempts = 0

    def _urlopen(request, timeout=None):
        nonlocal attempts
        attempts += 1
        raise _http_error(500, {"error": {"code": "internal", "message": "env exploded"}})

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)

    with pytest.raises(RolloutTransportError) as excinfo:
        _session().run_rollouts([{"id": 1}])

    assert excinfo.value.origin == "sandbox"
    assert "env exploded" in str(excinfo.value)
    assert attempts == 1, "a sandbox-reported error must not be retried"


def test_a_transport_failure_reaching_no_host_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection that never landed is the case a retry genuinely fixes."""
    attempts = 0

    def _urlopen(request, timeout=None):
        nonlocal attempts
        attempts += 1
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)

    with pytest.raises(RolloutTransportError) as excinfo:
        _session(rollout_max_attempts=2).run_rollouts([{"id": 1}])

    assert attempts == 2
    assert excinfo.value.origin == "proxy"


def test_a_proxy_error_is_attributed_to_the_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The proxy's envelope is flat, and empty when its read timeout fires.

    Matching only the host's nested envelope would let this fall through every branch and surface
    as an unexpected response shape -- a parser bug with no message, for the failure most likely
    to happen in production.
    """
    session = _session()
    error = session._transport_error(json.dumps({"code": "timeout", "message": ""}), 504, [{}], 0.0)

    assert error.origin == "proxy"
    assert not error.retryable
    assert "rejected by the sandbox proxy" in str(error)


def test_an_unrecognised_body_stays_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither envelope means the request died in transit, which a retry can genuinely fix."""
    session = _session()
    error = session._transport_error("<html>502 Bad Gateway</html>", 502, [{}], 0.0)

    assert error.origin == "proxy"
    assert error.retryable
    assert "502 Bad Gateway" in str(error)


def test_a_200_carrying_an_error_is_a_sandbox_failure() -> None:
    """The host commits its status line before the work finishes, so a late failure rides the body.

    Reading only the status would treat this as a successful batch and hand the caller an
    ``error`` dict where results were expected.
    """
    session = _session()

    with pytest.raises(RolloutTransportError) as excinfo:
        session._decode_results(json.dumps({"error": {"code": "deadline_exceeded"}}).encode())

    assert excinfo.value.origin == "sandbox"
    assert not excinfo.value.retryable


@pytest.mark.parametrize("status", [200, 503])
def test_a_host_reported_error_carries_the_hosts_output(status: int) -> None:
    """The output tail is what says why the host failed; the code alone does not."""
    session = _session()
    envelope = {"error": {"code": "bootstrap_failed", "message": "not ready", "host_output_tail": ["No module"]}}

    if status == 200:
        with pytest.raises(RolloutTransportError) as excinfo:
            session._decode_results(json.dumps(envelope).encode())
        error = excinfo.value
    else:
        error = session._transport_error(json.dumps(envelope), status, [{}], 0.0)

    assert error.origin == "sandbox"
    assert "bootstrap_failed" in str(error)
    assert "--- gym host output (1 lines) ---\nNo module" in str(error)


def test_heartbeat_whitespace_does_not_break_decoding() -> None:
    """The host pads the body while it works; that padding reaches this decoder verbatim."""
    session = _session()

    assert session._decode_results(b'    {"results": [{"reward": 1.0}]}') == [{"reward": 1.0}]


# --------------------------------------------------------------------------------------------
# Retry
# --------------------------------------------------------------------------------------------


def test_a_transport_failure_is_retried_to_the_attempt_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def _post(self: SandboxedGymSession, chunk: list[dict]) -> list[Any]:
        nonlocal attempts
        attempts += 1
        raise RolloutTransportError("in transit", retryable=True, origin="proxy")

    monkeypatch.setattr(SandboxedGymSession, "_post_chunk", _post)

    with pytest.raises(RolloutTransportError) as excinfo:
        _session(rollout_max_attempts=3).run_rollouts([{"id": 1}])

    assert attempts == 3
    assert "failed after 3 attempt(s)" in str(excinfo.value)


def test_a_retry_that_succeeds_returns_the_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    def _post(self: SandboxedGymSession, chunk: list[dict]) -> list[Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RolloutTransportError("in transit", retryable=True, origin="proxy")
        return [{"reward": 1.0}]

    monkeypatch.setattr(SandboxedGymSession, "_post_chunk", _post)

    assert _session().run_rollouts([{"id": 1}]) == [{"reward": 1.0}]


def test_every_chunk_failing_points_at_the_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """One bad chunk is a request problem; all of them is the sandbox, and reads very differently."""

    def _post(self: SandboxedGymSession, chunk: list[dict]) -> list[Any]:
        raise RolloutTransportError("gone", retryable=False, origin="proxy")

    monkeypatch.setattr(SandboxedGymSession, "_post_chunk", _post)

    with pytest.raises(RolloutTransportError) as excinfo:
        _session(rollout_chunk_size=1).run_rollouts([{"id": 1}, {"id": 2}])

    assert "Every chunk failed" in str(excinfo.value)
    assert "2 of 2" in str(excinfo.value)


def test_a_body_that_dies_mid_read_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reaching a body means the host accepted the chunk and started running it.

    The host answers ``200`` only after ``submit_rollouts`` is under way, and it has no request id
    to deduplicate against, so retrying here runs the same examples a second time while the first
    is still going. Duplicated generation and duplicated environment side effects are worse than a
    failed chunk. The failure is still *classified* -- it carries the chunk, the URL and the
    elapsed time -- it just does not go round again.
    """
    attempts = 0

    class _DyingResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            raise ConnectionResetError("connection reset by peer")

    def _urlopen(request, timeout=None):
        nonlocal attempts
        attempts += 1
        return _DyingResponse()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)

    with pytest.raises(RolloutTransportError) as excinfo:
        _session(rollout_max_attempts=3).run_rollouts([{"id": 1}])

    assert attempts == 1, "a chunk the host had already accepted was sent again"
    assert excinfo.value.origin == "proxy"
    assert not excinfo.value.retryable
    assert "ConnectionResetError" in str(excinfo.value)


def test_a_connection_that_never_landed_is_still_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """The counterpart: no response headers means the host never began the chunk.

    This is the case a retry genuinely fixes, and the one that must survive the restriction above.
    """
    attempts = 0

    def _urlopen(request, timeout=None):
        nonlocal attempts
        attempts += 1
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)

    with pytest.raises(RolloutTransportError):
        _session(rollout_max_attempts=2).run_rollouts([{"id": 1}])

    assert attempts == 2


def test_a_heartbeat_only_body_names_the_host_as_the_failure() -> None:
    """A host that dies mid-batch leaves exactly this: a 200, some padding, then nothing.

    ``json.loads`` alone would call it malformed JSON, which reads as a response-contract bug and
    sends the reader to the wrong system. It is the sandbox having gone away.
    """
    session = _session()

    with pytest.raises(RolloutTransportError) as excinfo:
        session._decode_results(b"    ", 42.0)

    assert excinfo.value.origin == "sandbox"
    assert not excinfo.value.retryable
    assert "OOMKilled" in str(excinfo.value)


def test_an_empty_body_names_both_causes_rather_than_picking_one() -> None:
    """Zero bytes is produced by a dead host *and* by a request the proxy truncated.

    A heartbeatless image cut at the proxy's cap and a host OOMKilled mid-batch both land here.
    Without a Content-Length the body ends at the close either way, so the client cannot tell them
    apart and must not claim to.
    """
    session = _session()

    with pytest.raises(RolloutTransportError) as excinfo:
        session._decode_results(b"", 180.0)

    message = str(excinfo.value)
    assert "OOMKill" in message
    assert "sandbox.image" in message


def test_an_empty_body_mid_range_does_not_blame_the_image_alone() -> None:
    session = _session()

    with pytest.raises(RolloutTransportError) as excinfo:
        session._decode_results(b"", 60.0)

    assert "OOMKill" in str(excinfo.value)


def test_an_empty_body_that_arrived_fast_blames_the_sandbox_alone() -> None:
    session = _session()

    with pytest.raises(RolloutTransportError) as excinfo:
        session._decode_results(b"", 2.0)

    message = str(excinfo.value)
    assert excinfo.value.origin == "sandbox"
    assert "OOMKilled" in message
    assert "sandbox.image" not in message


def test_a_malformed_non_empty_body_is_a_contract_failure() -> None:
    """Distinct from the above, and deliberately not retried: the host said something wrong."""
    session = _session()

    with pytest.raises(RolloutTransportError) as excinfo:
        session._decode_results(b"<html>gateway</html>")

    assert not excinfo.value.retryable
    assert "not JSON" in str(excinfo.value)


def test_a_non_utf8_body_is_a_contract_failure_not_a_decode_crash() -> None:
    """An uncaught decode raises UnicodeDecodeError, which is not a RolloutTransportError.

    ``_post_chunk_with_retries`` only catches RolloutTransportError, so anything else reaches
    ``_post_all_chunks`` and is re-raised as a bug in this process -- losing the chunk label and
    the sandbox attribution for a failure the host produced.
    """
    session = _session()

    with pytest.raises(RolloutTransportError) as excinfo:
        session._decode_results(b'{"results": ["\xff"]}')

    assert excinfo.value.origin == "sandbox"
    assert not excinfo.value.retryable
    assert "not UTF-8" in str(excinfo.value)


def test_a_json_refusal_that_is_not_a_decode_error_is_a_contract_failure() -> None:
    """``json`` has refusals that are not ``JSONDecodeError``, and they escape a narrow catch.

    Both bodies below are well within ``max_response_bytes``, so nothing upstream stops them.
    """
    session = _session()

    for payload in (b"1" * 5000, b"[" * 100_000 + b"]" * 100_000):
        with pytest.raises(RolloutTransportError) as excinfo:
            session._decode_results(payload)

        assert excinfo.value.origin == "sandbox"
        assert not excinfo.value.retryable


def test_a_non_list_results_field_is_a_contract_failure() -> None:
    """``{"results": null}`` would raise TypeError from ``list(None)`` and escape the same way."""
    session = _session()

    with pytest.raises(RolloutTransportError) as excinfo:
        session._decode_results(b'{"results": null}')

    assert excinfo.value.origin == "sandbox"
    assert not excinfo.value.retryable
    assert "not a list" in str(excinfo.value)


def test_a_top_level_scalar_response_is_a_contract_failure() -> None:
    """The host produced it, so it is attributed to the sandbox like every other bad envelope."""
    session = _session()

    with pytest.raises(RolloutTransportError) as excinfo:
        session._decode_results(b"42")

    assert excinfo.value.origin == "sandbox"
    assert not excinfo.value.retryable
    assert "unexpected rollout response shape" in str(excinfo.value)


def test_a_bug_in_this_process_is_not_disguised_as_a_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrapping every failure would give a programming error a plausible origin and retryable flag.

    It also swallows the stack that explains it, which is how a missing method once read as a
    sandbox that had stopped answering.
    """

    def _post(self: SandboxedGymSession, chunk: list[dict]) -> list[Any]:
        raise AttributeError("no such method")

    monkeypatch.setattr(SandboxedGymSession, "_post_chunk", _post)

    with pytest.raises(AttributeError, match="no such method"):
        _session().run_rollouts([{"id": 1}])
