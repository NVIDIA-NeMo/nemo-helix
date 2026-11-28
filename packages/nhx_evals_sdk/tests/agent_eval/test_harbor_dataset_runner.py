# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``DatasetRunnerAgent`` answers every row of a shard and leaves one responses file in the sandbox."""

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

pytest.importorskip("harbor", reason="DatasetRunnerAgent subclasses Harbor's BaseAgent")

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from nhx_evals_sdk.agent_eval.runtimes.harbor import dataset_runner
from nhx_evals_sdk.agent_eval.runtimes.harbor.dataset_runner import (
    BASE_URL_ENV,
    RESPONSES_PATH,
    SHARD_SCHEMA,
    DatasetRunnerAgent,
    parse_shard,
)

BASE_URL = "http://gateway.test/v1"


class FakeEnvironment:
    """Records what the agent puts in the sandbox."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.files: dict[str, str] = {}

    async def exec(self, command: str, **_: Any) -> None:
        self.commands.append(command)

    async def upload_file(self, source: Path, target: str) -> None:
        self.files[target] = Path(source).read_text(encoding="utf-8")


def _sandbox(environment: FakeEnvironment) -> BaseEnvironment:
    """The fake as the agent sees it: it touches only ``exec`` and ``upload_file``."""
    return cast(BaseEnvironment, environment)


def _instruction(*rows: dict[str, Any]) -> str:
    return json.dumps({"schema": SHARD_SCHEMA, "rows": list(rows)})


def _row(row_id: str, text: str = "hi", **params: Any) -> dict[str, Any]:
    return {"id": row_id, "messages": [{"role": "user", "content": text}], "params": params}


def _completion(text: str, completion_tokens: int = 3) -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": completion_tokens, "total_tokens": 5 + completion_tokens},
    }


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch):
    """Route the agent's client through a handler the test sets."""
    state: dict[str, Any] = {"requests": []}

    def handle(request: httpx.Request) -> httpx.Response:
        state["requests"].append(request)
        return state["handler"](request)

    real_client = httpx.AsyncClient

    def client(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(dataset_runner, "AsyncClient", client)
    return state


def _agent(tmp_path: Path, **kwargs: Any) -> DatasetRunnerAgent:
    return DatasetRunnerAgent(
        logs_dir=tmp_path / "agent",
        model_name="default/model",
        extra_env={BASE_URL_ENV: BASE_URL},
        retry_backoff_sec=0,
        **kwargs,
    )


def _records(environment: FakeEnvironment) -> dict[str, dict[str, Any]]:
    lines = environment.files[RESPONSES_PATH].splitlines()
    return {record["id"]: record for record in map(json.loads, lines)}


async def test_answers_every_row_with_the_rows_request(tmp_path: Path, transport: dict[str, Any]) -> None:
    transport["handler"] = lambda request: httpx.Response(
        200, json=_completion(json.loads(request.content)["messages"][0]["content"].upper())
    )
    environment, context = FakeEnvironment(), AgentContext()

    await _agent(tmp_path).run(
        _instruction(_row("a", "one", max_tokens=256, temperature=0), _row("b", "two")), _sandbox(environment), context
    )

    records = _records(environment)
    assert {row_id: record["response"]["choices"][0]["message"]["content"] for row_id, record in records.items()} == {
        "a": "ONE",
        "b": "TWO",
    }
    assert all(record["status"] == "ok" and record["attempts"] == 1 for record in records.values())
    first = json.loads(transport["requests"][0].content)
    assert first == {
        "max_tokens": 256,
        "temperature": 0,
        "model": "default/model",
        "messages": [{"role": "user", "content": "one"}],
    }
    assert str(transport["requests"][0].url) == f"{BASE_URL}/chat/completions"
    assert transport["requests"][0].headers["Authorization"] == "Bearer not-used"
    assert context.n_input_tokens == 10 and context.n_output_tokens == 6
    assert context.metadata is not None and context.metadata["dataset_runner"]["ok"] == 2


async def test_retries_a_retryable_status_then_records_the_answer(tmp_path: Path, transport: dict[str, Any]) -> None:
    replies = iter([httpx.Response(503, text="busy"), httpx.Response(200, json=_completion("ok"))])
    transport["handler"] = lambda request: next(replies)
    environment = FakeEnvironment()

    await _agent(tmp_path).run(_instruction(_row("a")), _sandbox(environment), AgentContext())

    assert _records(environment)["a"]["status"] == "ok"
    assert _records(environment)["a"]["attempts"] == 2


async def test_a_failed_row_is_recorded_without_failing_the_shard(tmp_path: Path, transport: dict[str, Any]) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        text = json.loads(request.content)["messages"][0]["content"]
        return (
            httpx.Response(400, text="bad image") if text == "broken" else httpx.Response(200, json=_completion(text))
        )

    transport["handler"] = handle
    environment = FakeEnvironment()

    await _agent(tmp_path).run(
        _instruction(_row("a", "broken"), _row("b", "fine")), _sandbox(environment), AgentContext()
    )

    records = _records(environment)
    assert records["a"]["status"] == "error" and records["a"]["attempts"] == 1
    assert records["a"]["response"] is None and "HTTP 400" in records["a"]["error"]
    assert records["b"]["status"] == "ok"


async def test_a_row_that_keeps_timing_out_is_a_timeout(tmp_path: Path, transport: dict[str, Any]) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    transport["handler"] = handle
    environment = FakeEnvironment()

    await _agent(tmp_path, row_max_attempts=2).run(_instruction(_row("a")), _sandbox(environment), AgentContext())

    record = _records(environment)["a"]
    assert (record["status"], record["attempts"]) == ("timeout", 2)


async def test_never_holds_more_rows_in_flight_than_max_concurrency(tmp_path: Path, monkeypatch) -> None:
    in_flight, peak = 0, 0

    async def answer(self: DatasetRunnerAgent, client: httpx.AsyncClient, row: Any) -> dict[str, Any]:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await dataset_runner.asyncio.sleep(0)
        in_flight -= 1
        return {"id": row.id, "status": "ok", "response": _completion("x"), "attempts": 1, "latency_ms": 0}

    monkeypatch.setattr(DatasetRunnerAgent, "_answer", answer)
    environment = FakeEnvironment()

    await _agent(tmp_path, max_concurrency=3).run(
        _instruction(*(_row(str(index)) for index in range(20))), _sandbox(environment), AgentContext()
    )

    assert peak == 3
    assert len(_records(environment)) == 20


async def test_refuses_to_run_without_an_endpoint(tmp_path: Path) -> None:
    agent = DatasetRunnerAgent(logs_dir=tmp_path, model_name="default/model")
    with pytest.raises(ValueError, match=BASE_URL_ENV):
        await agent.run(_instruction(_row("a")), _sandbox(FakeEnvironment()), AgentContext())


@pytest.mark.parametrize(
    ("instruction", "message"),
    [
        ("not json", "not a nemo.dataset-shard/v1 document"),
        (json.dumps({"schema": "other/v1", "rows": [_row("a")]}), "this agent reads"),
        (_instruction(_row("a"), _row("a")), "repeats a row id"),
        (_instruction(_row("a", model="other")), "the agent sets those"),
    ],
)
def test_parse_shard_refuses_what_it_cannot_run(instruction: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_shard(instruction)


def test_kwargs_given_as_strings_are_read_as_numbers(tmp_path: Path) -> None:
    agent = DatasetRunnerAgent(logs_dir=tmp_path, max_concurrency="4", row_timeout_sec="30", row_max_attempts="2")
    assert (agent.max_concurrency, agent.row_timeout_sec, agent.row_max_attempts) == (4, 30.0, 2)
    with pytest.raises(ValueError, match="max_concurrency"):
        DatasetRunnerAgent(logs_dir=tmp_path, max_concurrency=0)
