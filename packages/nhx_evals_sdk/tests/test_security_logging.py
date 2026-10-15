# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-controlled SDK log fields must not forge additional log lines."""

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import openai
import pytest
from nhx_evals_sdk.execution.metric_execution import resolve_target_structured_output_mode
from nhx_evals_sdk.inference import make_inference_request
from nhx_evals_sdk.logging_utils import escape_log_value
from nhx_evals_sdk.resilience.config import ResilienceConfig
from nhx_evals_sdk.resilience.scheduler import ResilienceScheduler
from nhx_evals_sdk.retrieval.beir import BeirCorpusDocument, BeirDataset, BeirQuery
from nhx_evals_sdk.retrieval.dense_search import dense_search
from nhx_evals_sdk.retrieval.nim_embeddings import NimEmbeddingClient
from nhx_evals_sdk.structured_output import InferenceStructuredOutput, StructuredOutputMode
from nhx_evals_sdk.values.models import Model
from nhx_evals_sdk.values.params import RunConfigOnlineModel

_FORGED_NAME = "model\r\nINFO forged-success\u2028next-entry"


@pytest.mark.parametrize("separator", list("\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"))
def test_log_value_cannot_create_another_line(separator: str) -> None:
    escaped = escape_log_value(f"model{separator}INFO forged-success")
    assert escaped.splitlines() == [escaped]
    assert "model" in escaped and "INFO forged-success" in escaped
    assert escape_log_value("ordinary/model") == "ordinary/model"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["rate-limit", "unexpected", "success"])
async def test_inference_logs_escape_model_and_url_without_changing_request(
    failure: str, caplog: pytest.LogCaptureFixture
) -> None:
    model = Model(url="https://example.test/v1/chat/completions", name=_FORGED_NAME)
    client = Mock()
    client.base_url = "https://example.test/v1\r\nINFO forged-url"
    client.api_key = "test-key"
    completion = Mock()
    completion.model_dump.return_value = {"choices": []}
    operation = AsyncMock(return_value=completion)
    if failure == "rate-limit":
        operation.side_effect = openai.RateLimitError(
            "limited", response=httpx.Response(429, request=httpx.Request("POST", model.url)), body=None
        )
    elif failure == "unexpected":
        operation.side_effect = ValueError("unexpected")
    client.chat.completions.create = operation
    request = {"messages": [{"role": "user", "content": "prompt\r\nwith newlines"}]}

    with caplog.at_level(logging.INFO):
        if failure == "success":
            assert await make_inference_request(model, request, max_retries=0, client=client) == {"choices": []}
        else:
            with pytest.raises(RuntimeError):
                await make_inference_request(model, request, max_retries=0, client=client)

    operation.assert_awaited_once_with(model=_FORGED_NAME, **request)
    records = [record for record in caplog.records if record.name == "nhx_evals_sdk.inference"]
    assert records
    assert all(len(record.getMessage().splitlines()) == 1 for record in records)
    assert any(r"\r\nINFO forged-url" in record.getMessage() for record in records)


@pytest.mark.asyncio
async def test_scheduler_logs_escape_endpoint_but_preserve_state_key(caplog: pytest.LogCaptureFixture) -> None:
    scheduler = ResilienceScheduler(ResilienceConfig(cooldown_seconds_soft=0, backoff_initial_ms=0, backoff_cap_ms=0))
    operation = AsyncMock(side_effect=[httpx.ReadTimeout("timeout"), "ok"])
    with caplog.at_level(logging.DEBUG, logger="nhx_evals_sdk.resilience.scheduler"):
        assert await scheduler.run_with_resilience(_FORGED_NAME, operation, max_attempts=2, deadline_at=None) == "ok"

    records = [record for record in caplog.records if hasattr(record, "endpoint_key")]
    assert len(records) >= 3  # Started, retried, completed, plus any admission changes.
    assert all(len(record.endpoint_key.splitlines()) == 1 for record in records)
    assert (await scheduler._get_controller(_FORGED_NAME)).state.key == _FORGED_NAME


@pytest.mark.asyncio
async def test_retrieval_logs_escape_model_without_changing_embedding_requests(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = Model(url="https://example.test/v1", name=_FORGED_NAME)
    dataset = BeirDataset(
        root=Path("."),
        corpus={"doc": BeirCorpusDocument(id="doc", text="passage")},
        queries={"query": BeirQuery(id="query", text="question")},
        qrels={},
    )
    requests: list[bytes] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request.content)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    with caplog.at_level(logging.INFO, logger="nhx_evals_sdk.retrieval.dense_search"):
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await dense_search(dataset, NimEmbeddingClient(model=model), client=client)

    assert result == {"query": {"doc": 1.0}}
    assert len(requests) == 2
    assert all(json.loads(request)["model"] == _FORGED_NAME for request in requests)
    records = [record for record in caplog.records if record.name == "nhx_evals_sdk.retrieval.dense_search"]
    assert records
    assert all(len(record.getMessage().splitlines()) == 1 for record in records)
    assert model.name == _FORGED_NAME


@pytest.mark.asyncio
async def test_structured_output_logs_escape_target_name(caplog: pytest.LogCaptureFixture) -> None:
    model = Model(url="https://example.test/v1/chat/completions", name=_FORGED_NAME)
    schema = {"type": "object", "properties": {"score": {"type": "integer"}}}
    hook = InferenceStructuredOutput(StructuredOutputMode.UNSUPPORTED, {"schema": schema})
    inference_fn = AsyncMock(return_value={"choices": [{"finish_reason": "length", "message": {"content": ""}}]})

    with caplog.at_level(logging.INFO):
        await resolve_target_structured_output_mode(
            preprocess_hooks=[hook],
            model=model,
            inference_fn=inference_fn,
            params=RunConfigOnlineModel(structured_output={"schema": schema}),
        )

    records = [record for record in caplog.records if "target" in record.getMessage() or "probe" in record.getMessage()]
    assert len(records) == 4  # Three truncated probes and the chosen fallback.
    assert all(len(record.getMessage().splitlines()) == 1 for record in records)
    assert all(call.args[0].name == _FORGED_NAME for call in inference_fn.await_args_list)
