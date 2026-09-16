# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import math
from pathlib import Path

import httpx
import pytest
from nemo_evaluator_sdk.retrieval.beir import BeirDataset
from nemo_evaluator_sdk.retrieval.dense_search import dense_search, retrieve
from nemo_evaluator_sdk.retrieval.nim_embeddings import NimEmbeddingClient, NimEmbeddingError
from nemo_evaluator_sdk.values.models import Model
from nemo_evaluator_sdk.values.retrieval import Retrieval


def _model() -> Model:
    return Model(url="https://embed.example.test/v1", name="embed-model")


def _response(request: httpx.Request, vectors: list[list[float]]) -> httpx.Response:
    return httpx.Response(
        200,
        request=request,
        content=json.dumps(
            {"data": [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)]},
            allow_nan=True,
        ),
        headers={"content-type": "application/json"},
    )


@pytest.mark.asyncio
async def test_embedding_client_sends_nim_input_type_and_checks_dimension() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(request, [[1.0, 0.0]])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        vectors = await NimEmbeddingClient(model=_model(), dimensions=2).encode(
            ["question"],
            input_type="query",
            client=client,
        )

    assert vectors == [[1.0, 0.0]]
    assert requests[0].url == "https://embed.example.test/v1/embeddings"
    assert json.loads(requests[0].content) == {
        "model": "embed-model",
        "input": ["question"],
        "input_type": "query",
        "encoding_format": "float",
        "modality": "text",
    }


@pytest.mark.asyncio
async def test_embedding_client_replaces_chat_completion_route() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(request, [[1.0, 0.0]])

    model = Model(
        url="https://igw.example.test/v1/chat/completions?model=embed",
        name="embed-model",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await NimEmbeddingClient(model=model, dimensions=2).encode(
            ["question"],
            input_type="query",
            client=client,
        )

    assert requests[0].url == "https://igw.example.test/v1/embeddings?model=embed"


@pytest.mark.asyncio
async def test_embedding_client_retries_http_503(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("nemo_evaluator_sdk.retrieval.nim_embeddings.asyncio.sleep", fake_sleep)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 4:
            return httpx.Response(503, request=request, text="unavailable")
        return _response(request, [[1.0, 0.0]])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await NimEmbeddingClient(model=_model(), dimensions=2).encode(
            ["question"],
            input_type="query",
            client=client,
        )

    assert attempts == 4
    assert sleeps == [0.5, 1.0, 2.0]
    assert result == [[1.0, 0.0]]


@pytest.mark.asyncio
async def test_embedding_client_does_not_retry_http_400() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, request=request, text="bad request")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(NimEmbeddingError, match="embedding HTTP 400"):
            await NimEmbeddingClient(model=_model(), dimensions=2).encode(
                ["question"],
                input_type="query",
                client=client,
            )

    assert attempts == 1


@pytest.mark.asyncio
async def test_embedding_client_does_not_retry_vlm_image_503() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            503,
            request=request,
            text='{"detail":"{\\"message\\":\\"image inputs require VLM serving to be enabled on this server\\"}"}',
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(NimEmbeddingError, match="image inputs require VLM"):
            await NimEmbeddingClient(model=_model(), dimensions=2).encode(
                ["data:image/png;base64,abc"],
                input_type="passage",
                client=client,
            )

    assert attempts == 1


@pytest.mark.asyncio
async def test_embedding_client_includes_http_error_body(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr("nemo_evaluator_sdk.retrieval.nim_embeddings.asyncio.sleep", fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request, text='{"detail":"queue full"}')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(NimEmbeddingError, match="queue full"):
            await NimEmbeddingClient(model=_model(), dimensions=2, max_retries=0).encode(
                ["question"],
                input_type="query",
                client=client,
            )


@pytest.mark.asyncio
async def test_embedding_client_retries_non_finite_response(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(delay: float) -> None:
        del delay

    monkeypatch.setattr("nemo_evaluator_sdk.retrieval.nim_embeddings.asyncio.sleep", fake_sleep)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        vector = [math.nan, 0.0] if attempts == 1 else [1.0, 0.0]
        return _response(request, [vector])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await NimEmbeddingClient(model=_model(), dimensions=2, max_retries=1).encode(
            ["question"],
            input_type="query",
            client=client,
        )

    assert attempts == 2
    assert result == [[1.0, 0.0]]


@pytest.mark.asyncio
async def test_embedding_client_accepts_native_width_when_dimensions_omitted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _response(request, [[1.0, 0.0, 0.0, 0.0]])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        vectors = await NimEmbeddingClient(model=_model()).encode(
            ["question"],
            input_type="query",
            client=client,
        )

    assert vectors == [[1.0, 0.0, 0.0, 0.0]]


@pytest.mark.asyncio
async def test_embedding_client_rejects_wrong_dimension() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _response(request, [[1.0]])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(NimEmbeddingError, match="expected embedding dimension 2"):
            await NimEmbeddingClient(model=_model(), dimensions=2).encode(
                ["question"],
                input_type="query",
                client=client,
            )


@pytest.mark.asyncio
async def test_embedding_client_rejects_duplicate_indexes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            content=json.dumps(
                {"data": [{"index": 0, "embedding": [1.0, 0.0]}, {"index": 0, "embedding": [0.0, 1.0]}]}
            ),
            headers={"content-type": "application/json"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(NimEmbeddingError, match="invalid indexes"):
            await NimEmbeddingClient(model=_model(), dimensions=2).encode(
                ["first", "second"],
                input_type="query",
                client=client,
            )


@pytest.mark.asyncio
async def test_embedding_client_rejects_skipped_indexes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            content=json.dumps(
                {"data": [{"index": 0, "embedding": [1.0, 0.0]}, {"index": 2, "embedding": [0.0, 1.0]}]}
            ),
            headers={"content-type": "application/json"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(NimEmbeddingError, match="invalid indexes"):
            await NimEmbeddingClient(model=_model(), dimensions=2).encode(
                ["first", "second"],
                input_type="query",
                client=client,
            )


@pytest.mark.asyncio
async def test_dense_search_ranks_documents_and_uses_passage_then_query(tmp_path: Path) -> None:
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text(
        '{"_id":"d1","text":"alpha"}\n{"_id":"d2","text":"beta"}\n',
        encoding="utf-8",
    )
    (tmp_path / "queries.jsonl").write_text('{"_id":"q1","text":"alpha?"}\n', encoding="utf-8")
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
        encoding="utf-8",
    )
    input_types: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        input_types.append(payload["input_type"])
        vectors = [[1.0, 0.0], [0.0, 1.0]] if payload["input_type"] == "passage" else [[0.8, 0.2]]
        return _response(request, vectors)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await dense_search(
            BeirDataset.from_path(tmp_path),
            NimEmbeddingClient(model=_model(), dimensions=2),
            client=client,
        )

    assert input_types == ["passage", "query"]
    assert list(results["q1"]) == ["d1", "d2"]
    assert results["q1"]["d1"] > results["q1"]["d2"]


@pytest.mark.asyncio
async def test_dense_search_pipelines_two_embedding_requests(tmp_path: Path) -> None:
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text(
        '{"_id":"d1","text":"a"}\n{"_id":"d2","text":"b"}\n{"_id":"d3","text":"c"}\n{"_id":"d4","text":"d"}\n',
        encoding="utf-8",
    )
    (tmp_path / "queries.jsonl").write_text('{"_id":"q1","text":"a?"}\n', encoding="utf-8")
    (tmp_path / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")

    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    class PipelinedTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, max_in_flight
            payload = json.loads(request.content)
            async with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.05)
            vectors = [[1.0, 0.0] for _ in payload["input"]]
            async with lock:
                in_flight -= 1
            return _response(request, vectors)

    async with httpx.AsyncClient(transport=PipelinedTransport()) as client:
        results = await dense_search(
            BeirDataset.from_path(tmp_path),
            NimEmbeddingClient(model=_model(), dimensions=2),
            batch_size=1,
            in_flight=2,
            client=client,
        )

    assert max_in_flight == 2
    assert list(results["q1"])[0] == "d1"


def _write_beir(tmp_path: Path, *, title: str = "", text: str = "alpha") -> BeirDataset:
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text(
        json.dumps({"_id": "d1", "title": title, "text": text}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "queries.jsonl").write_text('{"_id":"q1","text":"alpha?"}\n', encoding="utf-8")
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
        encoding="utf-8",
    )
    return BeirDataset.from_path(tmp_path)


@pytest.mark.asyncio
async def test_retrieve_truncates_long_documents_before_embed(tmp_path: Path) -> None:
    from nemo_evaluator_sdk.retrieval.passages import DOCUMENT_CHARACTER_LIMIT

    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload.get("input_type") == "passage":
            sent.extend(payload["input"])
            return _response(request, [[1.0, 0.0]])
        return _response(request, [[1.0, 0.0]])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await retrieve(
            _write_beir(tmp_path, title="Title", text="x" * (DOCUMENT_CHARACTER_LIMIT + 80)),
            Retrieval(embeddings=_model(), embedding_dimensions=2, truncate_long_documents="end"),
            client=client,
        )

    assert len(sent) == 1
    assert len(sent[0]) == DOCUMENT_CHARACTER_LIMIT


@pytest.mark.asyncio
async def test_retrieve_reranks_dense_hits(tmp_path: Path) -> None:
    ranking_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/ranking"):
            payload = json.loads(request.content)
            ranking_bodies.append(payload)
            return httpx.Response(
                200,
                request=request,
                content=json.dumps({"rankings": [{"index": 0, "logit": 4.2}]}),
                headers={"content-type": "application/json"},
            )
        payload = json.loads(request.content)
        vectors = [[1.0, 0.0]] if payload["input_type"] == "passage" else [[1.0, 0.0]]
        return _response(request, vectors)

    target = Retrieval(
        embeddings=_model(),
        reranker=Model(url="https://rank.example.test/v1", name="rerank"),
        embedding_dimensions=2,
        first_stage_k=1,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await retrieve(_write_beir(tmp_path), target, client=client)

    assert ranking_bodies[0]["query"] == {"text": "alpha?"}
    assert results["q1"] == {"d1": 4.2}
