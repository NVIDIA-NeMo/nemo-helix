# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import math
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from nemo_evaluator_sdk.retrieval import dense_search as dense_search_module
from nemo_evaluator_sdk.retrieval.beir import BeirDataset
from nemo_evaluator_sdk.retrieval.dense_search import dense_search, retrieve
from nemo_evaluator_sdk.retrieval.nim_embeddings import InputType, NimEmbeddingClient, NimEmbeddingError
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
async def test_encode_batches_cancels_siblings_on_first_failure() -> None:
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    completed: list[str] = []

    class FailingEmbeddings(NimEmbeddingClient):
        async def encode(
            self,
            inputs: list[str],
            input_type: InputType,
            client: httpx.AsyncClient | None = None,
        ) -> list[list[float]]:
            text = inputs[0]
            if text == "bad":
                await sibling_started.wait()
                raise RuntimeError("encode failed")
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise
            completed.append(text)
            return [[1.0]]

    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError, match="encode failed"):
            await dense_search_module._encode_batches(
                FailingEmbeddings(model=_model()),
                ["bad", "slow", "queued"],
                input_type="query",
                batch_size=1,
                in_flight=2,
                client=client,
            )

    assert sibling_cancelled.is_set()
    assert completed == []


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


@pytest.mark.asyncio
async def test_dense_search_breaks_top_k_ties_on_document_id(tmp_path: Path) -> None:
    """A tie group straddling the top_k cutoff keeps the lowest document ids, not corpus order."""
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text(
        '{"_id":"d4","text":"a"}\n{"_id":"d3","text":"b"}\n{"_id":"d1","text":"c"}\n{"_id":"d2","text":"d"}\n',
        encoding="utf-8",
    )
    (tmp_path / "queries.jsonl").write_text('{"_id":"q1","text":"a?"}\n', encoding="utf-8")
    (tmp_path / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        # Every passage scores identically, so only the id tie-break orders them.
        vectors = [[1.0, 0.0] for _ in payload["input"]]
        return _response(request, vectors)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await dense_search(
            BeirDataset.from_path(tmp_path),
            NimEmbeddingClient(model=_model(), dimensions=2),
            top_k=2,
            client=client,
        )

    assert list(results["q1"]) == ["d1", "d2"]


def test_score_top_k_matches_full_deterministic_sort() -> None:
    rng = np.random.default_rng(42)
    document_ids = ["d07", "d02", "d09", "d01", "d05", "d03", "d08", "d04", "d06", "d00"]
    document_vectors = rng.normal(size=(len(document_ids), 5)).tolist()
    # Duplicate vectors exercise score ties as well as ordinary random rankings.
    document_vectors[7] = document_vectors[1]
    query_ids = ["q0", "q1", "q2"]
    query_vectors = rng.normal(size=(len(query_ids), 5)).tolist()

    documents = dense_search_module._unit_rows(document_vectors)
    queries = dense_search_module._unit_rows(query_vectors)
    scores = queries @ documents.T
    for top_k in (1, 3, 7):
        actual = dense_search_module._score(
            document_ids,
            document_vectors,
            query_ids,
            query_vectors,
            top_k,
            "test-model",
        )
        for row, query_id in enumerate(query_ids):
            expected = sorted(range(len(document_ids)), key=lambda index: (-scores[row, index], document_ids[index]))
            assert list(actual[query_id]) == [document_ids[index] for index in expected[:top_k]]


@pytest.mark.asyncio
async def test_dense_search_scores_off_the_event_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scoring must not starve a concurrently running dense search of its embedding turns."""
    dataset = _write_beir(tmp_path)
    ticks = 0

    async def tick() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.005)
            ticks += 1

    real_score = dense_search_module._score

    def slow_score(
        document_ids: list[str],
        document_vectors: list[list[float]],
        query_ids: list[str],
        query_vectors: list[list[float]],
        top_k: int | None,
        model_name: str,
    ) -> dict[str, dict[str, float]]:
        time.sleep(0.2)
        return real_score(document_ids, document_vectors, query_ids, query_vectors, top_k, model_name)

    monkeypatch.setattr(dense_search_module, "_score", slow_score)

    def handler(request: httpx.Request) -> httpx.Response:
        return _response(request, [[1.0, 0.0] for _ in json.loads(request.content)["input"]])

    ticker = asyncio.create_task(tick())
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await dense_search(dataset, NimEmbeddingClient(model=_model(), dimensions=2), client=client)
    finally:
        ticker.cancel()

    assert ticks > 5


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
