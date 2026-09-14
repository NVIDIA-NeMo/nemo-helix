# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math

import httpx
import pytest
from nemo_evaluator_sdk.retrieval.nim_ranking import NimRankingClient, NimRankingError
from nemo_evaluator_sdk.values.models import Model, RankingInference


@pytest.mark.asyncio
async def test_ranking_client_posts_v1_ranking_and_accepts_logits() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            request=request,
            content=json.dumps({"rankings": [{"index": 1, "logit": 0.2}, {"index": 0, "logit": 0.9}]}),
            headers={"content-type": "application/json"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ranked = await NimRankingClient(model=Model(url="https://rank.example.test/v1", name="rerank")).rank(
            "q",
            ["first", "second"],
            client=client,
        )

    assert ranked == [(1, 0.2), (0, 0.9)]
    assert requests[0].url == "https://rank.example.test/v1/ranking"
    assert json.loads(requests[0].content)["passages"] == [{"text": "first"}, {"text": "second"}]


@pytest.mark.asyncio
async def test_ranking_client_rewrites_reranking_and_embeddings_routes() -> None:
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(
            200,
            request=request,
            content=json.dumps({"results": [{"index": 0, "score": 1.0}]}),
            headers={"content-type": "application/json"},
        )

    model = Model(url="https://igw.example.test/v1/embeddings", name="rerank")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await NimRankingClient(model=model).rank("q", ["only"], client=client)

    assert urls == ["https://igw.example.test/v1/ranking"]


@pytest.mark.asyncio
async def test_ranking_client_falls_back_to_hosted_rerank_contract() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/ranking"):
            return httpx.Response(
                502,
                request=request,
                json={"detail": 'Backend returned 404: {"detail":"Not Found"}'},
            )
        return httpx.Response(
            200,
            request=request,
            json={"results": [{"index": 0, "relevance_score": 0.75}]},
        )

    ranker = NimRankingClient(
        model=Model(
            url="https://igw.example.test/model/reranker/-/v1",
            name="flattened-reranker",
            served_model_name="publisher/hosted-reranker",
        ),
        max_retries=0,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ranked = await ranker.rank("query", ["document"], client=client)

    assert ranked == [(0, 0.75)]
    assert [request.url.path for request in requests] == [
        "/model/reranker/-/v1/ranking",
        "/model/reranker/-/v1/rerank",
    ]
    assert json.loads(requests[1].content) == {
        "model": "publisher/hosted-reranker",
        "query": "query",
        "documents": ["document"],
    }
    assert ranker.resolved_contract == "hosted-rerank-v1"
    assert ranker.resolved_path == "/rerank"


@pytest.mark.asyncio
async def test_ranking_client_falls_back_to_model_specific_ranking_route() -> None:
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        if request.url.path.endswith("/ranking") or request.url.path.endswith("/rerank"):
            return httpx.Response(404, request=request)
        return httpx.Response(
            200,
            request=request,
            json={"results": [{"index": 0, "relevance_score": 0.8}]},
        )

    model = Model(
        url="https://igw.example.test/v1",
        name="reranker",
        served_model_name="nvidia/qwen3-vl-reranker-8b",
    )
    ranker = NimRankingClient(model=model, max_retries=0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resolved = await ranker.preflight(client=client)

    assert urls[-1] == "https://igw.example.test/v1/ranking/nvidia/qwen3-vl-reranker-8b"
    assert isinstance(resolved.inference, RankingInference)
    assert resolved.inference.contract == "hosted-ranking-v1"
    assert resolved.inference.path == "/ranking/nvidia/qwen3-vl-reranker-8b"


@pytest.mark.asyncio
async def test_ranking_client_falls_back_to_model_specific_retrieval_route() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if "/retrieval/" not in request.url.path:
            return httpx.Response(
                502,
                request=request,
                json={"detail": 'Backend returned 404: {"detail":"Not Found"}'},
            )
        return httpx.Response(
            200,
            request=request,
            json={"results": [{"index": 0, "relevance_score": 0.85}]},
        )

    model = Model(
        url="https://igw.example.test/v1",
        name="reranker",
        served_model_name="publisher/reranker",
    )
    ranker = NimRankingClient(model=model, max_retries=0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resolved = await ranker.preflight(client=client)

    assert [request.url.path for request in requests] == [
        "/v1/ranking",
        "/v1/rerank",
        "/v1/ranking/publisher/reranker",
        "/v1/retrieval/publisher/reranker/reranking",
    ]
    assert json.loads(requests[-1].content) == {
        "model": "publisher/reranker",
        "query": {"text": "reranker compatibility probe"},
        "passages": [{"text": "reranker compatibility probe"}],
        "truncate": "END",
    }
    assert isinstance(resolved.inference, RankingInference)
    assert resolved.inference.contract == "hosted-retrieval-reranking-v1"
    assert resolved.inference.path == "/retrieval/publisher/reranker/reranking"


@pytest.mark.asyncio
async def test_ranking_client_does_not_fall_back_after_auth_failure() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            502,
            request=request,
            json={"detail": "Backend returned 401: invalid credential"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await NimRankingClient(
                model=Model(
                    url="https://igw.example.test/v1",
                    name="reranker",
                    served_model_name="publisher/reranker",
                ),
                max_retries=0,
            ).rank("q", ["only"], client=client)

    assert attempts == 1


@pytest.mark.asyncio
async def test_ranking_client_retries_non_finite_logits() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        logit = math.nan if attempts == 1 else 0.9
        return httpx.Response(
            200,
            request=request,
            content=json.dumps({"rankings": [{"index": 0, "logit": logit}]}),
            headers={"content-type": "application/json"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ranked = await NimRankingClient(
            model=Model(url="https://rank.example.test/v1", name="rerank"),
            max_retries=1,
        ).rank("q", ["only"], client=client)

    assert attempts == 2
    assert ranked == [(0, 0.9)]


@pytest.mark.asyncio
async def test_ranking_client_uses_explicit_ranking_inference() -> None:
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(
            200,
            request=request,
            json={"results": [{"index": 0, "relevance_score": 0.5}]},
        )

    model = Model(
        url="https://igw.example.test/v1",
        name="reranker",
        served_model_name="publisher/reranker",
        inference=RankingInference(contract="hosted-rerank-v1", path="/rerank"),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await NimRankingClient(model=model, max_retries=0).rank("q", ["only"], client=client)

    assert urls == ["https://igw.example.test/v1/rerank"]


@pytest.mark.asyncio
async def test_ranking_client_rejects_non_object_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            content=json.dumps([]),
            headers={"content-type": "application/json"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(NimRankingError, match="invalid response"):
            await NimRankingClient(model=Model(url="https://rank.example.test/v1", name="rerank")).rank(
                "q",
                ["only"],
                client=client,
            )
