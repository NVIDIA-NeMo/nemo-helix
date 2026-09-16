# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact dense retrieval for small evaluation corpora."""

from __future__ import annotations

import asyncio
import math

import httpx
from nemo_evaluator_sdk.retrieval.beir import BeirDataset
from nemo_evaluator_sdk.retrieval.nim_embeddings import InputType, NimEmbeddingClient
from nemo_evaluator_sdk.retrieval.nim_ranking import NimRankingClient
from nemo_evaluator_sdk.retrieval.passages import Truncation, passage_text
from nemo_evaluator_sdk.values.retrieval import Retrieval

__all__ = ["dense_search", "retrieve"]


async def retrieve(
    dataset: BeirDataset,
    target: Retrieval,
    client: httpx.AsyncClient | None = None,
) -> dict[str, dict[str, float]]:
    """Encode the corpus, dense-search to ``first_stage_k``, then optionally rerank."""
    passages = {
        document_id: passage_text(document, target.truncate_long_documents)
        for document_id, document in dataset.corpus.items()
    }
    embeddings = NimEmbeddingClient(model=target.embeddings, dimensions=target.embedding_dimensions)
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=embeddings.timeout)
    try:
        rankings = await dense_search(
            dataset,
            embeddings,
            passages=passages,
            batch_size=target.batch_size,
            in_flight=target.embedding_in_flight,
            top_k=target.first_stage_k,
            client=client,
        )
        if target.reranker is None:
            return rankings
        return await _rerank(dataset, target, rankings, passages, client)
    finally:
        if owns_client:
            await client.aclose()


async def dense_search(
    dataset: BeirDataset,
    embeddings: NimEmbeddingClient,
    batch_size: int = 32,
    in_flight: int = 2,
    top_k: int | None = None,
    client: httpx.AsyncClient | None = None,
    passages: dict[str, str] | None = None,
    truncate_long_documents: Truncation | None = "end",
) -> dict[str, dict[str, float]]:
    """Score every query against the corpus with cosine similarity."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if in_flight < 1:
        raise ValueError("in_flight must be at least 1")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1")

    document_ids = list(dataset.corpus)
    query_ids = list(dataset.queries)
    if passages is None:
        passages = {
            document_id: passage_text(dataset.corpus[document_id], truncate_long_documents)
            for document_id in document_ids
        }
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=embeddings.timeout)
    try:
        document_vectors = await _encode_batches(
            embeddings,
            [passages[document_id] for document_id in document_ids],
            input_type="passage",
            batch_size=batch_size,
            in_flight=in_flight,
            client=client,
        )
        query_vectors = await _encode_batches(
            embeddings,
            [dataset.queries[query_id].text for query_id in query_ids],
            input_type="query",
            batch_size=batch_size,
            in_flight=in_flight,
            client=client,
        )
    finally:
        if owns_client:
            await client.aclose()

    normalized_documents = [_normalize(vector) for vector in document_vectors]
    results: dict[str, dict[str, float]] = {}
    for query_id, query_vector in zip(query_ids, query_vectors, strict=True):
        normalized_query = _normalize(query_vector)
        ranked = sorted(
            (
                (document_id, sum(left * right for left, right in zip(normalized_query, document_vector, strict=True)))
                for document_id, document_vector in zip(document_ids, normalized_documents, strict=True)
            ),
            key=lambda item: (-item[1], item[0]),
        )
        if top_k is not None:
            ranked = ranked[:top_k]
        results[query_id] = dict(ranked)
    return results


async def _rerank(
    dataset: BeirDataset,
    target: Retrieval,
    rankings: dict[str, dict[str, float]],
    passages: dict[str, str],
    client: httpx.AsyncClient | None,
) -> dict[str, dict[str, float]]:
    if target.reranker is None:
        raise ValueError("rerank requires a reranker model")
    ranker = NimRankingClient(model=target.reranker)
    truncate = "END" if target.truncate_long_documents != "start" else "START"
    reranked: dict[str, dict[str, float]] = {}
    for query_id, scores in rankings.items():
        document_ids = list(scores)
        ranked = await ranker.rank(
            dataset.queries[query_id].text,
            [passages[document_id] for document_id in document_ids],
            client=client,
            truncate=truncate,
        )
        reranked[query_id] = {document_ids[index]: logit for index, logit in ranked}
    return reranked


async def _encode_batches(
    embeddings: NimEmbeddingClient,
    texts: list[str],
    input_type: InputType,
    batch_size: int,
    in_flight: int,
    client: httpx.AsyncClient,
) -> list[list[float]]:
    batches = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]
    if not batches:
        return []
    semaphore = asyncio.Semaphore(in_flight)

    async def _encode(batch: list[str]) -> list[list[float]]:
        async with semaphore:
            return await embeddings.encode(batch, input_type=input_type, client=client)

    encoded = await asyncio.gather(*(_encode(batch) for batch in batches))
    vectors: list[list[float]] = []
    for part in encoded:
        vectors.extend(part)
    return vectors


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("cannot search with a zero-length embedding")
    return [value / norm for value in vector]
