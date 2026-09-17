# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact dense retrieval for small evaluation corpora."""

from __future__ import annotations

import asyncio
import logging
import math

import httpx
import numpy as np
from nemo_evaluator_sdk.retrieval.beir import BeirDataset
from nemo_evaluator_sdk.retrieval.nim_embeddings import InputType, NimEmbeddingClient
from nemo_evaluator_sdk.retrieval.nim_ranking import NimRankingClient
from nemo_evaluator_sdk.retrieval.passages import Truncation, passage_text
from nemo_evaluator_sdk.values.retrieval import Retrieval

__all__ = ["dense_search", "retrieve"]

logger = logging.getLogger(__name__)


# Cells in one query-chunk score block, bounding it to ~256 MB of float32.
_SCORE_BLOCK_CELLS = 64_000_000


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
    logger.info(
        f"dense search {embeddings.model.name}: {len(document_ids)} passages, {len(query_ids)} queries, "
        f"batch_size={batch_size}, in_flight={in_flight}"
    )
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

    return await asyncio.to_thread(
        _score,
        document_ids,
        document_vectors,
        query_ids,
        query_vectors,
        top_k,
        embeddings.model.name,
    )


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
    n_batches = len(batches)
    model_name = embeddings.model.name
    logger.info(
        f"encoding {input_type} for {model_name}: {len(texts)} texts in {n_batches} batches "
        f"(batch_size={batch_size}, in_flight={in_flight})"
    )
    semaphore = asyncio.Semaphore(in_flight)
    progress_every = max(10, math.ceil(n_batches * 0.05))
    completed = 0
    completed_lock = asyncio.Lock()

    async def _encode(index: int, batch: list[str]) -> list[list[float]]:
        nonlocal completed
        offset = index * batch_size
        chars = sum(len(text) for text in batch)
        async with semaphore:
            logger.debug(
                f"encode {model_name} {input_type} batch {index + 1}/{n_batches} "
                f"offset={offset} n={len(batch)} chars={chars}"
            )
            try:
                result = await embeddings.encode(batch, input_type=input_type, client=client)
            except Exception as error:
                logger.error(
                    f"encode {model_name} {input_type} failed batch {index + 1}/{n_batches} "
                    f"offset={offset} n={len(batch)} chars={chars}: {error}"
                )
                raise
            async with completed_lock:
                completed += 1
                if completed == n_batches or completed % progress_every == 0:
                    logger.info(
                        f"encoded {model_name} {input_type} {completed}/{n_batches} batches "
                        f"({100 * completed / n_batches:.0f}%)"
                    )
            return result

    tasks = [asyncio.create_task(_encode(index, batch)) for index, batch in enumerate(batches)]
    try:
        encoded = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    vectors: list[list[float]] = []
    for part in encoded:
        vectors.extend(part)
    logger.info(f"finished encoding {input_type} for {model_name}: {len(texts)} texts")
    return vectors


def _score(
    document_ids: list[str],
    document_vectors: list[list[float]],
    query_ids: list[str],
    query_vectors: list[list[float]],
    top_k: int | None,
    model_name: str,
) -> dict[str, dict[str, float]]:
    """Cosine-score every query against the corpus, one chunk of queries at a time.

    Called on a worker thread: the matmul is the bulk of the work and BLAS both releases
    the GIL and spreads it over every core, so a concurrent dense search keeps embedding
    while this runs.
    """
    documents = _unit_rows(document_vectors)
    queries = _unit_rows(query_vectors)
    n_documents = len(document_ids)
    n_queries = len(query_ids)
    k = n_documents if top_k is None else min(top_k, n_documents)
    # Rank of each id in lexicographic order, so score ties break on document id.
    document_rank = np.argsort(np.argsort(np.asarray(document_ids), kind="stable")).astype(np.int32)
    chunk = max(1, _SCORE_BLOCK_CELLS // n_documents)
    logger.info(f"scoring {model_name}: {n_queries} queries x {n_documents} passages, top_k={k}, query_chunk={chunk}")

    results: dict[str, dict[str, float]] = {}
    progress_every = max(1, math.ceil(math.ceil(n_queries / chunk) * 0.1))
    for number, start in enumerate(range(0, n_queries, chunk), start=1):
        scores = queries[start : start + chunk] @ documents.T
        for row, query_id in enumerate(query_ids[start : start + chunk]):
            row_scores = scores[row]
            if k < n_documents:
                # Partition gives the k-th best score, but splits ties arbitrarily. Widening to
                # every document at that score keeps the whole boundary tie group, so the sort
                # below breaks it on document id rather than on corpus order.
                cutoff_index = n_documents - k
                cutoff = row_scores[np.argpartition(row_scores, cutoff_index)[cutoff_index]]
                candidates = np.flatnonzero(row_scores >= cutoff)
            else:
                candidates = np.arange(n_documents)
            ordered = candidates[np.lexsort((document_rank[candidates], -row_scores[candidates]))][:k]
            results[query_id] = {document_ids[index]: float(row_scores[index]) for index in ordered}
        if number % progress_every == 0 or len(results) == n_queries:
            logger.info(
                f"scored {model_name} {len(results)}/{n_queries} queries ({100 * len(results) / n_queries:.0f}%)"
            )
    return results


def _unit_rows(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not norms.all():
        raise ValueError("cannot search with a zero-length embedding")
    return matrix / norms
