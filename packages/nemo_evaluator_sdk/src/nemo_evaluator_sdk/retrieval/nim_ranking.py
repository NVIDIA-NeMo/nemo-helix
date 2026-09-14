# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA NIM reranking client."""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

import httpx
from nemo_evaluator_sdk.constants import PLACEHOLDER_INFERENCE_API_KEY
from nemo_evaluator_sdk.values.models import Model, RankingContract, RankingInference
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

__all__ = ["NimRankingClient", "NimRankingError"]


class NimRankingError(RuntimeError):
    """Raised when NIM returns an unusable ranking response."""


@dataclass(frozen=True, slots=True)
class _RankingCandidate:
    contract: RankingContract
    url: str


class _UnsupportedRankingCandidate(Exception):
    """A candidate route or schema is not supported by the endpoint."""

    def __init__(self, candidate: _RankingCandidate, status_code: int) -> None:
        self.candidate = candidate
        self.status_code = status_code
        super().__init__(f"{candidate.contract} at {candidate.url}: HTTP {status_code}")


class NimRankingClient(BaseModel):
    """Rank query/passage pairs using an explicit or discovered ranking contract."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Model
    max_retries: int = Field(default=3, ge=0)
    timeout: float = Field(default=60.0, gt=0)
    _resolved_contract: RankingContract | None = PrivateAttr(default=None)
    _resolved_path: str | None = PrivateAttr(default=None)

    @property
    def resolved_contract(self) -> RankingContract | None:
        """Contract selected by the most recent successful request."""
        return self._resolved_contract

    @property
    def resolved_path(self) -> str | None:
        """Route selected by the most recent successful request."""
        return self._resolved_path

    async def preflight(self, client: httpx.AsyncClient | None = None) -> Model:
        """Probe one query/passage and return a model stamped with the working contract."""
        await self.rank("reranker compatibility probe", ["reranker compatibility probe"], client=client)
        assert self._resolved_contract is not None
        assert self._resolved_path is not None
        return self.model.model_copy(
            update={
                "inference": RankingInference(
                    type="ranking",
                    contract=self._resolved_contract,
                    path=self._resolved_path,
                )
            }
        )

    async def rank(
        self,
        query: str,
        passages: list[str],
        client: httpx.AsyncClient | None = None,
        truncate: str = "END",
    ) -> list[tuple[int, float]]:
        """Return ``(passage_index, logit)`` pairs in NIM ranking order."""
        if not passages:
            return []

        owns_client = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)
        try:
            rejected: list[_UnsupportedRankingCandidate] = []
            for candidate in _ranking_candidates(self.model):
                try:
                    ranked = await self._rank_candidate(
                        candidate,
                        query=query,
                        passages=passages,
                        truncate=truncate,
                        client=client,
                    )
                except _UnsupportedRankingCandidate as error:
                    rejected.append(error)
                    continue
                self._resolved_contract = candidate.contract
                self._resolved_path = _relative_ranking_path(self.model.url, candidate.url)
                return ranked
        finally:
            if owns_client:
                await client.aclose()

        if rejected:
            attempts = "; ".join(str(error) for error in rejected)
            raise NimRankingError(f"no compatible ranking contract; attempted {attempts}")
        raise NimRankingError(f"ranking endpoint returned non-finite values after {self.max_retries + 1} attempts")

    async def _rank_candidate(
        self,
        candidate: _RankingCandidate,
        *,
        query: str,
        passages: list[str],
        truncate: str,
        client: httpx.AsyncClient,
    ) -> list[tuple[int, float]]:
        for attempt in range(self.max_retries + 1):
            response = await client.post(
                candidate.url,
                headers=_headers(self.model),
                json=_ranking_payload(candidate.contract, self.model, query, passages, truncate),
            )
            if response.is_error:
                effective_status = _effective_status_code(response)
                if effective_status in {400, 404, 405, 422}:
                    raise _UnsupportedRankingCandidate(candidate, effective_status)
                response.raise_for_status()
            ranked = _parse_rankings(response, expected_count=len(passages))
            if all(math.isfinite(score) for _, score in ranked):
                return ranked
            if attempt < self.max_retries:
                await asyncio.sleep(min(0.1 * 2**attempt, 1.0))
        raise NimRankingError(f"{candidate.contract} returned non-finite values after {self.max_retries + 1} attempts")


def _ranking_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/embeddings"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    if path.endswith("/reranking"):
        path = f"{path[: -len('/reranking')]}/ranking"
    if not path.endswith("/ranking"):
        path = f"{path}/ranking"
    return urlunparse(parsed._replace(path=path))


def _base_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/embeddings", "/ranking", "/rerank"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunparse(parsed._replace(path=path))


def _route_url(base_url: str, path: str) -> str:
    parsed = urlparse(base_url)
    route = f"{parsed.path.rstrip('/')}/{path.lstrip('/')}"
    return urlunparse(parsed._replace(path=route))


def _default_path(contract: RankingContract, served_model_name: str | None) -> str | None:
    if contract == "nim-ranking-v1":
        return "/ranking"
    if contract == "hosted-rerank-v1":
        return "/rerank"
    if served_model_name is None:
        return None
    if contract == "hosted-ranking-v1":
        return f"/ranking/{served_model_name}"
    return f"/retrieval/{served_model_name}/reranking"


def _ranking_candidates(model: Model) -> list[_RankingCandidate]:
    base = _base_url(model.url)
    ranking = model.inference if isinstance(model.inference, RankingInference) else None
    if ranking is not None and ranking.contract is not None:
        path = ranking.path or _default_path(ranking.contract, model.served_model_name)
        if path is None:
            raise NimRankingError(f"{ranking.contract} requires served_model_name or an explicit ranking path")
        return [_RankingCandidate(ranking.contract, _route_url(base, path))]

    candidates = [
        _RankingCandidate("nim-ranking-v1", _ranking_url(model.url)),
        _RankingCandidate("hosted-rerank-v1", _route_url(base, "/rerank")),
    ]
    if model.served_model_name:
        candidates.extend(
            [
                _RankingCandidate(
                    "hosted-ranking-v1",
                    _route_url(base, f"/ranking/{model.served_model_name}"),
                ),
                _RankingCandidate(
                    "hosted-retrieval-reranking-v1",
                    _route_url(base, f"/retrieval/{model.served_model_name}/reranking"),
                ),
            ]
        )
    return candidates


def _ranking_payload(
    contract: RankingContract,
    model: Model,
    query: str,
    passages: list[str],
    truncate: str,
) -> dict[str, object]:
    model_name = model.served_model_name or model.name
    if contract in {"nim-ranking-v1", "hosted-retrieval-reranking-v1"}:
        return {
            "model": model.name if contract == "nim-ranking-v1" else model_name,
            "query": {"text": query},
            "passages": [{"text": passage} for passage in passages],
            "truncate": truncate,
        }
    return {
        "model": model_name,
        "query": query,
        "documents": passages,
    }


def _effective_status_code(response: httpx.Response) -> int:
    """Recover an upstream status wrapped by Inference Gateway as HTTP 502."""
    if response.status_code != 502:
        return response.status_code
    match = re.search(r"Backend returned (\d{3})", response.text)
    return int(match.group(1)) if match else response.status_code


def _relative_ranking_path(model_url: str, ranking_url: str) -> str:
    base_path = urlparse(_base_url(model_url)).path.rstrip("/")
    ranking_path = urlparse(ranking_url).path
    if ranking_path.startswith(base_path):
        return ranking_path[len(base_path) :] or "/"
    return ranking_path


def _headers(model: Model) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {model.api_key or PLACEHOLDER_INFERENCE_API_KEY}",
        "Content-Type": "application/json",
        **(model.default_headers or {}),
    }


def _parse_rankings(response: httpx.Response, expected_count: int) -> list[tuple[int, float]]:
    try:
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("ranking response must be an object")
        rankings = payload.get("rankings", payload.get("results"))
        if not isinstance(rankings, list):
            raise TypeError("ranking response must include a rankings list")
        parsed = [
            (
                int(item["index"]),
                float(item.get("logit", item.get("score", item.get("relevance_score")))),
            )
            for item in rankings
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise NimRankingError("ranking endpoint returned an invalid response") from error
    if len(parsed) != expected_count:
        raise NimRankingError(f"expected {expected_count} rankings, received {len(parsed)}")
    return parsed
