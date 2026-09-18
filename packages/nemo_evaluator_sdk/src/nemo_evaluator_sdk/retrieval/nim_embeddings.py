# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible NVIDIA NIM embedding client."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Literal
from urllib.parse import urlparse, urlunparse

import httpx
from nemo_evaluator_sdk.constants import PLACEHOLDER_INFERENCE_API_KEY
from nemo_evaluator_sdk.retrieval.http_retry import backoff_seconds, is_retryable_error
from nemo_evaluator_sdk.values.models import Model
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["NimEmbeddingClient", "NimEmbeddingError"]

InputType = Literal["query", "passage"]
logger = logging.getLogger(__name__)


class NimEmbeddingError(RuntimeError):
    """Raised when NIM returns unusable embeddings."""


class NimEmbeddingClient(BaseModel):
    """Encode query or passage batches using a NIM embedding endpoint."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Model
    dimensions: int | None = Field(default=None, gt=0)
    max_retries: int = Field(default=3, ge=0)
    timeout: float = Field(default=60.0, gt=0)

    async def encode(
        self,
        inputs: list[str],
        input_type: InputType,
        client: httpx.AsyncClient | None = None,
    ) -> list[list[float]]:
        """Encode ``inputs``, retrying transient HTTP failures and non-finite values."""
        if not inputs:
            return []

        owns_client = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post(
                        _embeddings_url(self.model.url),
                        headers=_headers(self.model),
                        json={
                            "model": self.model.name,
                            "input": inputs,
                            "input_type": input_type,
                            "encoding_format": "float",
                            "modality": "text",
                        },
                    )
                    response.raise_for_status()
                except httpx.HTTPStatusError as error:
                    body = (error.response.text or "").strip()
                    logger.warning(
                        f"embedding HTTP {error.response.status_code} attempt {attempt + 1}/{self.max_retries + 1} "
                        f"for {error.request.url}: {body or error}"
                    )
                    if not is_retryable_error(error) or attempt >= self.max_retries:
                        raise NimEmbeddingError(
                            f"embedding HTTP {error.response.status_code} after {attempt + 1} attempts: {body or error}"
                        ) from error
                    await asyncio.sleep(backoff_seconds(attempt))
                    continue
                except httpx.TransportError as error:
                    logger.warning(
                        f"embedding transport error attempt {attempt + 1}/{self.max_retries + 1} "
                        f"for {_embeddings_url(self.model.url)}: {error}"
                    )
                    if attempt >= self.max_retries:
                        raise
                    await asyncio.sleep(backoff_seconds(attempt))
                    continue
                embeddings = _parse_embeddings(response, expected_count=len(inputs), dimensions=self.dimensions)
                if embeddings and self.dimensions is None:
                    self.dimensions = len(embeddings[0])
                if all(math.isfinite(value) for embedding in embeddings for value in embedding):
                    return embeddings
                if attempt < self.max_retries:
                    await asyncio.sleep(backoff_seconds(attempt))
        finally:
            if owns_client:
                await client.aclose()

        raise NimEmbeddingError(f"embedding endpoint returned non-finite values after {self.max_retries + 1} attempts")


def _embeddings_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    if not path.endswith("/embeddings"):
        path = f"{path}/embeddings"
    return urlunparse(parsed._replace(path=path))


def _headers(model: Model) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {model.api_key or PLACEHOLDER_INFERENCE_API_KEY}",
        "Content-Type": "application/json",
        **(model.default_headers or {}),
    }


def _parse_embeddings(
    response: httpx.Response,
    expected_count: int,
    dimensions: int | None,
) -> list[list[float]]:
    try:
        data = response.json()["data"]
        ordered = sorted(data, key=lambda item: item["index"])
        if [item["index"] for item in ordered] != list(range(expected_count)):
            raise NimEmbeddingError("embedding endpoint returned invalid indexes")
        embeddings = [item["embedding"] for item in ordered]
    except (KeyError, TypeError, ValueError) as error:
        raise NimEmbeddingError("embedding endpoint returned an invalid response") from error
    if len(embeddings) != expected_count:
        raise NimEmbeddingError(f"expected {expected_count} embeddings, received {len(embeddings)}")
    expected_width = dimensions
    for embedding in embeddings:
        if not isinstance(embedding, list):
            raise NimEmbeddingError(
                f"expected embedding dimension {expected_width}, received {type(embedding).__name__}"
            )
        if expected_width is None:
            expected_width = len(embedding)
        elif len(embedding) != expected_width:
            raise NimEmbeddingError(f"expected embedding dimension {expected_width}, received {len(embedding)}")
        if not all(isinstance(value, int | float) and not isinstance(value, bool) for value in embedding):
            raise NimEmbeddingError("embedding values must be numbers")
    return embeddings
