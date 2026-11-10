# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-owned source paths and authenticated embeddings for Compass analysis."""

import asyncio
import json
from collections.abc import AsyncGenerator
from concurrent.futures import Future
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from compass.analysis import LiteLLMEmbeddingConfig, UserEmbeddingGenerator
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.types import PreparedRequest, RetryPolicy
from nemo_helix_plugin.models.client import AsyncModelsClient
from nemo_helix_plugin.models.refs import parse_workspace_name_ref
from nemo_helix_plugin.nooa_model_client import supported_backend_format
from pydantic import BaseModel


def resolve_code_base(configured: str | None, workspace: str | Path | None) -> Path | None:
    """Resolve an explicitly configured source directory within Fabric's workspace.

    An empty workspace is not evidence that source was supplied. Callers opt in
    through ``code_base`` after staging the source fileset as the job workdir.
    """
    if configured is None:
        return None
    if workspace is None:
        raise ValueError("Compass code_base requires a Fabric workspace containing the staged source")
    root = Path(workspace).resolve()
    relative = Path(configured)
    if relative.is_absolute():
        raise ValueError("Compass code_base must be relative to the Fabric workspace")
    code_base = (root / relative).resolve()
    if not code_base.is_relative_to(root):
        raise ValueError("Compass code_base must remain within the Fabric workspace")
    if not code_base.is_dir():
        raise ValueError(f"Compass code_base is not an existing directory: {configured}")
    return code_base


class _EmbeddingRow(BaseModel):
    index: int
    embedding: list[float]


class _EmbeddingResponse(BaseModel):
    data: list[_EmbeddingRow]


@asynccontextmanager
async def platform_embeddings(
    client: AsyncNemoClient,
    model_ref: str,
    *,
    config: LiteLLMEmbeddingConfig | None = None,
) -> AsyncGenerator[UserEmbeddingGenerator]:
    """Use a Platform embedding model while Compass owns preprocessing and batching.

    The selected model must serve Qwen3-Embedding-8B's 4096-dimensional vectors,
    which Compass's sentiment classifier expects. A worker's synchronous batch
    callback sends its request on the client's owning event loop, preserving
    token refresh and transport lifetime. No credentials enter saved options.
    """
    models_client = AsyncModelsClient.from_client(client)
    workspace, name = parse_workspace_name_ref(model_ref, label="Embedding model")
    entity = (await models_client.get_model(name=name, workspace=workspace)).data()
    if supported_backend_format(entity) != "OPENAI_CHAT":
        raise ValueError("Embedding models must expose an OpenAI-compatible embeddings endpoint")
    resolved = await models_client.resolve_model_reference(model_ref)
    base_url = str(client.base_url).rstrip("/")
    if not resolved.url.startswith(f"{base_url}/"):
        raise ValueError("Embedding model must resolve through this Helix platform")
    endpoint = resolved.url.removeprefix(base_url).rstrip("/") + "/embeddings"
    options = config or LiteLLMEmbeddingConfig(model=model_ref)
    transport = _EmbeddingTransport(client, endpoint, resolved.name, options)
    try:
        yield UserEmbeddingGenerator(litellm=options, embed_batch=transport.embed_batch)
    finally:
        await transport.aclose()


class _EmbeddingTransport:
    """Bridge Compass's synchronous workers to the host-owned async client."""

    def __init__(self, client: AsyncNemoClient, endpoint: str, model: str, options: LiteLLMEmbeddingConfig) -> None:
        self.client = client.with_options(timeout=options.request_timeout)
        self.endpoint = endpoint
        self.model = model
        self.retry = RetryPolicy(max_retries=options.max_retries)
        self.loop = asyncio.get_running_loop()
        self.batches: set[Future[list[list[float]]]] = set()
        self.requests: set[asyncio.Task] = set()
        self.lock = Lock()
        self.closed = False

    async def _send(self, inputs: list[str]) -> list[list[float]]:
        response = await self.client.send(
            PreparedRequest(
                method="POST",
                path_template=self.endpoint,
                path_params={},
                content=json.dumps({"model": self.model, "input": inputs, "encoding_format": "float"}).encode(),
                content_type="application/json",
                response_type=_EmbeddingResponse,
                extra_headers={"accept-encoding": "identity"},
            ),
            retry=self.retry,
        )
        rows = sorted(response.data().data, key=lambda row: row.index)
        if [row.index for row in rows] != list(range(len(inputs))):
            raise ValueError("Embedding response must contain one vector per input")
        return [row.embedding for row in rows]

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        if self.closed:
            raise RuntimeError("Compass embedding transport is closed")
        task = asyncio.current_task()
        assert task is not None
        self.requests.add(task)
        try:
            return await self._send(inputs)
        finally:
            self.requests.discard(task)

    def embed_batch(self, inputs: list[str]) -> list[list[float]]:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self.loop:
            raise RuntimeError("Compass embedding batches must run in a worker thread")
        with self.lock:
            if self.closed:
                raise RuntimeError("Compass embedding transport is closed")
            batch = asyncio.run_coroutine_threadsafe(self._embed(inputs), self.loop)
            self.batches.add(batch)
        try:
            return batch.result()
        finally:
            with self.lock:
                self.batches.discard(batch)

    async def aclose(self) -> None:
        # Cancelling analysis does not stop its to_thread worker. Stop queued
        # batches and await active requests before the caller closes its client.
        with self.lock:
            self.closed = True
            for batch in self.batches:
                batch.cancel()
        active = tuple(self.requests)
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
