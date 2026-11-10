# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

if sys.version_info >= (3, 14):
    pytest.skip("nemo-compass requires Python 3.12 or 3.13", allow_module_level=True)

from nemo_compass_plugin.analysis import inputs
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.models.refs import ResolvedModelReference


def test_code_base_requires_explicit_contained_source(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "escape").symlink_to(outside, target_is_directory=True)

    assert inputs.resolve_code_base(None, workspace) is None
    assert inputs.resolve_code_base("source", workspace) == source
    assert inputs.resolve_code_base(".", workspace) == workspace
    for path in ("../outside", "escape", str(outside), "missing"):
        with pytest.raises(ValueError):
            inputs.resolve_code_base(path, workspace)
    with pytest.raises(ValueError, match="requires a Fabric workspace"):
        inputs.resolve_code_base(".", None)


class _RotatingToken:
    def __init__(self) -> None:
        self.calls = 0

    def set_http_client(self, http_client: httpx.AsyncClient) -> None:
        pass

    async def get_access_token_async(self) -> str:
        self.calls += 1
        return f"token-{self.calls}"

    async def get_access_token_or_none_async(self) -> str:
        return await self.get_access_token_async()


def _stub_generator(monkeypatch: pytest.MonkeyPatch, *, url: str) -> None:
    models = SimpleNamespace(
        get_model=AsyncMock(return_value=SimpleNamespace(data=lambda: SimpleNamespace(backend_format="OPENAI_CHAT"))),
        resolve_model_reference=AsyncMock(
            return_value=ResolvedModelReference(url=url, name="Qwen/Qwen3-Embedding-8B", host_url=None)
        ),
    )
    monkeypatch.setattr(inputs.AsyncModelsClient, "from_client", lambda client: models)
    # Isolate transport from Compass's tokenizer/model downloads; the actual
    # generator contract is covered by Compass and the shared-pipeline tests.
    monkeypatch.setattr(inputs, "UserEmbeddingGenerator", lambda **kwargs: SimpleNamespace(**kwargs))


async def test_embedding_batches_refresh_auth_preserve_order_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_generator(monkeypatch, url="https://platform.test/prefix/models/default/qwen/-/v1")
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]})

    auth = _RotatingToken()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        async with AsyncNemoClient(
            base_url="https://platform.test/prefix",
            auth=auth,
            default_headers={"x-test": "kept"},
            http_client=http_client,
        ) as client:
            config = inputs.LiteLLMEmbeddingConfig(model="default/qwen", request_timeout=17, max_retries=0)
            async with inputs.platform_embeddings(client, "default/qwen", config=config) as generator:
                assert generator.embed_batch is not None
                for _ in range(2):
                    assert await asyncio.to_thread(generator.embed_batch, ["first", "second"]) == [[1.0], [2.0]]
            assert not http_client.is_closed
    assert [request.headers["authorization"] for request in seen] == ["Bearer token-1", "Bearer token-2"]
    assert all(
        str(request.url) == "https://platform.test/prefix/models/default/qwen/-/v1/embeddings" for request in seen
    )
    assert seen[0].headers["x-test"] == "kept"
    assert seen[0].extensions["timeout"]["read"] == 17
    assert json.loads(seen[0].content) == {
        "model": "Qwen/Qwen3-Embedding-8B",
        "input": ["first", "second"],
        "encoding_format": "float",
    }


async def test_embedding_context_cancels_workers_before_client_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_generator(monkeypatch, url="https://platform.test/models/default/qwen/-/v1")
    started, finished = asyncio.Event(), asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()
        raise AssertionError("request should have been cancelled")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        async with AsyncNemoClient(base_url="https://platform.test", http_client=http_client) as client:
            async with inputs.platform_embeddings(client, "default/qwen") as generator:
                assert generator.embed_batch is not None
                worker = asyncio.create_task(asyncio.to_thread(generator.embed_batch, ["first"]))
                await asyncio.wait_for(started.wait(), timeout=2)
            assert finished.is_set()
            assert not http_client.is_closed
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(worker, timeout=2)
            assert generator.embed_batch is not None
            with pytest.raises(RuntimeError, match="transport is closed"):
                await asyncio.to_thread(generator.embed_batch, ["after close"])


async def test_embedding_transport_rejects_foreign_route(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_generator(monkeypatch, url="https://other.test/models/default/qwen/-/v1")
    async with AsyncNemoClient(base_url="https://platform.test") as client:
        with pytest.raises(ValueError, match="through this Helix platform"):
            async with inputs.platform_embeddings(client, "default/qwen"):
                raise AssertionError("foreign route should be rejected")


@pytest.mark.parametrize("supply_inputs", [False, True])
async def test_empty_snapshot_preserves_library_capabilities_and_complete_result(
    tmp_path: Path, supply_inputs: bool
) -> None:
    """Exercise the real shared pipeline without model calls on an empty corpus."""
    from compass.analysis import AnalysisOptions, AnalysisResult, UserEmbeddingGenerator, analyze
    from compass.traces import TraceSnapshot
    from nemo_compass_plugin.analysis.pipeline import analyze_snapshot
    from nemo_helix_plugin.nooa_model_client import ConfiguredModelClients
    from nooa.unifiedllm import FakeLLMClient

    snapshot = TraceSnapshot([])
    options = AnalysisOptions()
    ethos = "# Intent\nAnswer questions accurately." if supply_inputs else None
    ethos_path = tmp_path / "ETHOS.md" if supply_inputs else None
    if ethos_path:
        assert ethos is not None
        ethos_path.write_text(ethos)
    generator = (
        UserEmbeddingGenerator(
            litellm=inputs.LiteLLMEmbeddingConfig(model="default/qwen"), embed_batch=lambda texts: []
        )
        if supply_inputs
        else None
    )
    direct = await analyze(
        snapshot,
        llm_factory=FakeLLMClient,
        options=options,
        code_base=tmp_path if supply_inputs else None,
        ethos_path=ethos_path,
        embedding_generator=generator,
    )
    projected = await analyze_snapshot(
        snapshot,
        existing=[],
        model_clients=ConfiguredModelClients(default=FakeLLMClient(), fast=FakeLLMClient()),
        ethos=ethos,
        analysis_options=options.model_dump(mode="json"),
        code_base=tmp_path if supply_inputs else None,
        embedding_generator=generator,
    )
    restored = AnalysisResult.model_validate(projected.analysis_result)
    assert restored == direct
    assert not projected.new_insights
    assert {summary.signal_name for summary in restored.signal_summaries} == {
        summary.signal_name for summary in direct.signal_summaries
    }
    if not supply_inputs:
        assert any(
            summary.signal_name == "user-sentiment" and summary.skip_reason for summary in restored.signal_summaries
        )
        assert any(stage.stage == "confidence" and stage.reason == "No codebase supplied" for stage in restored.stages)
