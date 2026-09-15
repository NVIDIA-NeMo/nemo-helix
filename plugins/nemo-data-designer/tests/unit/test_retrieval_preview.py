# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import data_designer.config as dd
import httpx
import pytest
from data_designer_nemo.errors import NDDInvalidConfigError
from nemo_data_designer_plugin.functions.retrieval_preview import RetrievalPreviewFrame, RetrievalPreviewFunction
from nemo_data_designer_plugin.jobs.retrieval_spec import RetrievalGenerateJobConfig, RetrievalPreviewSpec
from nemo_platform_plugin.client.errors import PermissionDeniedError
from nemo_platform_plugin.functions.frames import Done, Error


def _generate_config(tmp_path) -> RetrievalGenerateJobConfig:
    return RetrievalGenerateJobConfig(
        corpus=str(tmp_path),
        provider="default/nvidia-build",
        artifact_extraction_model="nvidia/nemotron-3-nano-30b-a3b",
        qa_generation_model="nvidia/nemotron-3-nano-30b-a3b",
        quality_judge_model="nvidia/nemotron-3-nano-30b-a3b",
        embed_model="nvidia/nemotron-3-embed-1b",
    )


@pytest.mark.asyncio
async def test_retrieval_preview_uses_preview_generation(tmp_path) -> None:
    spec = RetrievalPreviewSpec(
        generate=_generate_config(tmp_path),
        num_records=1,
    )
    (tmp_path / "doc.txt").write_text("hello", encoding="utf-8")
    providers = [dd.ModelProvider(name="default/nvidia-build", endpoint="http://igw")]
    dd_ctx = AsyncMock()
    dd_ctx.get_model_providers = AsyncMock(return_value=providers)
    ctx = Mock()
    ctx.workspace = "default"
    preview_result = SimpleNamespace(num_seed_records=1, num_preview_records=1)
    frames = []
    with (
        patch(
            "nemo_data_designer_plugin.functions.retrieval_preview.create_validation_context",
            return_value=dd_ctx,
        ),
        patch(
            "nemo_data_designer_plugin.retrieval.generation.execute_generation",
            return_value=preview_result,
        ) as execute,
    ):
        async for frame in RetrievalPreviewFunction().run(
            spec,
            ctx=ctx,
            sdk=Mock(),
            async_sdk=AsyncMock(),
            is_local=True,
        ):
            frames.append(frame)
    execute.assert_called_once()
    assert execute.call_args.kwargs["preview"] is True
    assert any(isinstance(frame, RetrievalPreviewFrame) for frame in frames)
    assert isinstance(frames[-1], Done)


@pytest.mark.asyncio
async def test_retrieval_preview_returns_error_frame_for_worker_failure(tmp_path) -> None:
    spec = RetrievalPreviewSpec(
        generate=_generate_config(tmp_path),
    )
    providers = [dd.ModelProvider(name="default/nvidia-build", endpoint="http://igw")]
    dd_ctx = AsyncMock()
    dd_ctx.get_model_providers = AsyncMock(return_value=providers)
    ctx = Mock(workspace="default")

    with (
        patch(
            "nemo_data_designer_plugin.functions.retrieval_preview.create_validation_context",
            return_value=dd_ctx,
        ),
        patch(
            "nemo_data_designer_plugin.functions.retrieval_preview.materialize_corpus",
            side_effect=ValueError("bad corpus"),
        ),
    ):
        frames = [
            frame
            async for frame in RetrievalPreviewFunction().run(
                spec,
                ctx=ctx,
                sdk=Mock(),
                async_sdk=AsyncMock(),
                is_local=True,
            )
        ]

    assert len(frames) == 1
    assert isinstance(frames[0], Error)
    assert frames[0].message == "bad corpus"


@pytest.mark.asyncio
async def test_retrieval_preview_forwards_num_records(tmp_path) -> None:
    spec = RetrievalPreviewSpec(generate=_generate_config(tmp_path), num_records=2)
    (tmp_path / "doc.txt").write_text("hello", encoding="utf-8")
    providers = [dd.ModelProvider(name="default/nvidia-build", endpoint="http://igw")]
    dd_ctx = AsyncMock()
    dd_ctx.get_model_providers = AsyncMock(return_value=providers)
    ctx = Mock(workspace="default")
    preview_result = SimpleNamespace(num_seed_records=2, num_preview_records=2)
    frames = []
    with (
        patch(
            "nemo_data_designer_plugin.functions.retrieval_preview.create_validation_context",
            return_value=dd_ctx,
        ),
        patch(
            "nemo_data_designer_plugin.retrieval.generation.execute_generation",
            return_value=preview_result,
        ) as execute,
    ):
        async for frame in RetrievalPreviewFunction().run(
            spec, ctx=ctx, sdk=Mock(), async_sdk=AsyncMock(), is_local=True
        ):
            frames.append(frame)

    assert execute.call_args.kwargs["num_records"] == 2
    frame = frames[0]
    assert isinstance(frame, RetrievalPreviewFrame)
    assert frame.num_preview_records == 2


@pytest.mark.asyncio
async def test_retrieval_preview_resolves_hf_token_secret(tmp_path) -> None:
    config = _generate_config(tmp_path)
    config = config.model_copy(update={"corpus": "hf://example/private-corpus", "hf_token_secret": "default/hf-token"})
    spec = RetrievalPreviewSpec(generate=config)
    providers = [dd.ModelProvider(name="default/nvidia-build", endpoint="http://igw")]
    dd_ctx = AsyncMock()
    dd_ctx.get_model_providers = AsyncMock(return_value=providers)
    ctx = Mock(workspace="default")
    preview_result = SimpleNamespace(num_seed_records=1, num_preview_records=1)
    resolved: dict[str, str | None] = {}

    async def resolve(async_sdk: object, hf_token_secret: str | None, workspace: str) -> str:
        resolved.update(secret=hf_token_secret, workspace=workspace)
        return "hf_secret_value"

    with (
        patch(
            "nemo_data_designer_plugin.functions.retrieval_preview.create_validation_context",
            return_value=dd_ctx,
        ),
        patch(
            "nemo_data_designer_plugin.functions.retrieval_preview.resolve_hf_token",
            new=resolve,
        ),
        patch(
            "nemo_data_designer_plugin.functions.retrieval_preview.materialize_corpus",
            return_value=tmp_path,
        ) as materialize,
        patch(
            "nemo_data_designer_plugin.retrieval.generation.execute_generation",
            return_value=preview_result,
        ),
    ):
        frames = [
            frame
            async for frame in RetrievalPreviewFunction().run(
                spec, ctx=ctx, sdk=Mock(), async_sdk=AsyncMock(), is_local=False
            )
        ]

    assert resolved == {"secret": "default/hf-token", "workspace": "default"}
    assert materialize.call_args.kwargs["hf_token"] == "hf_secret_value"
    assert isinstance(frames[-1], Done)


@pytest.mark.asyncio
async def test_retrieval_preview_raises_invalid_config_for_unauthorized_secret(tmp_path) -> None:
    config = _generate_config(tmp_path).model_copy(
        update={"corpus": "hf://example/private-corpus", "hf_token_secret": "other/hf-token"}
    )
    spec = RetrievalPreviewSpec(generate=config)
    providers = [dd.ModelProvider(name="default/nvidia-build", endpoint="http://igw")]
    dd_ctx = AsyncMock()
    dd_ctx.get_model_providers = AsyncMock(return_value=providers)
    ctx = Mock(workspace="default")

    with (
        patch(
            "nemo_data_designer_plugin.functions.retrieval_preview.create_validation_context",
            return_value=dd_ctx,
        ),
        patch(
            "nemo_data_designer_plugin.functions.retrieval_preview.resolve_hf_token",
            new=AsyncMock(
                side_effect=PermissionDeniedError(
                    httpx.Response(403, json={"detail": "denied"}, request=httpx.Request("GET", "http://secrets"))
                )
            ),
        ),
    ):
        with pytest.raises(NDDInvalidConfigError, match="other/hf-token"):
            async for _ in RetrievalPreviewFunction().run(
                spec, ctx=ctx, sdk=Mock(), async_sdk=AsyncMock(), is_local=False
            ):
                pass


@pytest.mark.asyncio
async def test_retrieval_preview_raises_invalid_config_for_unresolvable_providers(tmp_path) -> None:
    spec = RetrievalPreviewSpec(generate=_generate_config(tmp_path))
    dd_ctx = AsyncMock()
    ctx = Mock(workspace="default")

    with (
        patch(
            "nemo_data_designer_plugin.functions.retrieval_preview.create_validation_context",
            return_value=dd_ctx,
        ),
        patch(
            "nemo_data_designer_plugin.functions.retrieval_preview.resolve_retrieval_providers",
            new=AsyncMock(side_effect=NDDInvalidConfigError("no providers")),
        ),
    ):
        with pytest.raises(NDDInvalidConfigError, match="no providers"):
            async for _ in RetrievalPreviewFunction().run(
                spec, ctx=ctx, sdk=Mock(), async_sdk=AsyncMock(), is_local=True
            ):
                pass
