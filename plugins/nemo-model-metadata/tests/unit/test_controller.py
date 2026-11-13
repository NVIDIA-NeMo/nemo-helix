# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.models.types import (
    APIEndpointData,
    ModelCapability,
    ModelEntity,
    ModelMetadata,
    ModelProvider,
    ServedModelMapping,
)
from nemo_model_metadata.config import ModelMetadataConfig
from nemo_model_metadata.controller import ModelMetadataController, build_http_client, upstream_model_id
from nemo_model_metadata.merge import merge_metadata
from nemo_model_metadata.sources.base import MetadataSource, MetadataTarget

NOW = datetime(2026, 11, 10, tzinfo=timezone.utc)


class FakeSource(MetadataSource):
    def __init__(self, name: str, result: ModelMetadata | None) -> None:
        self.name = name
        self.result = result
        self.targets: list[MetadataTarget] = []

    async def fetch(self, target: MetadataTarget) -> ModelMetadata | None:
        self.targets.append(target)
        return self.result


class FailingSource(MetadataSource):
    name = "broken"

    async def fetch(self, target: MetadataTarget) -> ModelMetadata | None:
        raise RuntimeError("boom")


def _model(metadata: ModelMetadata | None = None, providers: list[str] | None = None) -> ModelEntity:
    return ModelEntity(
        id="model-llama",
        created_at=NOW,
        updated_at=NOW,
        name="llama",
        workspace="default",
        model_providers=["default/nim"] if providers is None else providers,
        metadata=metadata,
    )


def _provider() -> ModelProvider:
    return ModelProvider(
        created_at=NOW,
        updated_at=NOW,
        name="nim",
        workspace="default",
        host_url="https://integrate.api.nvidia.com/v1",
        served_models=[
            ServedModelMapping(model_entity_id="default/llama", served_model_name="meta/llama-3.1-8b-instruct")
        ],
    )


def _controller(sources: list[MetadataSource]) -> tuple[ModelMetadataController, MagicMock]:
    controller = ModelMetadataController()
    models_client = MagicMock()
    models_client.update_model = AsyncMock()
    controller._config = ModelMetadataConfig()
    controller._models_client = models_client
    controller._sources = sources
    return controller, models_client


def test_merge_takes_each_field_from_first_source_that_has_it():
    merged = merge_metadata(
        [
            ("provider", ModelMetadata(context_length=8192)),
            ("models.dev", ModelMetadata(display_name="Llama", context_length=131072, capabilities=["tool_calling"])),
            ("extra", ModelMetadata(display_name="ignored", max_output_tokens=4096)),
        ],
        refreshed_at=NOW,
    )

    assert merged == ModelMetadata(
        display_name="Llama",
        context_length=8192,
        capabilities=[ModelCapability.TOOL_CALLING],
        max_output_tokens=4096,
        sources=["provider", "models.dev", "extra"],
        refreshed_at=NOW,
    )


def test_merge_lists_only_sources_that_contributed():
    merged = merge_metadata(
        [("provider", ModelMetadata(context_length=8192)), ("models.dev", ModelMetadata(context_length=1))],
        refreshed_at=NOW,
    )

    assert merged is not None
    assert merged.sources == ["provider"]


def test_merge_returns_none_without_results():
    assert merge_metadata([("provider", ModelMetadata())], refreshed_at=NOW) is None


def test_upstream_model_id_uses_provider_served_model_name():
    assert upstream_model_id(_model(), _provider()) == "meta/llama-3.1-8b-instruct"


def test_upstream_model_id_falls_back_to_api_endpoint_then_name():
    with_endpoint = _model().model_copy(
        update={"api_endpoint": APIEndpointData(url="http://x/v1", model_id="upstream-id")}
    )

    assert upstream_model_id(with_endpoint, None) == "upstream-id"
    assert upstream_model_id(_model(), None) == "llama"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (_model(), True),
        (_model(providers=[]), False),
        (_model(ModelMetadata(display_name="hand written")), False),
        (_model(ModelMetadata(sources=["models.dev"], refreshed_at=NOW - timedelta(hours=1))), False),
        (_model(ModelMetadata(sources=["models.dev"], refreshed_at=NOW - timedelta(days=2))), True),
        (_model(ModelMetadata(sources=["models.dev"])), True),
    ],
    ids=["no-metadata", "no-provider", "manual", "fresh", "stale", "never-refreshed"],
)
def test_needs_refresh(model: ModelEntity, expected: bool):
    controller, _ = _controller([])

    assert controller.needs_refresh(model, NOW) is expected


async def test_reconcile_model_writes_merged_metadata():
    provider_source = FakeSource("provider", ModelMetadata(context_length=8192))
    catalog_source = FakeSource("models.dev", ModelMetadata(display_name="Llama 3.1 8B"))
    controller, models_client = _controller([provider_source, catalog_source])

    await controller.reconcile_model(_model(), _provider())

    assert provider_source.targets == [
        MetadataTarget(upstream_model_id="meta/llama-3.1-8b-instruct", provider=_provider())
    ]
    models_client.update_model.assert_awaited_once()
    kwargs = models_client.update_model.await_args.kwargs
    assert kwargs["workspace"] == "default"
    assert kwargs["name"] == "llama"
    metadata = kwargs["body"].metadata
    assert metadata.display_name == "Llama 3.1 8B"
    assert metadata.context_length == 8192
    assert metadata.sources == ["provider", "models.dev"]
    assert metadata.refreshed_at is not None


async def test_reconcile_model_continues_past_failing_source():
    controller, models_client = _controller([FailingSource(), FakeSource("models.dev", ModelMetadata(family="llama"))])

    await controller.reconcile_model(_model(), _provider())

    assert models_client.update_model.await_args.kwargs["body"].metadata.sources == ["models.dev"]


async def test_reconcile_model_skips_retry_after_a_miss():
    source = FakeSource("models.dev", None)
    controller, models_client = _controller([source])

    await controller.reconcile_model(_model(), _provider())
    await controller.reconcile_model(_model(), _provider())

    assert len(source.targets) == 1
    models_client.update_model.assert_not_awaited()


async def test_reconcile_model_tolerates_conflict():
    controller, models_client = _controller([FakeSource("models.dev", ModelMetadata(family="llama"))])
    models_client.update_model.side_effect = ConflictError(httpx.Response(409, json={"detail": "conflict"}))

    await controller.reconcile_model(_model(), _provider())

    models_client.update_model.assert_awaited_once()


async def test_http_client_does_not_request_zstd():
    async with build_http_client(ModelMetadataConfig()) as http:
        assert "zstd" not in http.headers["Accept-Encoding"]
