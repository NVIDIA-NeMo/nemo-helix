# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta, timezone
from importlib.metadata import EntryPoint
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_helix_plugin.client.errors import ConflictError, InternalServerError
from nemo_helix_plugin.models.metadata import (
    MetadataSourceContext,
    MetadataTarget,
    NemoModelMetadataSource,
    SourceUnavailableError,
)
from nemo_helix_plugin.models.types import (
    APIEndpointData,
    ModelEntity,
    ModelMetadata,
    ModelProvider,
    ServedModelMapping,
)
from nhx.core.models.config import MetadataControllerConfig
from nhx.core.models.metadata import controller as controller_module
from nhx.core.models.metadata.controller import ModelMetadataController, build_http_client, upstream_model_id
from pydantic import ValidationError

NOW = datetime(2026, 11, 10, tzinfo=timezone.utc)
BUILD_HOST = "https://integrate.api.nvidia.com/v1"


class FakeSource(NemoModelMetadataSource):
    name = "fake"

    def __init__(self, result: ModelMetadata | None, host: str = BUILD_HOST) -> None:
        self.result = result
        self.host = host
        self.targets: list[MetadataTarget] = []

    def handles(self, provider: ModelProvider) -> bool:
        return provider.host_url == self.host

    async def fetch(self, target: MetadataTarget, ctx: MetadataSourceContext) -> ModelMetadata | None:
        self.targets.append(target)
        return self.result


class FailingSource(FakeSource):
    async def fetch(self, target: MetadataTarget, ctx: MetadataSourceContext) -> ModelMetadata | None:
        raise SourceUnavailableError("NGC returned HTTP 503")


class BrokenHandlesSource(FakeSource):
    def handles(self, provider: ModelProvider) -> bool:
        raise RuntimeError("bug")


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


def _provider(name: str = "nim", host_url: str = BUILD_HOST) -> ModelProvider:
    return ModelProvider(
        created_at=NOW,
        updated_at=NOW,
        name=name,
        workspace="default",
        host_url=host_url,
        served_models=[
            ServedModelMapping(model_entity_id="default/llama", served_model_name="meta/llama-3.1-8b-instruct")
        ],
    )


def _controller(
    plugin_sources: list[NemoModelMetadataSource], current: ModelEntity | None = None
) -> tuple[ModelMetadataController, MagicMock]:
    controller = ModelMetadataController(
        config=MetadataControllerConfig(), client=MagicMock(), http=MagicMock(), plugin_sources=plugin_sources
    )
    models_client = MagicMock()
    models_client.update_model = AsyncMock()
    models_client.get_model = AsyncMock(return_value=MagicMock(data=MagicMock(return_value=current or _model())))
    controller._models_client = models_client
    controller._providers = {
        "default/nim": _provider(),
        "default/build": _provider("build"),
        "default/vllm": _provider("vllm", "http://vllm:8000/v1"),
    }
    return controller, models_client


def _written_metadata(models_client: MagicMock) -> ModelMetadata:
    return models_client.update_model.await_args.kwargs["body"].metadata


def test_upstream_model_id_uses_provider_served_model_name():
    assert upstream_model_id(_model(), _provider()) == "meta/llama-3.1-8b-instruct"


def test_upstream_model_id_falls_back_to_api_endpoint_then_name():
    unmapped = _provider().model_copy(update={"served_models": []})
    with_endpoint = _model().model_copy(
        update={"api_endpoint": APIEndpointData(url="http://x/v1", model_id="upstream-id")}
    )

    assert upstream_model_id(with_endpoint, unmapped) == "upstream-id"
    assert upstream_model_id(_model(), unmapped) == "llama"


def test_metadata_rejects_naive_refreshed_at():
    with pytest.raises(ValidationError):
        ModelMetadata(sources=["ngc"], refreshed_at=datetime(2026, 11, 10))


def test_source_for_picks_plugin_that_handles_provider():
    plugin = FakeSource(None)
    controller, _ = _controller([plugin])

    assert controller.source_for(_provider()) is plugin


@pytest.mark.parametrize("plugins", [[], [FakeSource(None, host="https://other")], [BrokenHandlesSource(None)]])
def test_source_for_falls_back_to_provider_models(plugins: list[NemoModelMetadataSource]):
    controller, _ = _controller(plugins)

    assert controller.source_for(_provider()).name == "provider"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (_model(), True),
        (_model(providers=[]), False),
        (_model(ModelMetadata(display_name="hand written")), False),
        (_model(ModelMetadata(sources=["ngc"], refreshed_at=NOW - timedelta(hours=1))), False),
        (_model(ModelMetadata(sources=["ngc"], refreshed_at=NOW - timedelta(days=2))), True),
        (_model(ModelMetadata(sources=["ngc"])), True),
    ],
    ids=["no-metadata", "no-provider", "manual", "fresh", "stale", "never-refreshed"],
)
def test_needs_refresh(model: ModelEntity, expected: bool):
    controller, _ = _controller([])

    assert controller.needs_refresh(model, NOW) is expected


async def test_reconcile_model_writes_metadata_from_the_provider_source():
    source = FakeSource(ModelMetadata(display_name="Llama 3.1 8B", context_length=131072))
    controller, models_client = _controller([source])

    await controller.reconcile_model(_model())

    assert source.targets == [MetadataTarget(upstream_model_id="meta/llama-3.1-8b-instruct", provider=_provider())]
    kwargs = models_client.update_model.await_args.kwargs
    assert (kwargs["workspace"], kwargs["name"]) == ("default", "llama")
    metadata = _written_metadata(models_client)
    assert metadata.display_name == "Llama 3.1 8B"
    assert metadata.sources == ["fake"]
    assert metadata.refreshed_at is not None


async def test_reconcile_model_skips_write_when_source_cannot_answer():
    stale = ModelMetadata(description="keep me", sources=["fake"], refreshed_at=NOW - timedelta(days=2))
    controller, models_client = _controller([FailingSource(None)])

    await controller.reconcile_model(_model(stale))

    models_client.update_model.assert_not_awaited()
    assert controller.needs_refresh(_model(), NOW) is True


async def test_reconcile_model_skips_retry_after_a_miss():
    controller, models_client = _controller([FakeSource(None)])

    await controller.reconcile_model(_model())

    assert controller.needs_refresh(_model(), NOW) is False
    models_client.update_model.assert_not_awaited()


async def test_reconcile_model_does_not_overwrite_metadata_set_by_hand_during_the_cycle():
    controller, models_client = _controller(
        [FakeSource(ModelMetadata(context_length=8192))], current=_model(ModelMetadata(display_name="hand written"))
    )

    await controller.reconcile_model(_model())

    models_client.update_model.assert_not_awaited()


async def test_reconcile_model_falls_through_to_next_provider():
    source = FakeSource(None)
    controller, models_client = _controller([source])
    controller._provider_source.fetch = AsyncMock(return_value=ModelMetadata(context_length=32768))

    await controller.reconcile_model(_model(providers=["default/nim", "default/vllm"]))

    metadata = _written_metadata(models_client)
    assert metadata.context_length == 32768
    assert metadata.sources == ["provider"]


async def test_reconcile_model_tolerates_conflict():
    controller, models_client = _controller([FakeSource(ModelMetadata(context_length=8192))])
    models_client.update_model.side_effect = ConflictError(httpx.Response(409, json={"detail": "conflict"}))

    await controller.reconcile_model(_model())

    models_client.update_model.assert_awaited_once()


async def test_reconcile_keeps_going_after_one_model_fails():
    controller, models_client = _controller([FakeSource(ModelMetadata(context_length=8192))])
    second = _model().model_copy(update={"name": "other"})
    models_client.update_model.side_effect = [InternalServerError(httpx.Response(500, json={})), None]
    controller.list_models_to_refresh = AsyncMock(return_value=[_model(), second])

    await controller.reconcile()

    assert models_client.update_model.await_count == 2


def test_load_plugin_sources_skips_sources_that_fail_to_load(monkeypatch: pytest.MonkeyPatch):
    good = MagicMock(spec=EntryPoint)
    good.load.return_value = lambda: FakeSource(None)
    broken = MagicMock(spec=EntryPoint)
    broken.load.side_effect = ImportError("missing dependency")
    monkeypatch.setattr(controller_module, "discover_entry_points", lambda group: {"ngc": good, "broken": broken})

    sources = controller_module.load_plugin_sources()

    assert [source.name for source in sources] == ["fake"]


async def test_http_client_does_not_request_zstd():
    async with build_http_client(MetadataControllerConfig()) as http:
        assert "zstd" not in http.headers["Accept-Encoding"]
