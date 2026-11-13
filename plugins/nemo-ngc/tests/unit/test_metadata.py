# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from nemo_helix_plugin.models.metadata import MetadataSourceContext, MetadataTarget, SourceUnavailableError
from nemo_helix_plugin.models.types import ModelCapability, ModelMetadata, ModelModality, ModelProvider
from nemo_ngc_plugin.config import NgcConfig
from nemo_ngc_plugin.metadata import (
    NgcMetadataSource,
    endpoint_name_candidates,
    parse_ngc_endpoint,
    parse_parameter_count,
    spec_model_id,
)

BASE_URL = "https://api.ngc.nvidia.com"
ORG = "qc69jvmznzxy"


def _openapi(model_id: str) -> str:
    return json.dumps(
        {
            "components": {
                "schemas": {
                    "ChatCompletionResponse": {"properties": {"model": {"type": "string"}}},
                    "NIMLLMChatCompletionRequest": {"properties": {"model": {"type": "string", "default": model_id}}},
                }
            }
        }
    )


def _spec(model_id: str | None = "moonshotai/kimi-k3") -> dict[str, Any]:
    attributes = {
        "specifications": {
            "contextLength": 1048576,
            "parameters": "2.8T",
            "inputModalities": ["Text", "Image"],
            "outputModalities": ["Text"],
        },
        "modelCapability": {"functionCalling": True, "structuredOutput": True, "reasoning": False},
    }
    return {
        "artifactName": "kimi-k3",
        "attributes": json.dumps(attributes),
        "openAPISpec": _openapi(model_id) if model_id is not None else "{}",
    }


ARTIFACT = {"artifact": {"name": "kimi-k3", "shortDescription": "Multimodal MoE for long-horizon coding"}}


def _provider(host_url: str = "https://integrate.api.nvidia.com/v1") -> ModelProvider:
    now = datetime(2026, 11, 13, tzinfo=timezone.utc)
    return ModelProvider(name="build", workspace="default", host_url=host_url, created_at=now, updated_at=now)


def _ctx(http: httpx.AsyncClient) -> MetadataSourceContext:
    return MetadataSourceContext(http=http, client=MagicMock())


def _routes(routes: dict[str, dict[str, Any]], seen: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        body = routes.get(request.url.path)
        if body is None:
            return httpx.Response(404, json={"requestStatus": {"statusCode": "NOT_FOUND"}})
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("moonshotai/kimi-k3", ["kimi-k3"]),
        ("meta/llama-3.1-8b-instruct", ["llama-3.1-8b-instruct", "llama-3_1-8b-instruct", "llama-3-1-8b-instruct"]),
        ("z-ai/glm-5.3", ["glm-5.3", "glm-5_3", "glm-5-3"]),
        ("Qwen/Qwen3-32B", ["qwen3-32b"]),
    ],
)
def test_endpoint_name_candidates(model_id: str, expected: list[str]):
    assert endpoint_name_candidates(model_id) == expected


def test_spec_model_id_reads_request_schema_default():
    assert spec_model_id(_openapi("moonshotai/kimi-k3")) == "moonshotai/kimi-k3"


@pytest.mark.parametrize(
    "openapi_spec",
    [
        None,
        "",
        "not json",
        "{}",
        json.dumps({"components": None}),
        json.dumps({"components": {"schemas": None}}),
        json.dumps({"components": {"schemas": []}}),
    ],
)
def test_spec_model_id_returns_none_without_model_default(openapi_spec: object):
    assert spec_model_id(openapi_spec) is None


def test_parse_ngc_endpoint():
    metadata = parse_ngc_endpoint(_spec(), ARTIFACT["artifact"])

    assert metadata == ModelMetadata(
        description="Multimodal MoE for long-horizon coding",
        capabilities=[ModelCapability.TOOL_CALLING, ModelCapability.STRUCTURED_OUTPUT],
        input_modalities=[ModelModality.TEXT, ModelModality.IMAGE],
        output_modalities=[ModelModality.TEXT],
        parameter_count=2_800_000_000_000,
        context_length=1048576,
    )


def test_parse_ngc_endpoint_returns_none_without_metadata():
    assert parse_ngc_endpoint({"attributes": "not json"}, None) is None


async def test_ngc_source_fetches_spec_and_description():
    seen: list[str] = []
    routes = {f"/v2/endpoints/{ORG}/kimi-k3/spec": _spec(), f"/v2/endpoints/{ORG}/kimi-k3": ARTIFACT}

    async with httpx.AsyncClient(transport=_routes(routes, seen)) as http:
        metadata = await NgcMetadataSource(NgcConfig(url=BASE_URL, org=ORG)).fetch(
            MetadataTarget(upstream_model_id="moonshotai/kimi-k3", provider=_provider()), _ctx(http)
        )

    assert seen == [f"/v2/endpoints/{ORG}/kimi-k3/spec", f"/v2/endpoints/{ORG}/kimi-k3"]
    assert metadata is not None
    assert metadata.description == "Multimodal MoE for long-horizon coding"
    assert metadata.context_length == 1048576


async def test_ngc_source_tries_underscore_name_after_dotted_name_misses():
    seen: list[str] = []
    routes = {f"/v2/endpoints/{ORG}/llama-3_1-8b-instruct/spec": _spec("meta/llama-3.1-8b-instruct")}

    async with httpx.AsyncClient(transport=_routes(routes, seen)) as http:
        metadata = await NgcMetadataSource(NgcConfig(url=BASE_URL, org=ORG)).fetch(
            MetadataTarget(upstream_model_id="meta/llama-3.1-8b-instruct", provider=_provider()), _ctx(http)
        )

    assert seen[:2] == [
        f"/v2/endpoints/{ORG}/llama-3.1-8b-instruct/spec",
        f"/v2/endpoints/{ORG}/llama-3_1-8b-instruct/spec",
    ]
    assert metadata is not None
    assert metadata.description is None


async def test_ngc_source_rejects_endpoint_serving_a_different_model():
    seen: list[str] = []
    routes = {f"/v2/endpoints/{ORG}/kimi-k3/spec": _spec("someone-else/kimi-k3")}

    async with httpx.AsyncClient(transport=_routes(routes, seen)) as http:
        metadata = await NgcMetadataSource(NgcConfig(url=BASE_URL, org=ORG)).fetch(
            MetadataTarget(upstream_model_id="moonshotai/kimi-k3", provider=_provider()), _ctx(http)
        )

    assert metadata is None


async def test_ngc_source_accepts_endpoint_without_model_default():
    seen: list[str] = []
    routes = {f"/v2/endpoints/{ORG}/kimi-k3/spec": _spec(model_id=None)}

    async with httpx.AsyncClient(transport=_routes(routes, seen)) as http:
        metadata = await NgcMetadataSource(NgcConfig(url=BASE_URL, org=ORG)).fetch(
            MetadataTarget(upstream_model_id="moonshotai/kimi-k3", provider=_provider()), _ctx(http)
        )

    assert metadata is not None
    assert metadata.context_length == 1048576


@pytest.mark.parametrize(
    ("host_url", "expected"),
    [("https://integrate.api.nvidia.com/v1", True), ("https://api.openai.com/v1", False)],
)
def test_ngc_source_handles_only_nvidia_build(host_url: str, expected: bool):
    assert NgcMetadataSource(NgcConfig(url=BASE_URL, org=ORG)).handles(_provider(host_url)) is expected


@pytest.mark.parametrize(
    "response",
    [
        httpx.ConnectError("unreachable"),
        httpx.Response(503),
        httpx.Response(429),
        httpx.Response(200, text="<html>"),
        httpx.Response(200, json=[]),
        httpx.Response(200, json=None),
    ],
    ids=["unreachable", "server-error", "rate-limited", "not-json", "json-array", "json-null"],
)
async def test_ngc_source_raises_when_ngc_cannot_answer(response: httpx.Response | httpx.ConnectError):
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(response, Exception):
            raise response
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(SourceUnavailableError):
            await NgcMetadataSource(NgcConfig(url=BASE_URL, org=ORG)).fetch(
                MetadataTarget(upstream_model_id="moonshotai/kimi-k3", provider=_provider()), _ctx(http)
            )


async def test_ngc_source_treats_private_endpoint_as_not_found():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(403)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        metadata = await NgcMetadataSource(NgcConfig(url=BASE_URL, org=ORG)).fetch(
            MetadataTarget(upstream_model_id="mistralai/mistral-large", provider=_provider()), _ctx(http)
        )

    assert metadata is None
    assert seen == [f"/v2/endpoints/{ORG}/mistral-large/spec"]


async def test_ngc_source_returns_none_when_no_endpoint_matches():
    seen: list[str] = []

    async with httpx.AsyncClient(transport=_routes({}, seen)) as http:
        metadata = await NgcMetadataSource(NgcConfig(url=BASE_URL, org=ORG)).fetch(
            MetadataTarget(upstream_model_id="z-ai/glm-5.3", provider=_provider()), _ctx(http)
        )

    assert metadata is None
    assert len(seen) == 3


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2.8T", 2_800_000_000_000),
        ("552B", 552_000_000_000),
        ("8b", 8_000_000_000),
        ("560M", 560_000_000),
        (8030261248, 8030261248),
        ("8030261248", 8030261248),
        ("8x7B", None),
        ("", None),
        (None, None),
        (0, None),
        (True, None),
    ],
)
def test_parse_parameter_count(value: object, expected: int | None):
    assert parse_parameter_count(value) == expected


async def test_ngc_source_stops_calling_ngc_for_the_rest_of_the_cycle_after_a_failure():
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        raise httpx.ConnectError("unreachable")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = NgcMetadataSource(NgcConfig(url=BASE_URL, org=ORG))
        for model_id in ("moonshotai/kimi-k3", "z-ai/glm-5.3"):
            with pytest.raises(SourceUnavailableError):
                await source.fetch(MetadataTarget(upstream_model_id=model_id, provider=_provider()), _ctx(http))
        source.begin_cycle()
        with pytest.raises(SourceUnavailableError):
            await source.fetch(MetadataTarget(upstream_model_id="moonshotai/kimi-k3", provider=_provider()), _ctx(http))

    assert requests == [f"/v2/endpoints/{ORG}/kimi-k3/spec", f"/v2/endpoints/{ORG}/kimi-k3/spec"]
