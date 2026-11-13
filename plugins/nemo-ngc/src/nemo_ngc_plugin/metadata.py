# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from typing import Any
from urllib.parse import quote, urlparse

import httpx
from nemo_helix_plugin.models.metadata import (
    MAX_DESCRIPTION_LENGTH,
    MetadataSourceContext,
    MetadataTarget,
    NemoModelMetadataSource,
    SourceUnavailableError,
    as_positive_int,
    bounded_text,
    modalities,
)
from nemo_helix_plugin.models.types import ModelCapability, ModelMetadata, ModelProvider
from nemo_ngc_plugin.config import NgcConfig

NVIDIA_BUILD_HOST = "integrate.api.nvidia.com"

# NGC answers 403 rather than 404 for endpoints that exist but are retired or private.
_NOT_PUBLIC_STATUSES = frozenset({401, 403, 404})

_CAPABILITY_FLAGS: dict[str, ModelCapability] = {
    "functionCalling": ModelCapability.TOOL_CALLING,
    "reasoning": ModelCapability.REASONING,
    "structuredOutput": ModelCapability.STRUCTURED_OUTPUT,
}


_PARAMETER_SUFFIXES = {"K": 10**3, "M": 10**6, "B": 10**9, "T": 10**12}
_PARAMETER_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*([KMBT])$", re.IGNORECASE)


def parse_parameter_count(value: object) -> int | None:
    if not isinstance(value, str):
        return as_positive_int(value)
    match = _PARAMETER_PATTERN.match(value.strip())
    if match is None:
        return as_positive_int(int(value)) if value.strip().isdigit() else None
    return as_positive_int(round(float(match.group(1)) * _PARAMETER_SUFFIXES[match.group(2).upper()]))


# NGC names an endpoint after the model id's last segment, sometimes with dots turned into underscores or dashes.
def endpoint_name_candidates(model_id: str) -> list[str]:
    name = model_id.rsplit("/", 1)[-1].lower()
    return list(dict.fromkeys([name, name.replace(".", "_"), name.replace(".", "-")]))


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def spec_model_id(openapi_spec: object) -> str | None:
    components = _json_object(openapi_spec).get("components")
    schemas = components.get("schemas") if isinstance(components, dict) else None
    if not isinstance(schemas, dict):
        return None
    for schema in schemas.values():
        properties = schema.get("properties") if isinstance(schema, dict) else None
        model = properties.get("model") if isinstance(properties, dict) else None
        default = model.get("default") if isinstance(model, dict) else None
        if isinstance(default, str) and default:
            return default
    return None


def parse_ngc_endpoint(spec: dict[str, Any], artifact: dict[str, Any] | None) -> ModelMetadata:
    attributes = _json_object(spec.get("attributes"))
    specifications = attributes.get("specifications")
    if not isinstance(specifications, dict):
        specifications = {}
    flags = attributes.get("modelCapability")
    if not isinstance(flags, dict):
        flags = {}
    capabilities = [capability for flag, capability in _CAPABILITY_FLAGS.items() if flags.get(flag) is True]
    return ModelMetadata(
        description=bounded_text((artifact or {}).get("shortDescription"), MAX_DESCRIPTION_LENGTH),
        capabilities=capabilities,
        input_modalities=modalities(specifications.get("inputModalities")),
        output_modalities=modalities(specifications.get("outputModalities")),
        parameter_count=parse_parameter_count(specifications.get("parameters")),
        context_length=as_positive_int(specifications.get("contextLength")),
    )


class NgcMetadataSource(NemoModelMetadataSource):
    """Reads the NGC catalog entry behind a build.nvidia.com model."""

    name = "ngc"

    def __init__(self, config: NgcConfig | None = None) -> None:
        config = config or NgcConfig.get()
        self._base_url = config.url.rstrip("/")
        self._org = config.org
        self._unavailable: SourceUnavailableError | None = None

    def begin_cycle(self) -> None:
        super().begin_cycle()
        self._unavailable = None

    def handles(self, provider: ModelProvider) -> bool:
        return urlparse(provider.host_url).hostname == NVIDIA_BUILD_HOST

    async def fetch(self, target: MetadataTarget, ctx: MetadataSourceContext) -> ModelMetadata | None:
        if self._unavailable is not None:
            raise self._unavailable
        try:
            return await self._fetch(target, ctx)
        except SourceUnavailableError as exc:
            self._unavailable = exc
            raise

    async def _fetch(self, target: MetadataTarget, ctx: MetadataSourceContext) -> ModelMetadata | None:
        for endpoint_name in endpoint_name_candidates(target.upstream_model_id):
            endpoint_url = f"{self._base_url}/v2/endpoints/{quote(self._org)}/{quote(endpoint_name)}"
            spec = await _get_json(ctx.http, f"{endpoint_url}/spec")
            if spec is None:
                continue
            served_model_id = spec_model_id(spec.get("openAPISpec"))
            if served_model_id is not None and served_model_id != target.upstream_model_id:
                continue
            details = await _get_json(ctx.http, endpoint_url)
            artifact = details.get("artifact") if details is not None else None
            return parse_ngc_endpoint(spec, artifact if isinstance(artifact, dict) else None)
        return None


async def _get_json(http: httpx.AsyncClient, url: str) -> dict[str, Any] | None:
    try:
        response = await http.get(url)
    except httpx.HTTPError as exc:
        raise SourceUnavailableError(f"NGC request to {url} failed: {exc}") from exc
    if response.status_code in _NOT_PUBLIC_STATUSES:
        return None
    if response.status_code != 200:
        raise SourceUnavailableError(f"NGC request to {url} returned HTTP {response.status_code}")
    try:
        body: object = response.json()
    except ValueError as exc:
        raise SourceUnavailableError(f"NGC returned invalid JSON for {url}") from exc
    if not isinstance(body, dict):
        raise SourceUnavailableError(f"NGC returned a non-object response for {url}")
    return body
