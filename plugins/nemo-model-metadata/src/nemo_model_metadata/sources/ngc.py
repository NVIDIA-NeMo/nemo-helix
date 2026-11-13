# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
from typing import Any
from urllib.parse import quote, urlparse

import httpx
from nemo_helix_plugin.models.types import ModelCapability, ModelMetadata
from nemo_model_metadata.config import SourceName
from nemo_model_metadata.sources.base import MetadataSource, MetadataTarget
from nemo_model_metadata.sources.parsing import as_positive_int, modalities

logger = logging.getLogger(__name__)

NVIDIA_BUILD_HOST = "integrate.api.nvidia.com"

_CAPABILITY_FLAGS: dict[str, ModelCapability] = {
    "functionCalling": ModelCapability.TOOL_CALLING,
    "reasoning": ModelCapability.REASONING,
    "structuredOutput": ModelCapability.STRUCTURED_OUTPUT,
}

_MAX_DESCRIPTION_LENGTH = 4000

_PARAMETER_SUFFIXES = {"K": 10**3, "M": 10**6, "B": 10**9, "T": 10**12}
_PARAMETER_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*([KMBT])$", re.IGNORECASE)


def parse_parameter_count(value: object) -> int | None:
    """Read NGC's parameter count, which is either an integer or a string such as ``"8B"`` or ``"2.8T"``."""
    if not isinstance(value, str):
        return as_positive_int(value)
    match = _PARAMETER_PATTERN.match(value.strip())
    if match is None:
        return as_positive_int(int(value)) if value.strip().isdigit() else None
    return as_positive_int(round(float(match.group(1)) * _PARAMETER_SUFFIXES[match.group(2).upper()]))


def endpoint_name_candidates(model_id: str) -> list[str]:
    """NGC endpoint names are the model id's last segment, sometimes with dots turned into underscores."""
    name = model_id.rsplit("/", 1)[-1].lower()
    return list(dict.fromkeys([name, name.replace(".", "_")]))


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def spec_model_id(openapi_spec: object) -> str | None:
    """Return the default ``model`` of the request schema in the endpoint's OpenAPI document."""
    schemas = _json_object(openapi_spec).get("components", {}).get("schemas", {})
    if not isinstance(schemas, dict):
        return None
    for schema in schemas.values():
        properties = schema.get("properties") if isinstance(schema, dict) else None
        model = properties.get("model") if isinstance(properties, dict) else None
        default = model.get("default") if isinstance(model, dict) else None
        if isinstance(default, str) and default:
            return default
    return None


def parse_ngc_endpoint(spec: dict[str, Any], artifact: dict[str, Any] | None) -> ModelMetadata | None:
    attributes = _json_object(spec.get("attributes"))
    specifications = attributes.get("specifications")
    if not isinstance(specifications, dict):
        specifications = {}
    flags = attributes.get("modelCapability")
    if not isinstance(flags, dict):
        flags = {}
    capabilities = [capability for flag, capability in _CAPABILITY_FLAGS.items() if flags.get(flag) is True]
    description = (artifact or {}).get("shortDescription")

    metadata = ModelMetadata(
        description=description[:_MAX_DESCRIPTION_LENGTH] if isinstance(description, str) and description else None,
        capabilities=capabilities,
        input_modalities=modalities(specifications.get("inputModalities")),
        output_modalities=modalities(specifications.get("outputModalities")),
        parameter_count=parse_parameter_count(specifications.get("parameters")),
        context_length=as_positive_int(specifications.get("contextLength")),
    )
    if metadata == ModelMetadata():
        return None
    return metadata


class NgcSource(MetadataSource):
    """Reads the NGC catalog entry behind a build.nvidia.com model."""

    name = SourceName.NGC

    def __init__(self, http: httpx.AsyncClient, base_url: str, org: str) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._org = org

    async def fetch(self, target: MetadataTarget) -> ModelMetadata | None:
        if target.provider is None or urlparse(target.provider.host_url).hostname != NVIDIA_BUILD_HOST:
            return None
        for endpoint_name in endpoint_name_candidates(target.upstream_model_id):
            endpoint_url = f"{self._base_url}/v2/endpoints/{quote(self._org)}/{quote(endpoint_name)}"
            spec = await self._get_json(f"{endpoint_url}/spec")
            if spec is None:
                continue
            served_model_id = spec_model_id(spec.get("openAPISpec"))
            if served_model_id is not None and served_model_id != target.upstream_model_id:
                continue
            details = await self._get_json(endpoint_url)
            artifact = details.get("artifact") if details is not None else None
            return parse_ngc_endpoint(spec, artifact if isinstance(artifact, dict) else None)
        return None

    async def _get_json(self, url: str) -> dict[str, Any] | None:
        try:
            response = await self._http.get(url)
        except httpx.HTTPError as exc:
            logger.debug("NGC request to %s failed: %s", url, exc)
            return None
        if response.status_code != 200:
            return None
        try:
            body: object = response.json()
        except ValueError:
            return None
        return body if isinstance(body, dict) else None
