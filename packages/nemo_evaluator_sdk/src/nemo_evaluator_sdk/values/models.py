# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-related value types."""

from __future__ import annotations

import logging
import os
from functools import cached_property
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator
from pydantic.config import JsonDict

from nemo_evaluator_sdk.enums import ModelFormat
from nemo_evaluator_sdk.values.common import SecretRef

logger = logging.getLogger(__name__)

_AUTH_HEADER_PATTERNS: tuple[str, ...] = (
    "auth",
    "token",
    "key",
    "secret",
    "credential",
    "bearer",
    "cookie",
    "set-cookie",
)
# Keep this aligned with nmp.common.entities.constants.NAME_PATTERN without adding an SDK dependency on nmp_common.
# This pydantic-core-compatible form avoids lookarounds while preserving the naming syntax:
# start with lowercase alpha, require at least one more valid character, allow single hyphens between non-hyphen chars,
# and do not end in hyphen.
_ENTITY_NAME_SEGMENT = r"[a-z](?:[a-z0-9@.+_]|-[a-z0-9@.+_]){1,62}"
_QUALIFIED_MODEL_REF_PATTERN = rf"^{_ENTITY_NAME_SEGMENT}/{_ENTITY_NAME_SEGMENT}$"

RankingContract: TypeAlias = Literal[
    "nim-ranking-v1",
    "hosted-rerank-v1",
    "hosted-ranking-v1",
    "hosted-retrieval-reranking-v1",
]


def _require_type_in_json_schema(schema: dict[str, Any]) -> None:
    """Require the discriminator in serialized inference payloads."""
    required = schema.setdefault("required", [])
    if "type" not in required:
        required.append("type")


class RankingInference(BaseModel):
    """Reranking invocation contract and route.

    Additional inference variants, such as embeddings, join InferenceConfig as new
    type members rather than adding more fields to Model.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra=_require_type_in_json_schema)

    type: Literal["ranking"] = "ranking"
    contract: RankingContract | None = Field(
        default=None,
        description="Explicit reranking request and response contract selected during preflight.",
        json_schema_extra={"nullable": True},
    )
    path: str | None = Field(
        default=None,
        description="Explicit reranking route relative to the model's resolved /v1 inference-gateway URL.",
        json_schema_extra={"nullable": True},
    )


InferenceConfig: TypeAlias = Annotated[
    RankingInference,
    Field(discriminator="type"),
]


_ModelRefRoot = Annotated[
    str,
    Field(
        pattern=_QUALIFIED_MODEL_REF_PATTERN,
        description="Reference to a model (format: workspace/name).",
        examples=[
            "workspace/model_name",
        ],
    ),
]


class ReasoningParams(BaseModel):
    """Custom settings that control the model's reasoning behavior."""

    end_token: str | None = Field(
        default=None,
        description="Configure the end token to trim reasoning context based on the model's reasoning API. Example for Nemotron models: '</think>'",
    )
    include_if_not_finished: bool | None = Field(
        default=None,
        description="Configure whether to include reasoning context if the model has not finished reasoning.",
    )
    effort: str | None = Field(
        default=None,
        description="Option for OpenAI models to specify low, medium, or high reasoning effort.",
    )


class ModelRef(RootModel[_ModelRefRoot]):
    """Reference to a model that can be resolved by an evaluator backend."""


def _strip_internal_fields(schema: JsonDict) -> None:
    """Remove internal-only fields from the JSON schema so they don't appear in the OpenAPI spec."""
    props = schema.get("properties")
    if isinstance(props, dict):
        props.pop("default_headers", None)
        props.pop("host_url", None)


def normalize_header_name(header_name: str) -> str:
    """Normalize a transport header name for policy checks."""
    return header_name.strip().lower().replace("_", "-")


def is_auth_header_name(header_name: str) -> bool:
    """Return whether a header name appears to carry authentication material."""
    normalized_header_name = normalize_header_name(header_name)
    return any(pattern in normalized_header_name for pattern in _AUTH_HEADER_PATTERNS)


def filter_auth_headers(headers: dict[str, str] | None) -> dict[str, str] | None:
    """Return only non-auth headers from a header mapping."""
    if headers is None:
        return None

    filtered_headers: dict[str, str] = {}
    for header_name, header_value in headers.items():
        if is_auth_header_name(header_name):
            logger.debug(
                f"Filtered header {header_name} because it was recognized as an auth header",
            )
            continue
        filtered_headers[header_name] = header_value

    return filtered_headers or None


class Model(BaseModel):
    """Model definition for use without persisting to the Models API."""

    model_config = ConfigDict(extra="forbid", json_schema_extra=_strip_internal_fields)

    url: str = Field(description="URL of the model.")
    name: str = Field(description="Name of the model.")
    default_headers: dict[str, str] | None = Field(
        default=None,
        exclude=True,
        description="Runtime-only non-auth headers automatically applied to requests made with this model. "
        "Authentication must be configured via model.api_key_secret; auth headers such as Authorization will be rejected.",
    )
    host_url: str | None = Field(
        default=None,
        description="Direct NIM endpoint URL (http://host:port). Populated when resolved from a ModelRef. "
        "Used by EvalFactory containers that reject path-based URLs (e.g., Haystack NvidiaDocumentEmbedder).",
    )
    served_model_name: str | None = Field(
        default=None,
        description="Provider model identifier preserved when resolving a ModelRef.",
        json_schema_extra={"nullable": True},
    )
    inference: InferenceConfig | None = Field(
        default=None,
        description="Capability-specific invocation metadata. Ranking is the only variant today.",
        json_schema_extra={"nullable": True},
    )
    api_key_secret: SecretRef | None = Field(
        default=None,
        description="API key secret reference for the model. Format: workspace/secret_name or secret_name within the job workspace.",
    )
    format: Literal[ModelFormat.NVIDIA_NIM, ModelFormat.OPEN_AI, ModelFormat.LLAMA_STACK] = Field(
        default=ModelFormat.OPEN_AI,
        deprecated=(
            "`format` is ignored and will be removed. Structured output support is detected from "
            "the endpoint during preflight rather than inferred from this label."
        ),
        description=(
            "Deprecated and ignored. Previously selected the structured output encoding; that is "
            "now detected per endpoint, because support is a property of the endpoint rather than "
            "of the label attached to the model."
        ),
    )

    @field_validator("default_headers")
    @classmethod
    def validate_default_headers(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        """Reject auth-style headers and direct users to api_key_secret for credentials."""
        if value is None:
            return None

        for header_name in value:
            if is_auth_header_name(header_name):
                raise ValueError(
                    f"Header {header_name} in model.default_headers is rejected because model.default_headers cannot include authentication headers (Authorization, X-API-Key, etc.). "
                    f"Header names are not allowed to contain the following substrings: {', '.join(_AUTH_HEADER_PATTERNS)}. "
                    f"Configure model auth via model.api_key_secret instead."
                )
        return value

    @cached_property
    def api_key_env(self) -> str | None:
        if self.api_key_secret:
            env_name = self.api_key_secret.root
            if env_name[0].isdigit():
                env_name = f"_{env_name}"  # prefix with valid character for environment variable
            return env_name.replace("-", "_").replace("/", "_")

    @cached_property
    def api_key(self) -> str | None:
        if self.api_key_secret:
            api_key_env = self.api_key_env
            assert api_key_env is not None
            return os.getenv(api_key_env) or os.getenv(api_key_env.upper())

    def with_default_headers(self, headers: dict[str, str] | None) -> "Model":
        """Return a copy of the model with merged runtime default headers."""
        if not headers:
            return self

        return self.model_copy(
            update={
                "default_headers": {
                    **(self.default_headers or {}),
                    **headers,
                }
            },
            # flat dict, no deep copy is needed
            deep=False,
        )
