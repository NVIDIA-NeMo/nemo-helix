# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import binascii

from nhx.common.config import create_service_config_class, get_service_config, internal_field
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

REFRESH_MODEL_CACHE_INTERVAL_SEC = 3
SECRETS_TTL_SEC = 0  # 0 = refresh secrets on every provider cache refresh


class ServedModel(BaseModel):
    served_model_name: str
    model_entity_id: str


class DebugModelProvider(BaseModel):
    workspace: str
    name: str
    host_url: str
    served_models: list[ServedModel] = Field(default_factory=list)


class InferenceGatewayConfig(create_service_config_class("inference_gateway")):  # type: ignore
    """
    Configuration for the Inference Gateway Service.
    This service acts as a reverse proxy for inference requests.

    """

    model_config = ConfigDict(hide_input_in_errors=True)

    debug_model_providers: list[DebugModelProvider] = internal_field(
        default_factory=list,
        description="When this field has elements, we'll use them instead of filling the cache with response data from Models. This is useful for debugging purposes and running IGW in isolation.",
    )
    refresh_model_cache_interval_sec: int = Field(
        default=REFRESH_MODEL_CACHE_INTERVAL_SEC,
        description="How frequently (in seconds) to refresh the internal model cache from the Models service. If set to 0, disable automatic refreshing.",
    )
    secrets_ttl_sec: int = Field(
        default=SECRETS_TTL_SEC,
        description="Time-to-live (in seconds) for cached secrets before they expire and need refreshing. If set to 0, we will refresh this each time the model providers are refreshed.",
    )
    caller_context_key: SecretStr | None = Field(
        default=None,
        description="Base64-encoded 32-byte key the gateway seals a caller's credential with when it calls a model "
        "provider that calls models back as its caller. Every gateway process must share it. When unset, each "
        "process uses a key of its own, which works only while a router's calls back reach the process that called "
        "it.",
    )
    mock_provider_prefix: str | None = internal_field(
        default=None,
        description="When set, providers whose name starts with this prefix return mock "
        "responses instead of proxying to real backends. Use X-Mock-Response header to "
        "specify the JSON response body. Set to 'igw-mock-' for standard testing. "
        "Internal platform dev use only.",
    )

    @field_validator("caller_context_key")
    @classmethod
    def _caller_context_key_is_32_bytes(cls, key: SecretStr | None) -> SecretStr | None:
        if key is None:
            return key
        try:
            decoded = base64.b64decode(key.get_secret_value(), validate=True)
        except binascii.Error as exc:
            raise ValueError("caller_context_key must be base64-encoded") from exc
        if len(decoded) != 32:
            raise ValueError(f"caller_context_key must decode to 32 bytes, not {len(decoded)}")
        return key


# Actual instance of the config
config = get_service_config(InferenceGatewayConfig)
