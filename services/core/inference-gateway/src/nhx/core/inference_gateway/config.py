# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx.common.config import create_service_config_class, get_service_config, internal_field
from pydantic import BaseModel, Field

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
    mock_provider_prefix: str | None = internal_field(
        default=None,
        description="When set, providers whose name starts with this prefix return mock "
        "responses instead of proxying to real backends. Use X-Mock-Response header to "
        "specify the JSON response body. Set to 'igw-mock-' for standard testing. "
        "Internal platform dev use only.",
    )


# Actual instance of the config
config = get_service_config(InferenceGatewayConfig)
