# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Guardrails service."""

import json
import logging
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Dict, List

from nhx.common.config.base import create_service_config_class, get_service_config
from nhx.guardrails.app.constants import FALLBACK_DEFAULT_ENDPOINT_URL
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class NIMEndpointSettings(BaseSettings):
    """Settings for the NIM endpoint timeouts."""

    model_config = SettingsConfigDict(extra="ignore")

    connect_timeout: float = Field(default=5.0, alias="NIM_ENDPOINT_CONNECT_TIMEOUT")
    read_timeout: float = Field(default=10.0, alias="NIM_ENDPOINT_READ_TIMEOUT")
    base_url: str = Field(default=FALLBACK_DEFAULT_ENDPOINT_URL, alias="NIM_ENDPOINT_URL")


class GuardrailsServiceConfig(create_service_config_class("guardrails")):
    """
    Configuration for the Guardrails service.

    Read from the global config file under the 'guardrails' key and from
    environment variables with the NHX_GUARDRAILS_ prefix.
    """

    # API settings
    api_request_headers: ContextVar = Field(default_factory=lambda: ContextVar("headers"))

    # Config store settings
    config_store_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv("CONFIG_STORE_PATH", str(Path(__file__).parent / "assets" / "config-store"))
        ),
        description="Path to the config store directory",
    )
    storage_options: Dict = Field(
        default_factory=lambda: json.loads(os.getenv("STORAGE_OPTIONS", "{}")),
        description="Storage options for fsspec",
    )

    # Config source settings
    config_sources: List[Dict] = Field(
        default_factory=lambda: [
            {
                "source_type": "file",
                "config_path": os.environ.get("CONFIG_STORE_PATH", "/config-store"),
            },
        ],
        description="List of config sources",
    )
    is_single_config_mode: bool = Field(
        default=False,
        description="Single config mode flag",
    )

    # UI/Demo settings
    disable_chat_ui: bool = Field(default=True)
    auto_reload: bool = Field(default=False)
    demo: bool = Field(
        default=False,
        description="Demo mode flag",
    )

    # External service settings
    fetch_nim_app_models: bool = Field(
        default=False,
        description="Fetch models from NIM app",
    )

    # NIM endpoint settings
    nim_endpoint_settings: NIMEndpointSettings = Field(default_factory=NIMEndpointSettings)

    # LLM provider settings
    default_llm_provider: str = Field(
        default="nim",
        description="Default LLM provider",
    )

    # Cache settings
    config_cache_ttl: int = Field(
        default=60,
        description="Config cache TTL in seconds",
    )
    config_cache_staleness_threshold: int = Field(
        default=45,
        description="Config cache staleness threshold in seconds",
    )

    def configure(self, config_path: str, source_type: str = "file"):
        """Reconfigure settings with new config path."""
        self.config_sources = [{"source_type": source_type, "config_path": config_path}]


# Singleton instances
settings = get_service_config(GuardrailsServiceConfig)
