# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Files service."""

from functools import cache
from urllib.parse import urlparse

from nmp.common.config import (
    create_service_config_class,
    get_service_config,
)
from nmp.core.files.app.backends.factory import StorageConfigField
from nmp.core.files.app.backends.local import LocalStorageConfig
from pydantic import Field, model_validator


# TODO(v2): CONFIG
class FilesConfig(create_service_config_class("files")):  # type: ignore
    """
    Configuration for the Files Service.

    Environment variables use the NMP_FILES_ prefix.
    """

    default_storage_config: StorageConfigField = Field(
        default_factory=lambda: LocalStorageConfig(path="/data/files_storage")
    )

    allowed_external_hosts: str = Field(
        default="https://api.ngc.nvidia.com,https://huggingface.co,https://api.github.com",
        description="Comma-separated list of external hosts the Files service is allowed to access.",
    )

    def get_allowed_external_hosts(self) -> list[str]:
        """Return the list of allowed external hosts (trimmed, non-empty)."""
        return [host.strip() for host in self.allowed_external_hosts.split(",") if host.strip()]

    @model_validator(mode="after")
    def validate_allowed_external_hosts(self) -> "FilesConfig":
        """Ensure each allowed external host is a valid URL."""
        for host in self.get_allowed_external_hosts():
            parsed = urlparse(host)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Allowed external host {host!r} must be a valid URL (scheme and netloc required)")
        return self

    allow_user_local_storage: bool = Field(
        default=False,
        description="Allow users to explicitly create filesets with local storage config. "
        "Security-sensitive: enable only in trusted deployments.",
    )

    file_lock_ttl_seconds: int = Field(
        default=300,
        description="TTL for file locks in seconds (default 5 minutes)",
    )

    cache_warming_max_concurrent: int = Field(
        default=3,
        description="Maximum concurrent downloads during cache warming",
    )

    hf_retry_attempts: int = Field(
        default=4,
        ge=1,
        description="Maximum Hugging Face request attempts for transient failures.",
    )

    hf_retry_initial_delay_seconds: float = Field(
        default=0.5,
        ge=0.0,
        description="Initial Hugging Face retry delay in seconds before exponential backoff.",
    )

    hf_retry_max_delay_seconds: float = Field(
        default=5.0,
        ge=0.0,
        description="Maximum Hugging Face retry delay in seconds.",
    )


# TODO(v2): CONFIG
@cache
def files_config() -> FilesConfig:
    return get_service_config(FilesConfig)
