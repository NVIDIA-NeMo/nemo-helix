# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator configuration for the builder: the ``builder:`` section, or ``NEMO_BUILDER_*``.

How a build runs is set here, never in the request.
"""

from __future__ import annotations

from typing import Any, ClassVar, Self

from nemo_builder_plugin.identity import ImageIdentityError, validate_registry_host, validate_repository
from nemo_helix_plugin.config import NemoConfig
from pydantic import Field, ModelWrapValidatorHandler, PrivateAttr, field_validator, model_validator


class BuilderConfig(NemoConfig):
    """Configuration for in-cluster container image builds."""

    plugin_name: ClassVar[str] = "builder"
    plugin_description: ClassVar[str] = "In-cluster container image builds for NeMo Helix."

    namespace: str = Field(default="nhx-builds", description="Namespace the build pods run in.")
    work_pvc: str = Field(default="nhx-build-work", description="Claim of the work volume the steps share.")
    sandbox_image: str | None = Field(
        default=None,
        description=(
            "The kaniko image the sandbox runs. It must have the layout of kaniko's `debug` image: "
            "`/kaniko/executor`, and a shell at `/busybox/sh`. Unset refuses every submit."
        ),
    )
    node_selector: dict[str, str] = Field(
        default_factory=lambda: {"nhx.nvidia.com/build-node": "true"},
        description="Pins build pods to one node, as a ReadWriteOnce work volume requires.",
    )
    sandbox_cpu: str = Field(default="2", description="CPU request for the sandbox.")
    sandbox_memory: str = Field(default="8Gi", description="Memory request for the sandbox.")
    sandbox_dns_nameservers: list[str] = Field(
        default_factory=lambda: ["8.8.8.8", "1.1.1.1"],
        description=(
            "Resolvers the sandbox uses instead of cluster DNS, so a network policy can block every cluster address."
        ),
    )

    fetch_profile: str = Field(default="build-fetch")
    control_profile: str = Field(default="build-control")
    push_profile: str = Field(default="build-push")

    registry: str | None = Field(
        default=None,
        description=(
            "Host of the registry every image is published to, with an optional port; prefix it with `http://` "
            "for plain HTTP. Unset refuses every submit."
        ),
    )
    repository_prefix: str = Field(
        default="",
        description=(
            "Path every image is published under, before `<workspace>/`. For GAR, `<project>/<repository>`; "
            "for Artifactory, the repository key."
        ),
    )
    _registry_plain_http: bool = PrivateAttr(default=False)

    @model_validator(mode="wrap")
    @classmethod
    def _registry_scheme(cls, data: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        plain_http = False
        if isinstance(data, dict) and isinstance(data.get("registry"), str):
            scheme, separator, host = data["registry"].partition("://")
            if separator:
                if scheme not in ("http", "https"):
                    raise ValueError(
                        f"registry must be a host, optionally prefixed with http:// or https://, got {scheme}://"
                    )
                plain_http = scheme == "http"
                data = {**data, "registry": host}
        config = handler(data)
        config._registry_plain_http = plain_http
        return config

    @property
    def registry_plain_http(self) -> bool:
        return self._registry_plain_http

    @field_validator("registry")
    @classmethod
    def _registry_is_a_host(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if "/" in value:
            raise ValueError(
                f"registry must be a host without a path, got {value!r}. Put the path in `repository_prefix`."
            )
        try:
            return validate_registry_host(value)
        except ImageIdentityError as exc:
            raise ValueError(f"registry must be a host, optionally with a port, got {value!r}") from exc

    @field_validator("repository_prefix")
    @classmethod
    def _prefix_is_a_repository_path(cls, value: str) -> str:
        value = value.strip("/")
        return validate_repository(value) if value else value

    # The push step's credentials: platform secrets in the submitting workspace, which Jobs reads as the
    # submitter and gives the push step as environment variables. A workspace without them can't build.
    registry_username_secret: str = Field(
        default="builder-registry-username",
        min_length=1,
        description="The platform secret holding the username the push step logs in to `registry` with.",
    )
    registry_password_secret: str = Field(
        default="builder-registry-password",
        min_length=1,
        description="The platform secret holding that username's password or token.",
    )
    signing_key_secret: str = Field(
        default="builder-signing-key",
        min_length=1,
        description=(
            "The platform secret holding the PEM private key, EC or RSA and unencrypted, the push step signs "
            "every image with. Its public half is what the workspace's images verify against."
        ),
    )
