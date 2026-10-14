# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator configuration for the builder: the ``builder:`` section, or ``NEMO_BUILDER_*``.

How a build runs is set here, never in the request.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, Self

from nemo_builder_plugin.identity import ImageIdentityError, validate_registry_host, validate_repository
from nemo_helix_plugin.config import NemoConfig
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    field_validator,
    model_validator,
)


class SandboxConfig(BaseModel):
    """The sandboxes a build's Dockerfiles run in.

    Where they run isn't set here: they run in the build step's namespace, on the work volume and nodes of
    the fetch step's Jobs execution profile. From the environment, the section is one JSON value,
    ``NEMO_BUILDER_SANDBOX``: only its one-word fields can be set on their own.
    """

    model_config = ConfigDict(extra="forbid")

    image: str | None = Field(
        default=None,
        description=(
            "The kaniko image the sandbox runs. Unset, the release's own `nhx-kaniko`, from `platform.image_registry` "
            "at `platform.image_tag`. Another needs `/kaniko/executor` and a shell at `/busybox/sh`, and must take "
            "the osscontainertools fork's `--credential-helpers` flag, which Chainguard's fork doesn't have."
        ),
    )
    provider: Literal["kubernetes_pod"] = Field(
        default="kubernetes_pod",
        description=(
            "What runs each sandbox. `kubernetes_pod`, the only one so far, is a plain pod the build step creates, "
            "hardened by the builder, with an unrestricted network: use it only with Dockerfiles you trust."
        ),
    )
    cpu: str = Field(default="2", description="CPU for each sandbox: its request and its limit.")
    memory: str = Field(default="8Gi", description="Memory for each sandbox: its request and its limit.")
    ephemeral_storage: str = Field(
        default="20Gi",
        description="Ephemeral storage for each sandbox, its request and its limit: kaniko unpacks each base image into it.",
    )
    dns_nameservers: list[str] = Field(
        default_factory=lambda: ["8.8.8.8", "1.1.1.1"],
        description=(
            "Resolvers the sandbox uses instead of cluster DNS, so a network policy can block every cluster address."
        ),
    )


class BuilderConfig(NemoConfig):
    """Configuration for in-cluster container image builds."""

    plugin_name: ClassVar[str] = "builder"
    plugin_description: ClassVar[str] = "In-cluster container image builds for NeMo Helix."

    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)

    fetch_profile: str = Field(
        default="build-fetch",
        description=(
            "The Jobs execution profile, provider `cpu` on `kubernetes_job`, the fetch step runs on. The sandboxes use "
            "its work volume (`storage.pvc_name`) and nodes. All three profiles must name one namespace, and the push "
            "step's the same work volume."
        ),
    )
    control_profile: str = Field(
        default="build-control",
        description="The one the build step runs on, which starts the sandboxes.",
    )
    push_profile: str = Field(
        default="build-push",
        description="The one the push step runs on, the only step that holds the registry credential and signing key.",
    )

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
    # Null turns one off for the whole deployment.
    registry_username_secret: str | None = Field(
        default="builder-registry-username",
        min_length=1,
        description=(
            "The platform secret holding the username the push step logs in to `registry` with. Null, with "
            "`registry_password_secret`, pushes without logging in, which only a registry open to anonymous "
            "pushes allows."
        ),
    )
    registry_password_secret: str | None = Field(
        default="builder-registry-password",
        min_length=1,
        description="The platform secret holding that username's password or token.",
    )
    signing_key_secret: str | None = Field(
        default="builder-signing-key",
        min_length=1,
        description=(
            "The platform secret holding the PEM private key, EC or RSA and unencrypted, the push step signs "
            "every image with. Its public half is what the workspace's images verify against. Null publishes "
            "images unsigned."
        ),
    )

    @model_validator(mode="after")
    def _credential_is_whole(self) -> Self:
        if (self.registry_username_secret is None) != (self.registry_password_secret is None):
            raise ValueError("registry_username_secret and registry_password_secret are set together, or both null")
        return self
