# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator configuration for the builder: the ``builder:`` section, or ``NEMO_BUILDER_*``.

How a build runs is set here, never in the request. Settings with no safe default are ``None``,
and every submit is refused while one is unset. The credential broker reads the same section, so
the two agree on the registry, the prefix and the push step's identity.
"""

from __future__ import annotations

from typing import Any, ClassVar, Self
from urllib.parse import urlsplit

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from nemo_builder_plugin.identity import ImageIdentityError, validate_registry_host, validate_repository
from nemo_helix_plugin.config import NemoConfig
from pydantic import Field, ModelWrapValidatorHandler, PrivateAttr, field_validator, model_validator


class BuilderConfig(NemoConfig):
    """Configuration for in-cluster container image builds."""

    plugin_name: ClassVar[str] = "builder"
    plugin_description: ClassVar[str] = "In-cluster container image builds for NeMo Helix."

    # --- Where builds run ---

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
        description="Resolvers the sandbox uses instead of cluster DNS, which its network policy blocks.",
    )

    # --- Jobs execution profiles ---

    fetch_profile: str = Field(default="build-fetch")
    control_profile: str = Field(default="build-control")
    push_profile: str = Field(default="build-push")
    push_service_account: str = Field(
        default="nhx-build-push",
        description="The ServiceAccount `push_profile` runs as. The broker serves push pods only as this account.",
    )

    # --- Publishing ---

    registry: str | None = Field(
        default=None,
        description=(
            "Host of the registry every image is published to, with an optional port. Prefix it with "
            "`http://` for a registry served over plain HTTP. The registry must accept nested "
            "repository paths and create a repository on its first push, as Artifactory, GAR, Harbor "
            "and Distribution do. Unset refuses every submit."
        ),
    )
    repository_prefix: str = Field(
        default="",
        description=(
            "Path every image is published under: `<repository_prefix>/<workspace>/<output.repository>`, "
            "or `<repository_prefix>/<workspace>/<set>/<spec>` when a spec names no output. For "
            "Artifactory, the repository key; for GAR, `<project>/<repository>`."
        ),
    )
    registry_narrows_scope: bool = Field(
        default=False,
        description=(
            "Whether the registry's token service limits a token to the scope asked for. When set, the "
            "broker refuses a token whose scope is wider than it asked for."
        ),
    )

    _registry_plain_http: bool = PrivateAttr(default=False)

    @model_validator(mode="wrap")
    @classmethod
    def _registry_scheme(cls, data: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        """Strip an ``http://`` or ``https://`` from ``registry``, remembering which it was."""
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
        """Whether ``registry`` was given as ``http://...``."""
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

    # --- The credential broker, and the key signatures are verified against ---

    credential_broker: str | None = Field(
        default=None,
        description=(
            "URL of the credential broker, e.g. `http://nhx-build-broker.nhx-build-broker.svc:8080`. "
            "The push step gets its registry tokens and signatures from it. The broker speaks plain "
            "HTTP, so put TLS in front of it where platform auth is on. Unset refuses every submit."
        ),
    )
    signing_public_key: str | None = Field(
        default=None,
        description=(
            "PEM of the public key the broker signs with. A row becomes `ready` only on a signature "
            "this key verifies. Unset refuses every submit."
        ),
    )

    @field_validator("credential_broker")
    @classmethod
    def _broker_is_a_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        url = urlsplit(value)
        if url.scheme not in ("http", "https") or not url.netloc or url.query or url.fragment:
            raise ValueError(f"credential_broker must be an http(s) URL with no query, got {value!r}")
        return value.rstrip("/")

    @field_validator("signing_public_key")
    @classmethod
    def _public_key_is_a_public_key(cls, value: str | None) -> str | None:
        """Parsed here, so a bad key stops the platform from starting rather than failing every build."""
        if value is None:
            return None
        try:
            key = serialization.load_pem_public_key(value.encode())
        except ValueError as exc:
            raise ValueError("signing_public_key must be a PEM public key (cosign.pub)") from exc
        if not isinstance(key, ec.EllipticCurvePublicKey | rsa.RSAPublicKey):
            raise ValueError("signing_public_key must be an EC or RSA key, which is what cosign signs with")
        return value
