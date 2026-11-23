# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator configuration for the builder: the ``builder:`` section, or ``NEMO_BUILDER_*``.

How a build runs is set here, never in the request.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Self

from nemo_builder_plugin.identity import (
    PLATFORM_PATTERN,
    ImageIdentityError,
    normalize_registry,
    platform_name,
    validate_registry_host,
    validate_repository,
)
from nemo_helix_plugin.config import NemoConfig
from nemo_helix_plugin.refs import parse_entity_ref
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)


class SandboxConfig(BaseModel):
    """The pods a build's Dockerfiles run in.

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


class RemoteConfig(BaseModel):
    """The `remote` backend: a build service outside this cluster builds and publishes each image.

    Nothing here names a service. The step image holds the service's client; its Jobs profile holds whatever that
    client reads from its environment. From the environment, the section is one JSON value, ``NEMO_BUILDER_REMOTE``.
    """

    model_config = ConfigDict(extra="forbid")

    registry: str | None = Field(
        default=None,
        description=(
            "Host of the registry the service publishes every image to. Unset, with `backend: remote`, every "
            "submit is refused."
        ),
    )
    repository: str | None = Field(
        default=None,
        description=(
            "The one repository the service publishes to, in every workspace's name: an image is told apart by "
            "its digest. Unset, with `backend: remote`, every submit is refused."
        ),
    )
    profile: str = Field(
        default="build-remote",
        description=(
            "The Jobs execution profile, provider `cpu` on the `kubernetes_job` backend, the remote-build step runs "
            "on. Its `max_active_jobs` caps the build jobs that run at once. A job is one set, built an image at a "
            "time, so running jobs wait on at most that many builds at the service: set it to what the service lets "
            "this deployment queue. Its `ttl_seconds_active` is the only time limit Jobs puts on the step once its "
            "pod is running, and Jobs counts it from when the build was submitted: it must cover the longest a "
            "build waits for a slot, then `build_timeout_seconds` for each image. A profile whose limit is below "
            "the builds' own, plus two minutes, is refused. That check is a floor: it leaves out the time to fetch "
            "each build context and, once an image is built, up to five minutes of reporting it to the builder."
        ),
    )
    image: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The image the remote-build step runs: one that holds the service's step program. Unset, the profile's "
            "`default_task_image`, which must then be set."
        ),
    )
    command: list[str] = Field(
        default_factory=lambda: ["remote-build"],
        min_length=1,
        description="The step program in that image, and its arguments.",
    )

    @field_validator("image")
    @classmethod
    def _image_is_named(cls, value: str | None) -> str | None:
        """Without surrounding spaces, and not blank: the job would be created, and its image never pulled."""
        if value is None:
            return None
        if not value.strip():
            raise ValueError("remote.image is blank")
        return value.strip()

    @field_validator("command")
    @classmethod
    def _command_has_no_blank_part(cls, value: list[str]) -> list[str]:
        if any(not part.strip() for part in value):
            raise ValueError("remote.command has a blank entry")
        return value

    credential_secret: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The platform secret holding the service's credential, which Jobs fetches as the build's submitter and "
            "the step reads from its environment, whether or not the platform uses workload token exchange. Write "
            "it as `<workspace>/<name>` for one credential the whole deployment builds with: a bare name is read "
            "from the workspace the build is submitted in, so every workspace that builds would need its own. Every "
            "user who may build must be able to read it. Null when the step's Jobs profile mounts the credential "
            "instead: any user who may create a job can name that profile, and read a credential it mounts."
        ),
    )
    platforms: list[str] = Field(
        default_factory=lambda: ["linux/amd64"],
        min_length=1,
        description="The platforms the service builds, as `os/architecture`. A spec for any other is refused.",
    )
    max_specs_per_set: int = Field(
        default=1, ge=1, description="How many images one build set may hold. A larger set is refused."
    )
    allow_dockerfile_path: bool = Field(
        default=True,
        description=(
            "Take a spec whose Dockerfile isn't `Dockerfile` at the context root: the step puts that file at the "
            "root of the context it hands the service, in place of any `Dockerfile` already there. A build that "
            "copies its own root `Dockerfile` into the image then copies the named one. The Dockerfile's own "
            "ignore file (`<its path>.dockerignore`) goes to the root with it. False refuses such a spec."
        ),
    )
    build_timeout_seconds: int = Field(
        default=3600,
        ge=60,
        description=(
            "How long the step waits for one image, from when it asks the service to start, before it cancels the "
            "build and fails the image. Time the service makes the step wait for its queue counts. The step gives "
            "the same limit to the service's client, to pass to its service as the build's own deadline: only a "
            "service that enforces one ends a build whose step was killed."
        ),
    )
    allow_unverified_completion: bool = Field(
        default=False,
        description=(
            "Build where nothing can show who completes an image: with platform auth on and workload token "
            "exchange off. The platform then takes a request's identity from headers that anything able to reach "
            "its API can set, so the builder cannot tell an image's own remote-build step from any other caller, "
            "and by default it refuses every submit and every completion of a remote build. True accepts "
            "exactly that: anything able to reach the API can make a `pending` image `ready` at a digest, with a "
            "build and a runtime configuration, of its choosing. It changes nothing with platform auth off, "
            "where no caller is checked, nor with workload token exchange on, where only a step of the job the "
            "image's submit created completes it."
        ),
    )

    @field_validator("registry")
    @classmethod
    def _registry_is_a_host(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return validate_registry_host(value)
        except ImageIdentityError as exc:
            # Not the value: a registry given with a login in it is a credential, and this is logged.
            raise ValueError("remote.registry must be a host, optionally with a port") from exc

    @field_validator("repository")
    @classmethod
    def _repository_is_a_path(cls, value: str | None) -> str | None:
        return None if value is None else validate_repository(value.strip("/"))

    @field_validator("credential_secret")
    @classmethod
    def _credential_secret_is_a_reference(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                # As Jobs reads it when it creates the job, which is after the rows are written.
                parse_entity_ref(value, default_workspace="any")
            except ValueError as exc:
                raise ValueError("remote.credential_secret must be `<name>` or `<workspace>/<name>`") from exc
            # Without what Jobs trims: the builder's own check of the secret splits the reference as it is.
            return value.strip()
        return value

    @field_validator("platforms", mode="before")
    @classmethod
    def _platform_names(cls, value: Any) -> Any:
        return [platform_name(item) for item in value] if isinstance(value, list) else value


class BuilderConfig(NemoConfig):
    """Configuration for container image builds."""

    plugin_name: ClassVar[str] = "builder"
    plugin_description: ClassVar[str] = "Container image builds for NeMo Helix."

    backend: str = Field(
        default="execution",
        description=(
            "Which backend builds: `execution`, in this cluster, or `remote`, through a build service outside it "
            "(see `remote`). Any other value refuses every submit."
        ),
    )
    remote: RemoteConfig = Field(default_factory=RemoteConfig)

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

    max_concurrent_builds: int | None = Field(
        default=None,
        deprecated=True,
        description=(
            "Deprecated and ignored since 0.11: Jobs caps builds now. Set the build profiles' `max_active_jobs` "
            "instead, which the Helm chart's `builder.maxConcurrentBuilds` does."
        ),
    )

    registry: str | None = Field(
        default=None,
        description=(
            "Host of the registry the in-cluster build publishes every image to, with an optional port; prefix "
            "it with `http://` for plain HTTP. Unset, with `backend: execution`, every submit is refused."
        ),
    )
    repository_prefix: str = Field(
        default="",
        description=(
            "Path the in-cluster build publishes every image under, before `<workspace>/`. For GAR, "
            "`<project>/<repository>`; for Artifactory, the repository key. Not used with `backend: remote`."
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
                    # Not what was given: a registry written with a login in it is a credential.
                    raise ValueError("registry must be a host, optionally prefixed with http:// or https://")
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
            raise ValueError("registry must be a host without a path. Put the path in `repository_prefix`.")
        try:
            return validate_registry_host(value)
        except ImageIdentityError as exc:
            # Not the value: a registry given with a login in it is a credential, and this is logged.
            raise ValueError("registry must be a host, optionally with a port") from exc

    @field_validator("repository_prefix")
    @classmethod
    def _prefix_is_a_repository_path(cls, value: str) -> str:
        value = value.strip("/")
        return validate_repository(value) if value else value

    default_platform: str = Field(
        default="linux/amd64",
        pattern=PLATFORM_PATTERN,
        description=(
            "The platform a build or copy is for when its request names none, as `os/architecture`. Set it to your "
            "nodes' architecture: `linux/arm64` for a cluster on Apple silicon, such as Docker Desktop on a Mac. "
            "Changing it doesn't rebuild task images evals already has: request their builds with `rebuild` set."
        ),
    )

    import_registries: list[str] = Field(
        default_factory=lambda: ["docker.io", "ghcr.io", "nvcr.io", "quay.io"],
        description=(
            "Registries a build may copy a published image from, as hosts. A copy from any other registry is "
            "refused with `409`. Empty refuses every copy. Images are read anonymously, so only public ones copy."
        ),
    )

    _default_platform_name = field_validator("default_platform", mode="before")(platform_name)

    @field_validator("import_registries")
    @classmethod
    def _import_registries_are_hosts(cls, value: list[str]) -> list[str]:
        try:
            return sorted({normalize_registry(host) for host in value})
        except ImageIdentityError as exc:
            # Not the host it refused: a registry given with a login in it is a credential, and this is logged.
            # Nor an example of one: the log takes every entry of the list out of this sentence.
            raise ValueError("import_registries must be registry hosts, with no scheme, login or path") from exc

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
    def _warn_max_concurrent_builds(self) -> Self:
        # Read from the field set, not the attribute: reading a deprecated field warns on every read.
        if "max_concurrent_builds" in self.model_fields_set:
            logger.warning(
                "builder.max_concurrent_builds is deprecated and ignored: set the build profiles' max_active_jobs "
                "instead (the Helm chart's builder.maxConcurrentBuilds does)"
            )
        return self

    @model_validator(mode="after")
    def _warn_unknown_settings(self) -> Self:
        # The section takes keys it doesn't know, so a misspelt one changes nothing and says nothing: with
        # `bakend: remote`, builds would stay in this cluster. Warned of, not refused: a retired key must not
        # stop a deployment building.
        if self.model_extra:
            logger.warning("builder: these settings are not known, and are ignored: %s", ", ".join(self.model_extra))
        return self

    @model_validator(mode="after")
    def _credential_is_whole(self) -> Self:
        if (self.registry_username_secret is None) != (self.registry_password_secret is None):
            raise ValueError("registry_username_secret and registry_password_secret are set together, or both null")
        return self
