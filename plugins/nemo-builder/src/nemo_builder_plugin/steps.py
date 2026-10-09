# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The config each build step receives, and where things live on the work volume.

No config carries a credential: ``push`` gets the workspace's, from the platform's Secrets service, in its environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ContextSource(BaseModel):
    """One build context: a whole fileset or a directory within one, or a whole archive in a fileset or a directory
    within that.

    ``fileset`` is always ``<workspace>/<name>``, so no fileset's directory can nest inside another's.
    """

    fileset: str
    #: A tar archive in the fileset, which ``fetch`` unpacks. ``context_path`` is then a directory inside it.
    archive: str | None = None
    context_path: str | None = None

    @field_validator("fileset")
    @classmethod
    def _fileset_is_qualified(cls, value: str) -> str:
        workspace, _, name = value.partition("/")
        if not workspace or not name or "/" in name:
            raise ValueError(f"fileset must be qualified as <workspace>/<name>, got {value!r}")
        return value


def _relative(part: str) -> PurePosixPath:
    path = PurePosixPath(part)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{part!r} is not a relative path inside the work volume")
    return path


def _qualified(fileset: str) -> PurePosixPath:
    path = _relative(fileset)
    if len(path.parts) != 2:
        raise ValueError(f"{fileset!r} is not a qualified fileset reference, <workspace>/<name>")
    return path


@dataclass(frozen=True, slots=True)
class WorkLayout:
    """Where everything lives in one job's directory on the work volume."""

    root: PurePosixPath

    def fileset(self, name: str) -> PurePosixPath:
        return self.root / "context" / _qualified(name)

    @property
    def unpacked(self) -> PurePosixPath:
        return self.root / "unpacked"

    def archive(self, fileset: str, archive: str) -> PurePosixPath:
        """Where ``fetch`` unpacks an archive: apart from the fileset's own files, which may include the archive."""
        return self.unpacked / _qualified(fileset) / _relative(archive)

    @property
    def downloads(self) -> PurePosixPath:
        """Where ``fetch`` keeps an archive until it has unpacked it."""
        return self.root / "downloads"

    def context(self, source: ContextSource) -> PurePosixPath:
        if source.archive is not None:
            path = self.archive(source.fileset, source.archive)
        else:
            path = self.fileset(source.fileset)
        return path / _relative(source.context_path) if source.context_path else path

    @property
    def outputs(self) -> PurePosixPath:
        return self.root / "out"

    def output(self, image: str) -> PurePosixPath:
        """One image's OCI layout, by ``ContainerImage`` row name."""
        return self.outputs / _relative(image)


class FetchStepConfig(BaseModel):
    sources: list[ContextSource] = Field(
        min_length=1, description="Each distinct source, once. An archive is listed once, to unpack whole."
    )


class OpenSandboxServer(BaseModel):
    """An OpenSandbox server whose tenant for the build namespace the build step creates sandboxes as."""

    model_config = ConfigDict(extra="forbid")

    domain: str = Field(
        min_length=1,
        description="The server's host, optionally with a port, as the build step reaches it: its Service's DNS name.",
    )
    protocol: Literal["http", "https"] = Field(default="http", description="How the build step reaches the server.")
    api_key_secret: str = Field(
        default="opensandbox-builder-api-key",
        min_length=1,
        description=(
            "Kubernetes Secret in the build namespace holding the tenant's API key under `api-key`. The build step "
            "reads it with its ServiceAccount, so it's in neither the step's environment nor the sandbox's."
        ),
    )


#: What an OpenSandbox sandbox may reach unless the deployment says otherwise: public names under these, which
#: cover the common registries and package indexes. Nothing in a private range is reachable whatever this says.
DEFAULT_EGRESS_ALLOW = ["*.com", "*.org", "*.io", "*.dev"]


class SandboxSpec(BaseModel):
    """How ``supervise`` runs the sandboxes. All of it comes from operator config."""

    image: str = Field(description="The kaniko image.")
    provider: Literal["kubernetes_pod", "opensandbox"] = "kubernetes_pod"
    opensandbox: OpenSandboxServer | None = None
    work_pvc: str
    node_selector: dict[str, str]
    dns_nameservers: list[str]
    cpu: str
    memory: str
    #: Secrets the kubelet pulls the kaniko image with. Unset in a spec from before it existed: none.
    image_pull_secrets: list[str] = Field(default_factory=list)
    #: Unset in a spec from before it existed: neither requested nor limited.
    ephemeral_storage: str | None = None
    #: A spec from before it existed gets the default, rather than an unrestricted sandbox.
    egress_allow: list[str] = Field(default_factory=lambda: list(DEFAULT_EGRESS_ALLOW))


class SandboxImage(BaseModel):
    image: str = Field(description="The `ContainerImage` row name.")
    platform: str
    dockerfile: str = Field(description="Dockerfile path, relative to the group's context.")


class SandboxGroup(BaseModel):
    source: ContextSource
    images: list[SandboxImage] = Field(min_length=1)


class SuperviseStepConfig(BaseModel):
    sandbox: SandboxSpec
    groups: list[SandboxGroup] = Field(min_length=1)


#: Where ``push`` finds the workspace's registry credential and signing key, which Jobs fetches as the submitter.
REGISTRY_USERNAME_ENV = "NHX_BUILD_REGISTRY_USERNAME"
REGISTRY_PASSWORD_ENV = "NHX_BUILD_REGISTRY_PASSWORD"
SIGNING_KEY_ENV = "NHX_BUILD_SIGNING_KEY"


class PushImage(BaseModel):
    image: str = Field(description="The `ContainerImage` row name, which the builder's routes take.")
    system_ref: str = Field(description="`<registry>/<repository>:<system tag>`, which every build pushes.")
    caller_ref: str | None = Field(
        default=None, description="`<registry>/<repository>:<output.tag>`, when the spec names an output."
    )

    @property
    def refs(self) -> list[str]:
        """Where the image is pushed: the caller's tag, when the spec names an output, then the system tag."""
        return [*([self.caller_ref] if self.caller_ref else []), self.system_ref]


class PushStepConfig(BaseModel):
    images: list[PushImage] = Field(min_length=1)
    registry: str = Field(description="The registry host. Every reference in `images` is on it.")
    plain_http: bool = Field(default=False, description="Reach the registry over plain HTTP.")
    log_in: bool = Field(
        default=True,
        description=f"Log in with `{REGISTRY_USERNAME_ENV}` and `{REGISTRY_PASSWORD_ENV}`; otherwise push anonymously.",
    )
    sign: bool = Field(default=True, description=f"Sign each image with `{SIGNING_KEY_ENV}`.")

    @model_validator(mode="after")
    def _every_ref_is_on_the_registry(self) -> PushStepConfig:
        """The credential is for ``registry`` only, so a reference elsewhere would push anonymously."""
        for image in self.images:
            for ref in image.refs:
                if not ref.startswith(f"{self.registry}/"):
                    raise ValueError(f"{ref} is not on {self.registry}")
        return self
