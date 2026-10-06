# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The config each build step receives, and where things live on the work volume.

No config carries a credential: ``push`` gets the workspace's, from the platform's Secrets service, in its environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from pydantic import BaseModel, Field, field_validator, model_validator


class ContextSource(BaseModel):
    """One build context: a whole fileset, or a directory within one.

    ``fileset`` is always ``<workspace>/<name>``, so no fileset's directory can nest inside another's.
    """

    fileset: str
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


@dataclass(frozen=True, slots=True)
class WorkLayout:
    """Where everything lives in one job's directory on the work volume."""

    root: PurePosixPath

    def fileset(self, name: str) -> PurePosixPath:
        path = _relative(name)
        if len(path.parts) != 2:
            raise ValueError(f"{name!r} is not a qualified fileset reference, <workspace>/<name>")
        return self.root / "context" / path

    def context(self, source: ContextSource) -> PurePosixPath:
        path = self.fileset(source.fileset)
        return path / _relative(source.context_path) if source.context_path else path

    @property
    def outputs(self) -> PurePosixPath:
        return self.root / "out"

    def output(self, image: str) -> PurePosixPath:
        """One image's OCI layout, by ``ContainerImage`` row name."""
        return self.outputs / _relative(image)


class FetchStepConfig(BaseModel):
    sources: list[ContextSource] = Field(min_length=1, description="Each distinct source, once.")


class SandboxSpec(BaseModel):
    """How ``supervise`` builds the sandbox pod. All of it comes from operator config."""

    image: str = Field(description="The kaniko image.")
    work_pvc: str
    node_selector: dict[str, str]
    dns_nameservers: list[str]
    cpu: str
    memory: str


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
