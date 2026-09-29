# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The config each build step receives, and where things live on the work volume.

The compiler writes these configs, and each step reads its own with
``Model.model_validate(get_task_config())``. No config carries a credential: ``push`` gets registry
tokens and signatures from the credential broker. ``supervise`` isn't told where images are pushed,
and ``push`` isn't told what they were built from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from pydantic import BaseModel, Field, field_validator, model_validator


class ContextSource(BaseModel):
    """One build context: a whole fileset, or a directory within one.

    ``fileset`` is always ``<workspace>/<name>``, so that no fileset's directory on the work volume
    can nest inside another's.
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
    """``part``, refused if it is absolute or climbs with ``..``: joined to a path, either would leave it."""
    path = PurePosixPath(part)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{part!r} is not a relative path inside the work volume")
    return path


@dataclass(frozen=True, slots=True)
class WorkLayout:
    """Where everything lives in one job's directory on the work volume.

    Each step roots the layout where it sees that directory, so ``fetch``, the sandbox and ``push``
    agree on every path beneath it.
    """

    root: PurePosixPath

    def fileset(self, name: str) -> PurePosixPath:
        """Where ``fetch`` downloads the fileset ``<workspace>/<name>``."""
        path = _relative(name)
        if len(path.parts) != 2:
            raise ValueError(f"{name!r} is not a qualified fileset reference, <workspace>/<name>")
        return self.root / "context" / path

    def context(self, source: ContextSource) -> PurePosixPath:
        """A build context: its fileset's directory, or a directory within it."""
        path = self.fileset(source.fileset)
        return path / _relative(source.context_path) if source.context_path else path

    @property
    def outputs(self) -> PurePosixPath:
        """Where the sandbox writes OCI layouts, and ``push`` reads them."""
        return self.root / "out"

    def output(self, image: str) -> PurePosixPath:
        """One image's OCI layout, by ``ContainerImage`` row name."""
        return self.outputs / _relative(image)


class FetchStepConfig(BaseModel):
    """Config for ``fetch``, which downloads each build context as the submitter."""

    sources: list[ContextSource] = Field(min_length=1, description="Each distinct source, once.")


class SandboxSpec(BaseModel):
    """How ``supervise`` builds the sandbox pod. All of it comes from operator config."""

    image: str = Field(description="The kaniko image.")
    namespace: str
    work_pvc: str
    node_selector: dict[str, str]
    dns_nameservers: list[str]
    cpu: str
    memory: str


class SandboxImage(BaseModel):
    """One kaniko invocation."""

    image: str = Field(description="The `ContainerImage` row name.")
    platform: str
    dockerfile: str = Field(description="Dockerfile path, relative to the group's context.")


class SandboxGroup(BaseModel):
    """One sandbox pod: the images built from one source."""

    source: ContextSource
    images: list[SandboxImage] = Field(min_length=1)


class SuperviseStepConfig(BaseModel):
    """Config for ``supervise``, which runs the sandbox pods. It holds no credential."""

    sandbox: SandboxSpec
    groups: list[SandboxGroup] = Field(min_length=1)


class PushImage(BaseModel):
    """One OCI layout to publish."""

    image: str = Field(description="The `ContainerImage` row name, which is all the broker needs to sign it.")
    system_ref: str = Field(description="`<registry>/<repository>:<system tag>`, the reference the broker signs.")
    caller_ref: str | None = Field(
        default=None, description="`<registry>/<repository>:<output.tag>`, when the spec names an output."
    )

    @property
    def refs(self) -> list[str]:
        """The references to push. The system tag goes last, since it is the one that gets signed."""
        return [*([self.caller_ref] if self.caller_ref else []), self.system_ref]


class PushStepConfig(BaseModel):
    """Config for ``push``, which publishes the sandbox's output. It treats the layouts as untrusted."""

    images: list[PushImage] = Field(min_length=1)
    registry: str = Field(description="The registry host. Every reference in `images` is on it.")
    broker: str = Field(description="URL of the credential broker.")
    plain_http: bool = Field(default=False, description="Reach the registry over plain HTTP.")

    @model_validator(mode="after")
    def _every_ref_is_on_the_registry(self) -> PushStepConfig:
        """The broker's tokens are for ``registry`` only, so a reference elsewhere would push anonymously."""
        for image in self.images:
            for ref in image.refs:
                if not ref.startswith(f"{self.registry}/"):
                    raise ValueError(f"{ref} is not on {self.registry}")
        return self
