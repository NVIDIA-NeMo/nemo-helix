# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request and response bodies of helix-operator's routes."""

from typing import Literal

from nemo_helix_plugin.entity_naming import NAME_PATTERN
from pydantic import BaseModel, ConfigDict, Field


class BuildSource(BaseModel):
    """An uploaded checkout: an archive in a fileset of this workspace, next to its ``source.json``."""

    model_config = ConfigDict(extra="forbid", regex_engine="python-re")

    fileset: str = Field(pattern=NAME_PATTERN, description="The fileset `upload` wrote, in this workspace.")
    archive: str = Field(default="source.tar.gz", description="The archive in the fileset to build from.")


class CreateBuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: BuildSource


class SourceInfo(BaseModel):
    """``source.json``, which ``upload`` writes beside the archive."""

    commit: str = Field(pattern=r"^[0-9a-f]{12}$", description="The checkout's HEAD, first 12 characters.")
    dirty: bool = Field(description="Whether the checkout had changes not in HEAD.")


class HelixBuild(BaseModel):
    name: str = Field(description="The builder image name, which `GET /builds/{name}` takes.")
    status: Literal["queued", "pending", "ready", "failed"]
    tag: str = Field(description="`<commit>-r<revision>`.")
    api_image: str | None = Field(default=None, description="`<registry>/<repository>:<tag>`, once ready.")
    core_image: str | None = Field(default=None, description="The same image: one image serves api and core.")
    digest: str | None = None
    job: str | None = Field(default=None, description="`<workspace>/<name>` of the build job.")
    detail: str | None = None
