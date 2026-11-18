# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact extraction for Studio assistant chat history."""

from pydantic import AliasChoices, BaseModel, Field


class ChatSelectionArtifactResponse(BaseModel):
    """A user selection captured during the chat."""

    label: str
    value: str


class ChatFileArtifactResponse(BaseModel):
    """A file touched by the local assistant."""

    action: str
    path: str


class ChatLinkArtifactResponse(BaseModel):
    """A Studio link requested by the local assistant."""

    label: str
    destination: str | None = None
    href: str | None = None


class ChatJobArtifactResponse(BaseModel):
    """A Studio job referenced during the chat."""

    name: str
    job_type: str | None = None
    source: str | None = None
    href: str | None = None


class ChatArtifactsResponse(BaseModel):
    """Structured chat metadata shown in Studio's artifacts pane."""

    agent: str | None = None
    model: str | None = None
    model_source: str | None = None
    assistant_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("assistant_model", "copilot_model"),
    )
    workspace: str | None = None
    selections: list[ChatSelectionArtifactResponse] = Field(default_factory=list)
    files: list[ChatFileArtifactResponse] = Field(default_factory=list)
    links: list[ChatLinkArtifactResponse] = Field(default_factory=list)
    jobs: list[ChatJobArtifactResponse] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)


def record_assistant_model(artifacts: ChatArtifactsResponse, model: str | None) -> None:
    if not model:
        return
    artifacts.assistant_model = model
