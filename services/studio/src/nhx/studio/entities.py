# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity Store models owned by the Studio service."""

from typing import ClassVar, Literal

from nhx.common.entities.client import EntityBase
from nhx.studio.assistant_artifacts import ChatArtifactsResponse
from pydantic import BaseModel, Field


class AssistantMessage(BaseModel):
    """One user or assistant message in a persisted Assistant conversation."""

    role: Literal["user", "assistant"]
    content: str


class AssistantConversation(EntityBase):
    """A workspace-scoped, user-owned NeMo Assistant conversation."""

    __entity_type__: ClassVar[str] = "assistant_conversation"

    session_id: str = Field(description="Stable Studio session UUID exposed to the UI.")
    owner_id: str = Field(description="Principal that owns and may read this conversation.")
    messages: list[AssistantMessage] = Field(default_factory=list)
    chat_artifacts: ChatArtifactsResponse = Field(default_factory=ChatArtifactsResponse)


class LegacyAssistantConversation(AssistantConversation):
    """Read-compatible model for conversations persisted before the rename."""

    __entity_type__: ClassVar[str] = "copilot_conversation"
