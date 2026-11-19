# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared lifecycle stream contracts for Python, OpenAPI and web clients."""

import base64
import binascii
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator


class JobEventType(StrEnum):
    JOB_CREATED = "job.created"
    JOB_STATE_CHANGED = "job.state_changed"
    JOB_DELETED = "job.deleted"
    ATTEMPT_CREATED = "attempt.created"
    ATTEMPT_STATE_CHANGED = "attempt.state_changed"
    ATTEMPT_DELETED = "attempt.deleted"
    STEP_CREATED = "step.created"
    STEP_STATE_CHANGED = "step.state_changed"
    STEP_DELETED = "step.deleted"


class JobEventFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspace_id: str | None = Field(default=None, min_length=1, max_length=255, json_schema_extra={"nullable": True})
    event_types: list[JobEventType] | None = Field(
        default=None, min_length=1, max_length=9, json_schema_extra={"nullable": True}
    )

    @model_validator(mode="after")
    def normalize(self) -> Self:
        if self.event_types is not None:
            self.event_types = sorted(set(self.event_types))
        return self


class JobLifecycleEvent(BaseModel):
    id: str = Field(pattern=r"^[1-9][0-9]*$")
    event_type: JobEventType
    workspace_id: str
    job_id: str
    attempt_id: str | None = Field(json_schema_extra={"nullable": True})
    step_id: str | None = Field(json_schema_extra={"nullable": True})
    previous_state: str | None = Field(json_schema_extra={"nullable": True})
    new_state: str | None = Field(json_schema_extra={"nullable": True})
    occurred_at: datetime
    recorded_at: datetime
    snapshot: dict[str, JsonValue]


class JobEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cursor: str | None = Field(default=None, min_length=1, max_length=8192)
    filters: JobEventFilters | None = None
    start: Literal["oldest", "latest"] | None = None
    direction: Literal["forward", "backward"] | None = None
    limit: int = Field(default=100, ge=1, le=1000, strict=True)
    wait_seconds: float = Field(default=0, ge=0, le=30, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_position(self) -> Self:
        if self.cursor is not None and (self.start is not None or self.direction is not None):
            raise ValueError("cursor cannot be combined with start or direction")
        return self


class JobEventPage(BaseModel):
    data: list[JobLifecycleEvent]
    next_cursor: str
    prev_cursor: str | None = Field(json_schema_extra={"nullable": True})
    has_more: bool


class InvalidJobEventCursor(ValueError):
    pass


class ExpiredJobEventCursor(ValueError):
    pass


class JobEventCursor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    v: Literal[1] = 1
    position: str = Field(pattern=r"^(0|[1-9][0-9]*)$", max_length=19)
    direction: Literal["forward", "backward"] = "forward"
    filters: JobEventFilters = Field(default_factory=JobEventFilters)

    @model_validator(mode="after")
    def validate_bound(self) -> Self:
        if int(self.position) > 2**63 - 1:
            raise ValueError("cursor position exceeds supported range")
        return self

    def encode(self) -> str:
        return base64.urlsafe_b64encode(self.model_dump_json().encode()).decode().rstrip("=")

    @classmethod
    def decode(cls, token: str) -> Self:
        if not token or len(token) > 8192:
            raise InvalidJobEventCursor("invalid cursor length")
        try:
            raw = base64.b64decode(token + "=" * (-len(token) % 4), altchars=b"-_", validate=True)
            return cls.model_validate_json(raw)
        except (binascii.Error, ValueError, ValidationError) as error:
            raise InvalidJobEventCursor("invalid or unsupported cursor") from error


class JobEventErrorDetail(BaseModel):
    code: str
    message: str


class JobEventErrorResponse(BaseModel):
    detail: JobEventErrorDetail
