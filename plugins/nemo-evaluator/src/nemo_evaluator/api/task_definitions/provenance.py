# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stored origins for resolved task definitions."""

from nemo_evaluator.api.fields import TaskRef
from nemo_evaluator.content_hash import DIGEST_LENGTH, DIGEST_PATTERN
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskProvenance(BaseModel):
    """The digest identifies the published revision, not its expanded job snapshot."""

    model_config = ConfigDict(extra="forbid")
    entity_name: str
    revision_digest: str = Field(pattern=DIGEST_PATTERN, min_length=DIGEST_LENGTH, max_length=DIGEST_LENGTH)

    @field_validator("entity_name")
    @classmethod
    def _identity(cls, value: str) -> str:
        if "#" in value or "/" not in value:
            raise ValueError("entity_name must be qualified without a revision fragment")
        TaskRef(value)
        return value
