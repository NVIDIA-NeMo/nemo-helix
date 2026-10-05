# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime
from enum import Enum
from typing import Literal

import base58
from nemo_helix_plugin.jobs.schemas import FileStorageType as FileStorageType
from nemo_helix_plugin.jobs.schemas import HelixJobListResultResponse as HelixJobListResultResponse
from nemo_helix_plugin.jobs.schemas import HelixJobLog as HelixJobLog
from nemo_helix_plugin.jobs.schemas import HelixJobLogPage as HelixJobLogPage
from nemo_helix_plugin.jobs.schemas import HelixJobResultCreateRequest as HelixJobResultCreateRequest
from nemo_helix_plugin.jobs.schemas import HelixJobResultResponse as HelixJobResultResponse
from nemo_helix_plugin.jobs.schemas import HelixJobStatus as HelixJobStatus
from nemo_helix_plugin.jobs.schemas import HelixJobStatusResponse as HelixJobStatusResponse
from nemo_helix_plugin.jobs.schemas import HelixJobStepStatusResponse as HelixJobStepStatusResponse
from nemo_helix_plugin.jobs.schemas import HelixJobTaskStatusResponse as HelixJobTaskStatusResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# =============================================================================
# Pagination (stays in nhx-common — depends on base58)
# =============================================================================


class PaginationDirection(int, Enum):
    """Direction for cursor-based pagination."""

    FORWARD = 0
    BACKWARD = 1


class InvalidPageCursorError(ValueError):
    """Custom exception for invalid page cursor errors."""


class PageCursor(BaseModel):
    """Schema for cursor-based pagination."""

    start_id: int = Field(description="The ID to start pagination from")
    direction: PaginationDirection = Field(description="The direction of pagination")

    def encode(self) -> str:
        """Encode a page cursor from start_id and direction using compact tuple format with base58."""
        cursor_data = [self.start_id, self.direction]
        json_str = json.dumps(cursor_data)
        encoded = base58.b58encode(json_str.encode()).decode()
        return encoded

    @staticmethod
    def decode(page_cursor: str) -> "PageCursor":
        """Decode a page cursor to get PageCursor object using Pydantic schema with base58."""
        try:
            decoded = base58.b58decode(page_cursor.encode()).decode()
            start_id, direction_int = json.loads(decoded)
            direction = PaginationDirection(direction_int)
            return PageCursor(start_id=start_id, direction=direction)
        except (ValueError, TypeError, ValidationError) as e:
            raise InvalidPageCursorError("Invalid page cursor") from e


class LogPageCursorV1(BaseModel):
    """Versioned cursor for fetching the page preceding a returned boundary."""

    model_config = ConfigDict(populate_by_name=True)

    version: Literal[1] = Field(default=1, validation_alias="v", serialization_alias="v")
    boundary_timestamp: datetime = Field(validation_alias="t", serialization_alias="t")
    boundary_row_hash: str = Field(validation_alias="r", serialization_alias="r", min_length=32, max_length=32)
    query_scope_hash: str = Field(validation_alias="q", serialization_alias="q", min_length=32, max_length=32)
    emitted_boundary_rows: int = Field(default=0, validation_alias="e", serialization_alias="e", ge=0)

    def encode(self) -> str:
        """Encode the cursor using compact aliases and base58 JSON."""
        cursor_data = self.model_dump(mode="json", by_alias=True)
        json_str = json.dumps(cursor_data, separators=(",", ":"))
        return base58.b58encode(json_str.encode()).decode()


LogPageCursorV0 = PageCursor
LogPageCursor = LogPageCursorV0 | LogPageCursorV1


def decode_log_page_cursor(page_cursor: str) -> LogPageCursor:
    """Decode either a v0 page-number cursor or a v1 boundary cursor."""
    try:
        decoded = base58.b58decode(page_cursor.encode()).decode()
        payload = json.loads(decoded)
        if isinstance(payload, list):
            start_id, direction_int = payload
            return LogPageCursorV0(start_id=start_id, direction=PaginationDirection(direction_int))
        if isinstance(payload, dict) and payload.get("v") == 1:
            return LogPageCursorV1.model_validate(payload)
    except (ValueError, TypeError, ValidationError) as e:
        raise InvalidPageCursorError("Invalid page cursor") from e
    raise InvalidPageCursorError("Invalid page cursor")
