# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned entity keyset tokens, following the Jobs logs cursor pattern.

Tokens are opaque continuation positions, not credentials or snapshot handles.
Authorization and filters must still be applied independently on every request.
"""

import hashlib
import json
from typing import Literal, Self

import base58
from nhx.common.api.filter import FilterOperation
from pydantic import BaseModel, ConfigDict, Field, NaiveDatetime, ValidationError

MAX_PAGE_TOKEN_LENGTH = 4096
CURSOR_SORTS = {"created_at", "-created_at"}


class EntityPageToken(BaseModel):
    """The last emitted (UTC timestamp, ID) plus the effective query scope."""

    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1] = Field(alias="v")
    created_at: NaiveDatetime = Field(alias="t")
    entity_id: str = Field(alias="i", min_length=1, max_length=255)
    scope_hash: str = Field(alias="q", pattern=r"^[0-9a-f]{64}$")

    def encode(self) -> str:
        return base58.b58encode(self.model_dump_json(by_alias=True).encode()).decode()

    @classmethod
    def decode(cls, token: str, *, scope_hash: str) -> Self:
        try:
            if len(token) > MAX_PAGE_TOKEN_LENGTH:
                raise ValueError("Token too long")
            position = cls.model_validate_json(base58.b58decode(token))
        except (ValueError, TypeError, ValidationError) as exc:
            raise ValueError("Invalid page_token") from exc
        if position.scope_hash != scope_hash:
            raise ValueError("page_token does not match the current entity query")
        return position


def entity_page_scope(
    *,
    workspace: str,
    entity_type: str | None,
    sort: str,
    filter_op: FilterOperation | None,
    relationship_child_workspaces: set[str] | None,
) -> str:
    """Bind a token to the effective filters, including authorization scope.

    As in Jobs logs, a scope hash detects accidental reuse with other filters.
    It is not a signature and is never used to grant access. Page size can change.
    """
    scope = {
        "workspace": workspace,
        "entity_type": entity_type,
        "sort": sort,
        "filter": filter_op.to_dict() if filter_op is not None else None,
        "relationship_child_workspaces": (
            sorted(relationship_child_workspaces) if relationship_child_workspaces is not None else None
        ),
    }
    return hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
