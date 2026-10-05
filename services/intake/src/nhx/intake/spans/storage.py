# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for Intake span domain mapping."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from nhx.common.api.common import PaginationData
from nhx.intake.spans.domain import (
    INTAKE_PREVIEW_PAYLOAD_CHAR_LIMIT,
    IntakeResponseMode,
    SpanKind,
    SpanStatus,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stable_id(*parts: str, prefix: str | None = None) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        hasher.update(encoded)
    digest = hasher.hexdigest()
    return f"{prefix}-{digest[:32]}" if prefix else digest[:32]


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def json_dumps_preserve(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def json_loads_or_none(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def truncated_text_select(
    column_sql: str,
    *,
    alias: str,
    limit_parameter: str = "payload_char_limit",
) -> str:
    """Build a ClickHouse select expression that bounds a UTF-8 text column."""
    return f"substringUTF8({column_sql}, 1, %({limit_parameter})s) AS {alias}"


def text_select_for_mode(column_sql: str, *, alias: str, mode: IntakeResponseMode) -> str:
    """Project payload text without reading it for summary responses."""
    if mode == "summary":
        return f"'' AS {alias}"
    if mode == "preview":
        return truncated_text_select(column_sql, alias=alias)
    if mode == "detailed":
        return f"{column_sql} AS {alias}"
    raise ValueError(f"Unsupported Intake response mode: {mode}")


def text_query_parameters(mode: IntakeResponseMode) -> dict[str, int]:
    """Return the query parameters required by a mode's text projection."""
    if mode == "preview":
        return {"payload_char_limit": INTAKE_PREVIEW_PAYLOAD_CHAR_LIMIT}
    if mode in {"summary", "detailed"}:
        return {}
    raise ValueError(f"Unsupported Intake response mode: {mode}")


def text_for_mode(value: str | None, *, mode: IntakeResponseMode) -> str | None:
    """Enforce payload mode semantics again at the API response boundary."""
    if value is None or mode == "summary":
        return None
    if mode == "detailed":
        return value
    if mode == "preview":
        return value[:INTAKE_PREVIEW_PAYLOAD_CHAR_LIMIT]
    raise ValueError(f"Unsupported Intake response mode: {mode}")


def normalize_span_kind(value: Any) -> SpanKind:
    if value is None:
        return SpanKind.UNKNOWN
    try:
        return SpanKind(str(value).upper())
    except ValueError:
        return SpanKind.UNKNOWN


def normalize_span_status(value: Any) -> SpanStatus:
    if value is None:
        return SpanStatus.UNKNOWN
    try:
        return SpanStatus(str(value).lower())
    except ValueError:
        return SpanStatus.UNKNOWN


def make_pagination(*, page: int, page_size: int, current_page_size: int, total_results: int) -> PaginationData:
    if page < 1:
        raise ValueError("page must be >= 1")
    if page_size < 1:
        raise ValueError("page_size must be >= 1")
    if current_page_size < 0 or current_page_size > page_size:
        raise ValueError("current_page_size must be between 0 and page_size")
    if total_results < 0:
        raise ValueError("total_results must be >= 0")
    return PaginationData(
        page=page,
        page_size=page_size,
        current_page_size=current_page_size,
        total_results=total_results,
        total_pages=(total_results + page_size - 1) // page_size,
    )


def dict_to_row(row: dict[str, Any], columns: Sequence[str]) -> list[Any]:
    return [row.get(column) for column in columns]


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def str_or_none(value: Any) -> str | None:
    # ClickHouse String columns use "" as the null sentinel; treat it the same as SQL NULL.
    if value is None or value == "":
        return None
    return str(value)
