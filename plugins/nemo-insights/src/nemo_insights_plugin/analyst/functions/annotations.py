# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyst tools over Intake annotations: ``fetch_annotations`` and ``get_annotation``.

Intake is span-based: end-user / developer feedback, labels, notes, and
metadata are all *annotations* attached to a ``span_id`` (or session-wide, a
``session_id``), not legacy ``entries``. These tools are thin pass-throughs
over ``client.intake.annotations``; the analyst supplies the raw Intake filter.

Annotation kinds:

- ``feedback`` — thumbs sentiment; ``value_text`` is "positive" / "negative".
  This is the strongest signal of a real problem — start here.
- ``label`` — categorical (``value_text``) or scored (``value_numeric``)
  labels, e.g. a 1-5 ``helpfulness`` rating.
- ``note`` / ``metadata`` — free text / structured key-value.

Because annotations hang off spans, each result carries its ``span_id`` /
``session_id`` — feed those into ``fetch_spans`` / ``get_span`` to inspect the
interaction the annotation is about.

Schema source of truth (Intake changes often — design against source, not the
vendored wheel): ``services/intake/src/nhx/intake/spans/api/annotations_schemas.py``
in ``~/code/nemo-helix`` (``AnnotationFilter``).
"""

from typing import Any

from nemo_insights_plugin.analyst.deps import AnalystDeps


async def fetch_annotations(
    deps: AnalystDeps,
    filter: dict[str, Any] | None = None,
    sort: str = "-created_at",
    limit: int = 50,
) -> dict[str, Any]:
    """List span/session annotations (feedback, labels, notes), newest first.

    Returns ``{"annotations": [...], "count": int, "truncated": bool}``;
    ``truncated`` means more matched than ``limit``.

    Args:
        filter: Raw Intake annotation filter pushed to the server. Supported
            keys: ``kind`` ("feedback"/"label"/"note"/"metadata"),
            ``value_text`` (e.g. "negative" for feedback, or a label's text
            value), ``name`` (label name, e.g. "helpfulness"), ``value_numeric``
            (a range object, e.g. ``{"lte": 2}`` for low scores), ``span_id``,
            ``session_id``, ``created_by``, and ``created_at`` (a range).
            To start with negative feedback: ``{"kind": "feedback",
            "value_text": "negative"}``. Omit to list all annotations.
        sort: Sort field; "-created_at" (default, newest first) or "created_at".
        limit: Max annotations to pull across pages (clamped to the ceiling).
    """
    assert deps.backend is not None
    return await deps.backend.list_annotations(
        workspace=deps.workspace,
        filter=filter or None,
        sort=sort,
        limit=min(limit, deps.max_results),
        since=deps.since,
    )


async def get_annotation(deps: AnalystDeps, annotation_id: str) -> dict[str, Any]:
    """Fetch a single annotation by id.

    Args:
        annotation_id: Intake annotation id.
    """
    assert deps.backend is not None
    return await deps.backend.get_annotation(workspace=deps.workspace, annotation_id=annotation_id)
