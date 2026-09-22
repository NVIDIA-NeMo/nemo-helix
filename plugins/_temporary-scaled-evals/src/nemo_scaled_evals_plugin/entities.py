# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity Store projections of scaled-evals metadata.

Postgres stays authoritative during this phase. These entities are a derived
read model with exactly one writer (the controller's projection phase), which is
what makes them safe on a store that offers per-entity optimistic locking but no
multi-entity transaction and no atomic dequeue.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from nemo_platform_plugin.entity import NemoEntity
from pydantic import Field
from scaled_evals.api.schemas.evaluations import Evaluation

# Columns the SQL `q` filter matches with ILIKE. Projected into one lowercased
# blob so `$like` matches case-insensitively regardless of the store's own
# collation, which ILIKE would otherwise decide for us.
SEARCHABLE_COLUMNS = ("id", "name", "task_id", "status", "framework", "runtime")

# Exactly the columns the evaluation responses read, and nothing else. Derived
# from the response model so a new field follows automatically: `outcome` is
# computed rather than selected, `reward_value` feeds the typed-reward
# validator, and `result` is the detail-only envelope.
#
# Everything outside this set stays in Postgres on purpose. The detail read
# selects prompt content (`instruction_prefix`, `instruction_postfix`,
# `initial_user_turns`) that no response returns, and copying it into a second
# store would widen where that content lives for no read we serve.
PROJECTED_COLUMNS = (frozenset(Evaluation.model_fields) | {"reward_value", "result"}) - {"outcome"}

# Fixed width, so a lexicographic comparison of the whole sort key is decided by
# the timestamp before it ever reaches the id. `datetime.isoformat()` omits
# microseconds when they are zero, and that variable width inverts the order:
# `...T12:00:00|ev_b` sorts after `...T12:00:00.000001|ev_a`, because the
# separator outranks the `.` that a padded key would have put there.
_SORT_KEY_TIMESTAMP = "%Y-%m-%dT%H:%M:%S.%f"


def evaluation_sort_key(created_at: datetime, evaluation_id: str) -> str:
    """Pack a timestamp and id into one string that sorts by both.

    Comparing two of these lexicographically gives the same answer as SQL's
    `ORDER BY created_at, id`, so a store that sorts a single field can still
    reproduce the list API's order.

    `created_at` is normalized to UTC because Postgres orders TIMESTAMPTZ by the
    instant; a naive value is taken as UTC, not as local time.

    Agreeing with the Postgres order relies on ids being lowercase hex, which is
    what `make_id` mints. Letter case is the one axis the collations tested here
    disagreed on.
    """
    moment = created_at.replace(tzinfo=UTC) if created_at.tzinfo is None else created_at.astimezone(UTC)
    return f"{moment.strftime(_SORT_KEY_TIMESTAMP)}|{evaluation_id}"


class ScaledEvaluation(NemoEntity, entity_type="scaled_evals_evaluation_v2"):
    """One `evaluations` row, projected for reads.

    Fields promoted to the top level are exactly the ones the evaluations list
    endpoint filters, searches, or orders by; the rest of `PROJECTED_COLUMNS`
    rides in `detail` so the existing response schemas can be rebuilt without
    restating ~50 column types. `detail` is JSON-coerced, so timestamps are ISO
    strings that Pydantic re-parses on the way out.

    The entity type carries a version because `sort_key` is required and rows
    projected before it existed cannot be deserialized: the watermark read
    would fail on every controller pass. Reading a new type instead leaves the
    watermark empty, so the controller replays every row and rebuilds the
    projection by itself. That is only acceptable because this is a derived
    read model with one writer -- nothing here is a system of record.
    """

    evaluation_id: str
    owner_id: str | None = None
    task_id: str
    benchmark_run_id: str | None = None
    # `benchmark_run_id IS NULL` as a positive boolean. The default listing hides
    # benchmark members, and filtering a JSON null for equality is not a
    # guarantee the store makes, so the predicate is materialized instead.
    standalone: bool = True
    status: str
    visibility: str
    framework: str
    runtime: str
    # `deleted_at IS NULL` in SQL. Soft-deleted rows stay projected so reads can
    # answer 404 from the projection alone instead of falling back to Postgres.
    deleted: bool = False
    # The evaluation's own timestamps, not the projection's. The list cursor is
    # keyed on `row_created_at`, and the projection watermark on
    # `row_updated_at`; `EntityBase.created_at` only records when we last wrote.
    row_created_at: datetime
    row_updated_at: datetime
    # `(created_at, id)` collapsed into one orderable string. The store sorts a
    # single field, and keyset pagination needs a total order or evaluations
    # sharing a created_at are skipped or repeated across a page boundary --
    # which benchmark fan-out makes the common case, not the tail. Valid only
    # while `created_at` is the list API's one sort option: a second sortable
    # column needs its own key.
    sort_key: str
    search_blob: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


def searchable_blob(row: dict[str, Any]) -> str:
    """Return the lowercased haystack matching the SQL `q` ILIKE predicate."""
    return " ".join(str(row.get(column) or "") for column in SEARCHABLE_COLUMNS).lower()
