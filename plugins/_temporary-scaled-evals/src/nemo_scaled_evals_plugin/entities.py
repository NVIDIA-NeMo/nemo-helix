# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity Store entities for scaled-evals metadata.

Tasks, config profiles and agent bundles live only here. `ScaledEvaluation` is a derived
read model with exactly one writer (the controller's projection phase), which is
what makes it safe on a store that offers per-entity optimistic locking but no
multi-entity transaction and no atomic dequeue.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from nemo_helix_plugin.entity import NemoEntity
from pydantic import BaseModel, Field
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


class ScaledEvaluation(NemoEntity, entity_type="scaled_evals_evaluation"):
    """One `evaluations` row, projected for reads.

    Fields promoted to the top level are exactly the ones the evaluations list
    endpoint filters, searches, or orders by; the rest of `PROJECTED_COLUMNS`
    rides in `detail` so the existing response schemas can be rebuilt without
    restating ~50 column types. `detail` is JSON-coerced, so timestamps are ISO
    strings that Pydantic re-parses on the way out.
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
    search_blob: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


class ScaledEvaluationExecution(NemoEntity, entity_type="scaled_evals_execution"):
    """The immutable launch inputs of one evaluation execution.

    Named after the execution's Platform Job and written once, before that Job
    is created, so the Job launches from here rather than from Postgres. Unlike
    `ScaledEvaluation`, this carries prompt content and the private execution
    snapshot, because launching needs them, so it lives in a workspace only the
    scaled-evals service can read. Credentials appear only as provider to
    platform-secret references; the Job resolves the values at launch.
    """

    evaluation_id: str
    execution_number: int
    inputs: dict[str, Any] = Field(default_factory=dict)


class ScaledEvalsConfigProfile(NemoEntity, entity_type="scaled_evals_config_profile"):
    """A reusable framework or intake config profile, named by its profile id."""

    owner_id: str | None = None
    display_name: str
    type: str
    config: dict[str, Any] = Field(default_factory=dict)


class ScaledEvalsAgentBundle(NemoEntity, entity_type="scaled_evals_agent_bundle"):
    """A registered agent runtime bundle, named by its bundle id."""

    owner_id: str
    bundle_name: str
    agent_name: str
    agent_version: str
    image_ref: str
    image_digest: str
    entrypoint: str
    platform: str = "linux/amd64"
    runtime_abi: str = "glibc"
    bundle_layout_version: int = 1
    builder_profile: str = "node22-npm-v1"
    source_lock_digest: str
    fingerprint: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    visibility: str = "private"
    qualification_status: str = "registered"
    qualification_evidence: dict[str, Any] = Field(default_factory=dict)
    qualified_at: datetime | None = None
    qualified_by: str | None = None


class ScaledEvalsTaskRevision(BaseModel):
    """One immutable task-pack revision and its image build state."""

    revision: int
    status: str = "uploading"
    tarball_object_key: str
    tarball_size_bytes: int | None = None
    tarball_sha256: str | None = None
    image_ref: str | None = None
    image_digest: str | None = None
    verifier_image_ref: str | None = None
    verifier_image_digest: str | None = None
    build_backend: str | None = None
    build_payload: dict[str, Any] = Field(default_factory=dict)
    build_credentials: dict[str, str] = Field(default_factory=dict)
    build_error: str | None = None
    build_attempts: int = 0
    build_claimed_by: str | None = None
    build_claimed_at: datetime | None = None
    build_first_claimed_at: datetime | None = None
    build_next_attempt_at: datetime | None = None
    build_started_at: datetime | None = None
    build_completed_at: datetime | None = None
    created_at: datetime


class ScaledEvalsTask(NemoEntity, entity_type="scaled_evals_task"):
    """A task, named by its task id, with every revision embedded.

    Embedding keeps revision numbering, finalize, and build-state transitions
    to a single versioned write on one entity. Soft-deleted tasks are kept so
    evaluations that ran them still resolve their revision.
    """

    owner_id: str | None = None
    display_name: str
    slug: str
    description: str | None = None
    visibility: str = "private"
    current_revision: int | None = None
    deleted_at: datetime | None = None
    revisions: list[ScaledEvalsTaskRevision] = Field(default_factory=list)


def searchable_blob(row: dict[str, Any]) -> str:
    """Return the lowercased haystack matching the SQL `q` ILIKE predicate."""
    return " ".join(str(row.get(column) or "") for column in SEARCHABLE_COLUMNS).lower()
