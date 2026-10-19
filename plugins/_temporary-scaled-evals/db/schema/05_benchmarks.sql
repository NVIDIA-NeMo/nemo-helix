-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- Benchmarks: a named, revisioned COLLECTION of tasks. A benchmark revision is
-- an immutable snapshot of member tasks; each member optionally pins a task
-- revision (NULL = resolve to the task's latest at eval time). See docs/API.md
-- § Benchmarks.
--
-- This is the inverse of the task primitive (01_tasks.sql): a task is one
-- buildable unit, a benchmark groups many tasks. Mirrors the task identity +
-- integer-revision model (`(id, revision)` PK, current_revision = max). Loads
-- after 01_tasks.sql so the membership FKs to tasks/task_revisions resolve.

CREATE TYPE benchmark_visibility AS ENUM ('private', 'team', 'org', 'public');
CREATE TYPE benchmark_qualification AS ENUM ('registered', 'qualified', 'rejected');

CREATE TABLE benchmarks (
    id               TEXT PRIMARY KEY,
    owner_id         TEXT REFERENCES users(id),
    name             TEXT NOT NULL,
    slug             TEXT NOT NULL,
    description      TEXT,
    visibility       benchmark_visibility NOT NULL DEFAULT 'private',
    qualification_status benchmark_qualification NOT NULL DEFAULT 'registered',
    qualification_evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    qualified_at     TIMESTAMPTZ,
    qualified_by     TEXT,
    current_revision INTEGER,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at       TIMESTAMPTZ
    -- TBD: owner_user_id / team_id — pending auth integration, mirroring the
    --      tasks TBDs (01_tasks.sql).
);

-- Per-owner slug uniqueness on live rows (owner column lands with auth; global
-- for now, same as tasks). Benchmarks and tasks have separate slug namespaces.
CREATE UNIQUE INDEX benchmarks_slug_live_uq
    ON benchmarks (slug)
    WHERE deleted_at IS NULL;

CREATE INDEX benchmarks_created_at_idx
    ON benchmarks (created_at DESC);

-- Immutable per-revision snapshot. The member set lives in
-- benchmark_revision_tasks and is fixed at revision-creation time; changing the
-- set means a new revision (reproducible by construction).
CREATE TABLE benchmark_revisions (
    benchmark_id TEXT NOT NULL
        REFERENCES benchmarks(id) ON DELETE RESTRICT,
    revision     INTEGER NOT NULL,
    description  TEXT,
    -- Metadata-only variant lineage. NULL for ordinary revisions.
    derived_from_benchmark_id TEXT,
    derived_from_revision     INTEGER,
    -- Allowlisted operational overrides (e.g. {"agent_timeout_floor_sec": 7200}).
    operational_policy JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (benchmark_id, revision),
    CONSTRAINT benchmark_revisions_derived_pair_ck CHECK (
        (derived_from_benchmark_id IS NULL AND derived_from_revision IS NULL)
        OR (derived_from_benchmark_id IS NOT NULL AND derived_from_revision IS NOT NULL)
    ),
    CONSTRAINT benchmark_revisions_derived_from_fk
        FOREIGN KEY (derived_from_benchmark_id, derived_from_revision)
        REFERENCES benchmark_revisions (benchmark_id, revision)
        ON DELETE RESTRICT
);

-- Membership: which tasks (optionally pinned to a task revision) belong to a
-- given benchmark revision.
CREATE TABLE benchmark_revision_tasks (
    benchmark_id  TEXT NOT NULL,
    revision      INTEGER NOT NULL,
    task_id       TEXT NOT NULL
        REFERENCES tasks(id) ON DELETE RESTRICT,
    -- NULL = resolve to the task's current/latest revision at eval time. When
    -- pinned (non-NULL) the composite FK below enforces the (task_id, revision)
    -- pair exists; MATCH SIMPLE skips the check for NULL pins.
    task_revision INTEGER,
    -- Stable ordering of tasks within the benchmark revision.
    position      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (benchmark_id, revision, task_id),
    FOREIGN KEY (benchmark_id, revision)
        REFERENCES benchmark_revisions (benchmark_id, revision) ON DELETE CASCADE,
    FOREIGN KEY (task_id, task_revision)
        REFERENCES task_revisions (task_id, revision) ON DELETE RESTRICT
);

CREATE INDEX benchmark_revision_tasks_task_idx
    ON benchmark_revision_tasks (task_id);
