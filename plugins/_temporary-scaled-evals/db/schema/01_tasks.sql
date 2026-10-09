-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- Tasks: mutable identity/metadata + immutable per-revision build record.
-- See docs/API.md sections "Tasks", "Tenancy", "Identifiers".

CREATE TYPE task_visibility AS ENUM ('private', 'team', 'org', 'public');

CREATE TYPE task_revision_status AS ENUM (
    'pending',
    'uploading',
    'building',
    'ready',
    'failed'
);

CREATE TABLE tasks (
    id               TEXT PRIMARY KEY,
    owner_id         TEXT REFERENCES users(id),
    name             TEXT NOT NULL,
    slug             TEXT NOT NULL,
    description      TEXT,
    visibility       task_visibility NOT NULL DEFAULT 'private',
    current_revision INTEGER,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at       TIMESTAMPTZ
    -- TBD: owner_user_id — the identity provider's subject maps to an internal
    --      usr_... id, but users table shape, FK type, and
    --      JIT-vs-preprovisioned strategy are unspecified. Add when platform
    --      identity is bridged into the ownership model.
    -- TBD: team_id — referenced in API.md:28, :95, :175-176 but no POST /teams
    --      and docs/internals/ARCHITECTURE.md:157 lists per-team budget as open. Add when
    --      teams ownership is decided.
);

-- Per-owner slug uniqueness (API.md:213). Owner column does not exist yet,
-- so enforce global uniqueness on live rows in the meantime — narrower than
-- the eventual constraint, prevents seeding duplicates we'd later have to
-- migrate out. Replace with (owner_user_id, slug) when owner lands.
CREATE UNIQUE INDEX tasks_slug_live_uq
    ON tasks (slug)
    WHERE deleted_at IS NULL;

CREATE INDEX tasks_created_at_idx
    ON tasks (created_at DESC);

CREATE TABLE task_revisions (
    task_id       TEXT NOT NULL
        REFERENCES tasks(id) ON DELETE RESTRICT,
    revision           INTEGER NOT NULL,
    status             task_revision_status NOT NULL DEFAULT 'pending',
    tarball_object_key TEXT NOT NULL,
    tarball_size_bytes BIGINT,
    tarball_sha256     TEXT,
    image_ref          TEXT,
    image_digest       TEXT,
    verifier_image_ref    TEXT,
    verifier_image_digest TEXT,
    task_yaml     JSONB,
    tasks              JSONB,
    build_started_at   TIMESTAMPTZ,
    build_completed_at TIMESTAMPTZ,
    build_error        TEXT,
    build_backend      TEXT,
    build_payload      JSONB NOT NULL DEFAULT '{}'::jsonb,
    build_credentials  JSONB NOT NULL DEFAULT '{}'::jsonb,
    build_claimed_at   TIMESTAMPTZ,
    build_first_claimed_at TIMESTAMPTZ,
    build_claimed_by   TEXT,
    build_attempts     INTEGER NOT NULL DEFAULT 0,
    build_next_attempt_at TIMESTAMPTZ,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- TBD: created_by_user_id — pending auth integration (see tasks TBDs).
    PRIMARY KEY (task_id, revision)
);

CREATE INDEX task_revisions_active_idx
    ON task_revisions (status)
    WHERE status IN ('pending', 'uploading', 'building');

CREATE INDEX task_revisions_build_queue_idx
    ON task_revisions (build_next_attempt_at, build_started_at)
    WHERE status = 'building' AND build_backend IS NOT NULL;

CREATE TABLE service_heartbeats (
    service      TEXT NOT NULL,
    instance_id  TEXT NOT NULL,
    heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (service, instance_id)
);

CREATE INDEX service_heartbeats_freshness_idx
    ON service_heartbeats (service, heartbeat_at DESC);
