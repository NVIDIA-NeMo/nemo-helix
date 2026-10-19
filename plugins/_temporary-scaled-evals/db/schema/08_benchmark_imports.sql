-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- Durable orchestration records for materialized benchmark imports.
CREATE TABLE benchmark_imports (
    id              TEXT PRIMARY KEY,
    owner_id        TEXT NOT NULL REFERENCES users(id),
    manifest_sha256 TEXT NOT NULL CHECK (manifest_sha256 ~ '^[0-9a-f]{64}$'),
    manifest        JSONB NOT NULL CHECK (jsonb_typeof(manifest) = 'object'),
    visibility      task_visibility NOT NULL DEFAULT 'private',
    description     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (owner_id, manifest_sha256, visibility)
);

CREATE TABLE benchmark_import_tasks (
    import_id       TEXT NOT NULL REFERENCES benchmark_imports(id) ON DELETE CASCADE,
    position        INTEGER NOT NULL CHECK (position >= 0),
    slug            TEXT NOT NULL,
    task_id         TEXT NOT NULL,
    task_revision   INTEGER NOT NULL,
    pack_path       TEXT NOT NULL,
    pack_sha256     TEXT NOT NULL CHECK (pack_sha256 ~ '^[0-9a-f]{64}$'),
    image_ref       TEXT,
    image_digest    TEXT,
    image_metadata  JSONB NOT NULL DEFAULT '{}'::jsonb
                    CHECK (jsonb_typeof(image_metadata) = 'object'),
    PRIMARY KEY (import_id, slug),
    UNIQUE (import_id, position),
    FOREIGN KEY (task_id, task_revision)
        REFERENCES task_revisions(task_id, revision) ON DELETE RESTRICT
);

CREATE TABLE benchmark_import_benchmarks (
    import_id        TEXT NOT NULL REFERENCES benchmark_imports(id) ON DELETE CASCADE,
    position         INTEGER NOT NULL CHECK (position >= 0),
    slug             TEXT NOT NULL,
    name             TEXT NOT NULL,
    task_slugs       JSONB NOT NULL CHECK (jsonb_typeof(task_slugs) = 'array'),
    benchmark_id     TEXT REFERENCES benchmarks(id) ON DELETE RESTRICT,
    benchmark_revision INTEGER,
    PRIMARY KEY (import_id, slug),
    UNIQUE (import_id, position),
    FOREIGN KEY (benchmark_id, benchmark_revision)
        REFERENCES benchmark_revisions(benchmark_id, revision) ON DELETE RESTRICT
);

CREATE INDEX benchmark_imports_owner_created_idx
    ON benchmark_imports(owner_id, created_at DESC);
