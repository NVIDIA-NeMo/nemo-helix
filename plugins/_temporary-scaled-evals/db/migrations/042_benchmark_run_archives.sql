-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- Durable Harbor benchmark exports. Replayed on every plugin startup; preserve
-- existing queue rows and use the configured search_path rather than public.
CREATE TABLE IF NOT EXISTS benchmark_run_archives (
    benchmark_run_id TEXT PRIMARY KEY REFERENCES benchmark_runs(id),
    generation TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'building', 'ready', 'failed')),
    members JSONB NOT NULL,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    claimed_at TIMESTAMPTZ,
    claim_token TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    object_key TEXT,
    size_bytes BIGINT,
    partial BOOLEAN,
    built_at TIMESTAMPTZ,
    error TEXT
);
-- Additive so preview databases that already applied the initial 042 also upgrade.
ALTER TABLE benchmark_run_archives ADD COLUMN IF NOT EXISTS sha256 TEXT
    CHECK (sha256 IS NULL OR sha256 ~ '^[0-9a-f]{64}$');

CREATE INDEX IF NOT EXISTS benchmark_run_archives_queue ON benchmark_run_archives (requested_at)
    WHERE status IN ('queued', 'building');
