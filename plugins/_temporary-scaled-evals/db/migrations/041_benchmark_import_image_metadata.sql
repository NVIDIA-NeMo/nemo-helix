-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- Repair databases that applied an early revision of migration 039 before
-- benchmark import image metadata was added.
BEGIN;
SET LOCAL lock_timeout = '5s';

ALTER TABLE benchmark_import_tasks
    ADD COLUMN IF NOT EXISTS image_metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'benchmark_import_tasks'::regclass
          AND conname = 'benchmark_import_tasks_image_metadata_check'
    ) THEN
        ALTER TABLE benchmark_import_tasks
            ADD CONSTRAINT benchmark_import_tasks_image_metadata_check
            CHECK (jsonb_typeof(image_metadata) = 'object');
    END IF;
END
$$;

COMMIT;
