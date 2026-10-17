-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- Benchmarks, their revisions and benchmark imports live in Entity Store, so
-- other tables keep benchmark ids and revision numbers as plain values. The
-- orphaned benchmark and import tables go away with the rest of the schema;
-- earlier migrations still alter them on every run.
DO $$
DECLARE
    fk record;
BEGIN
    FOR fk IN
        SELECT conrelid::regclass AS table_name, conname
        FROM pg_constraint
        WHERE contype = 'f'
          AND confrelid IN ('benchmarks'::regclass, 'benchmark_revisions'::regclass, 'benchmark_imports'::regclass)
          AND conrelid NOT IN (
              'benchmarks'::regclass,
              'benchmark_revisions'::regclass,
              'benchmark_revision_tasks'::regclass,
              'benchmark_imports'::regclass,
              'benchmark_import_tasks'::regclass,
              'benchmark_import_benchmarks'::regclass
          )
    LOOP
        EXECUTE format('ALTER TABLE %s DROP CONSTRAINT %I', fk.table_name, fk.conname);
    END LOOP;
END $$;
