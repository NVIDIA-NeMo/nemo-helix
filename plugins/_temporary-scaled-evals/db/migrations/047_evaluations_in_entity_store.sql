-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- Evaluations and benchmark runs live in Entity Store, so benchmark run
-- archives keep run ids as plain values. The orphaned tables go away with the
-- rest of the schema; earlier migrations still alter them on every run.
DO $$
DECLARE
    fk record;
BEGIN
    FOR fk IN
        SELECT conrelid::regclass AS table_name, conname
        FROM pg_constraint
        WHERE contype = 'f'
          AND confrelid IN ('benchmark_runs'::regclass, 'evaluations'::regclass)
          AND conrelid = 'benchmark_run_archives'::regclass
    LOOP
        EXECUTE format('ALTER TABLE %s DROP CONSTRAINT %I', fk.table_name, fk.conname);
    END LOOP;
END $$;
