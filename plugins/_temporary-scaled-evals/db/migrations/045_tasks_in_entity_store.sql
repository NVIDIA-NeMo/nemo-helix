-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- Tasks and their revisions live in Entity Store, so other tables keep task ids
-- and revision numbers as plain values. The orphaned tasks and task_revisions
-- tables go away with the rest of the schema; earlier migrations still alter
-- them on every run.
DO $$
DECLARE
    fk record;
BEGIN
    FOR fk IN
        SELECT conrelid::regclass AS table_name, conname
        FROM pg_constraint
        WHERE contype = 'f'
          AND confrelid IN ('tasks'::regclass, 'task_revisions'::regclass)
          AND conrelid <> 'task_revisions'::regclass
    LOOP
        EXECUTE format('ALTER TABLE %s DROP CONSTRAINT %I', fk.table_name, fk.conname);
    END LOOP;
END $$;
