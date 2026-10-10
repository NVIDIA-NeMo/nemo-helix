-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- Switchyard is no longer managed by scaled-evals. Hide its profiles and
-- client credentials from the API, which no longer accepts either type. Rows are
-- soft-deleted, not dropped, so historical evaluations keep their references.
-- Re-appliable: only live rows are touched.

UPDATE config_profiles
SET deleted_at = NOW(), updated_at = NOW()
WHERE type = 'switchyard' AND deleted_at IS NULL;

UPDATE credentials
SET deleted_at = NOW()
WHERE provider = 'switchyard' AND deleted_at IS NULL;
