-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- Config profiles and agent bundles live in Entity Store, so evaluations and
-- benchmark runs keep profile ids as plain text. The orphaned config_profiles
-- and agent_bundles tables go away with the rest of the schema; earlier
-- migrations still alter them on every run.
ALTER TABLE evaluations
    DROP CONSTRAINT IF EXISTS evaluations_framework_profile_id_fkey,
    DROP CONSTRAINT IF EXISTS evaluations_harbor_profile_id_fkey,
    DROP CONSTRAINT IF EXISTS evaluations_switchyard_profile_id_fkey,
    DROP CONSTRAINT IF EXISTS evaluations_intake_profile_id_fkey;

ALTER TABLE benchmark_runs
    DROP CONSTRAINT IF EXISTS benchmark_runs_framework_profile_id_fkey,
    DROP CONSTRAINT IF EXISTS benchmark_runs_harbor_profile_id_fkey,
    DROP CONSTRAINT IF EXISTS benchmark_runs_switchyard_profile_id_fkey,
    DROP CONSTRAINT IF EXISTS benchmark_runs_intake_profile_id_fkey;
