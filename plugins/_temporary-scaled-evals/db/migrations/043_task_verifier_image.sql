-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- Tasks whose verifier runs in its own sandbox ([verifier] environment_mode =
-- "separate") need a second image, built from tests/. Both columns stay NULL
-- for tasks that verify inside the agent image.
ALTER TABLE task_revisions
    ADD COLUMN IF NOT EXISTS verifier_image_ref TEXT,
    ADD COLUMN IF NOT EXISTS verifier_image_digest TEXT;
