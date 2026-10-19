-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- Visibility controls discovery; qualification remains independent advisory metadata.
ALTER TABLE benchmarks
    DROP CONSTRAINT IF EXISTS benchmarks_public_requires_qualification_ck;
