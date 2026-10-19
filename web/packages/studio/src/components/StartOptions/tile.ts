// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** The design's tile is `radius/md`; Card's own `radius-density-xl` is visibly rounder. */
export const TILE_RADIUS = 'rounded-[var(--radius-md)]';

/** The design's tile sets its title at 14px semibold over a 12px description. */
export const TILE_LABEL_KIND = 'label/semibold/md' as const;
export const TILE_DESCRIPTION_KIND = 'label/regular/sm' as const;

/** The column the whole flow sits in, per the design. */
export const CONTENT_WIDTH = 'w-full max-w-[768px]';
