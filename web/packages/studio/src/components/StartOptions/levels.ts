// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { StartOptionTag } from '@studio/components/StartOptions/types';

/**
 * How much an entry point asks of you. The three read as a ladder, so they only mean
 * anything relative to each other — and a rung has to mean the same thing on every create
 * flow, which is why they live here rather than beside one flow's options.
 *
 * Not every flow offers all three: a flow without a guided path has no Beginner rung, and
 * that is not a reason to relabel the rungs it does have.
 */
export const BEGINNER: StartOptionTag = { label: 'Beginner', color: 'gray', kind: 'solid' };
export const INTERMEDIATE: StartOptionTag = {
  label: 'Intermediate',
  color: 'gray',
  kind: 'solid',
};
export const ADVANCED: StartOptionTag = { label: 'Advanced', color: 'gray', kind: 'solid' };
