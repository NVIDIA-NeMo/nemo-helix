// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { StartOption } from '@studio/components/CreateCustomizationStart/types';
import { Plus } from 'lucide-react';

/**
 * The non-template ways in. "Start from a template" is not among them — templates are
 * picked directly from the group below the divider rather than behind an option.
 */
export const START_OPTIONS: StartOption[] = [
  {
    id: 'scratch',
    title: 'Build from scratch',
    description: 'Open the full form with sensible defaults and choose your own model and data.',
    icon: Plus,
    enabled: true,
  },
];

/** All templates share a task and method today, so they form one group. */
export const TEMPLATE_GROUP_TITLE = 'Text-to-SQL, LoRA';
