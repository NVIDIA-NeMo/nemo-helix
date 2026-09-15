// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { StartOption } from '@studio/components/CreateCustomizationStart/types';
import { Bookmark, LayoutGrid, Plus } from 'lucide-react';

export const START_OPTIONS: StartOption[] = [
  {
    id: 'template',
    title: 'Start from a template',
    description:
      'Pick a ready-made NVIDIA recipe. It registers the model and loads the dataset for you, then opens the form filled in.',
    icon: LayoutGrid,
    enabled: true,
  },
  {
    id: 'saved',
    title: 'Use a saved template',
    description:
      'Reuse a configuration you saved earlier. Opens the form filled in with everything it captured, including the dataset.',
    icon: Bookmark,
    enabled: true,
  },
  {
    id: 'scratch',
    title: 'Build from scratch',
    description: 'Open the full form with sensible defaults and choose your own model and data.',
    icon: Plus,
    enabled: true,
  },
];
