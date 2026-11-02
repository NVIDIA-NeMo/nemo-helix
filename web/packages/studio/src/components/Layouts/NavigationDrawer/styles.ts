// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// KUI fixes rows at 36px; size them to their content instead.
export const COMPACT_NAV_ROW_CLASS = 'h-auto min-h-0';

// Row padding by role. `classnames` cannot resolve Tailwind conflicts — `cn('px-4', 'pr-10')`
// emits both and stylesheet order picks the winner — so each role names one padding set.
export const NAV_ROW_PADDING = {
  default: 'py-1',
  /** `pr-10` reserves room for the chevron overlaid on the trailing edge. */
  trailingToggle: 'w-full py-1 pr-10',
  fullWidth: 'w-full py-1',
} as const;
