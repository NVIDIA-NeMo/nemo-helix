// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Both `!` are load-bearing: KUI's active rule (0,4,0) outranks plain utilities, and
// `bg-transparent!` clears its fill so this gradient replaces it.
export const ACTIVE_NAV_ITEM_CLASS =
  'bg-transparent! bg-[image:linear-gradient(90deg,color-mix(in_srgb,var(--color-brand)_24%,transparent)_0%,var(--color-translucent-white-000)_50%)]!';

// KUI fixes rows at 36px; size them to their content instead.
export const COMPACT_NAV_ROW_CLASS = 'h-auto min-h-0';

// `px-0` drops KUI's inset so a sub-item's hover spans the same width as a top-level row.
export const SUB_LIST_CLASS = 'px-0';

// Row padding by role. `classnames` cannot resolve Tailwind conflicts — `cn('px-4', 'pr-10')`
// emits both and stylesheet order picks the winner — so each role names one padding set.
export const NAV_ROW_PADDING = {
  /** Carries the leading indent its sub-list gave up (see SUB_LIST_CLASS). */
  subItem: 'pl-6 py-1',
  default: 'py-1',
  /** With the list's own 8px inset, centers the icon in the 48px collapsed rail. */
  rail: 'py-1 px-2',
  /** `pr-10` reserves room for the chevron overlaid on the trailing edge. */
  trailingToggle: 'w-full py-1 pr-10',
  fullWidth: 'w-full py-1',
} as const;
