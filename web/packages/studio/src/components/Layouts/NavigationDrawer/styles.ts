// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Both `!` are load-bearing: KUI's active rule (0,4,0) outranks plain utilities, and
// `bg-transparent!` clears its fill so this gradient replaces it.
export const ACTIVE_NAV_ITEM_CLASS =
  'bg-transparent! bg-[image:linear-gradient(90deg,color-mix(in_srgb,var(--color-brand)_24%,transparent)_0%,var(--color-translucent-white-000)_50%)]!';

// `px-0` drops KUI's inset so a sub-item's hover spans the same width as a top-level row.
export const SUB_LIST_CLASS = 'px-0';

// Under `asChild` KUI still wraps the label in its own span, so the type token targets that span
// rather than the row — which also outranks the (0,4,0) rule that bolds an active sub-item's label.
export const NAV_LABEL_CLASS = '[&_.nv-vertical-nav-item-label]:text-label-semibold-md';

// Row padding by role. `classnames` cannot resolve Tailwind conflicts — `cn('px-4', 'pr-10')`
// emits both and stylesheet order picks the winner — so each role names one padding set.
export const NAV_ROW_PADDING = {
  /** Carries the leading indent its sub-list gave up (see SUB_LIST_CLASS). */
  subItem: 'pl-6 py-1 h-auto',
  default: 'py-1 px-4',
  /** `pr-10` reserves room for the chevron overlaid on the trailing edge. */
  trailingToggle: 'w-full py-1 pl-4 pr-10',
  fullWidth: 'w-full py-1 px-4',
} as const;
