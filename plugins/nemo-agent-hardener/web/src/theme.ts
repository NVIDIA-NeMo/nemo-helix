// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Colours for the swarm visualisation, as Studio theme custom properties.
 *
 * A plugin cannot use Tailwind palette classes: Studio's Tailwind only scans
 * `web/packages/**`, so a class it does not already emit has no CSS — and the
 * failure is silent, just uncoloured UI. These variables are defined globally
 * by Studio's stylesheet and follow its light/dark theme, so binding them
 * through `style` is both safe and theme-aware.
 *
 * Studio writes the same values as `text-[color:var(--text-color-accent-teal)]`
 * utilities. A plugin must not copy that form — those classes exist only while
 * some file under `web/packages/**` still uses them.
 */
export const ACCENT = {
  blue: 'var(--text-color-accent-blue)',
  gray: 'var(--text-color-accent-gray)',
  green: 'var(--text-color-accent-green)',
  purple: 'var(--text-color-accent-purple)',
  red: 'var(--text-color-accent-red)',
  teal: 'var(--text-color-accent-teal)',
  yellow: 'var(--text-color-accent-yellow)',
} as const;

export const FEEDBACK = {
  danger: 'var(--text-color-feedback-danger)',
  success: 'var(--text-color-feedback-success)',
  warning: 'var(--text-color-feedback-warning)',
} as const;

/** A translucent wash of `token`, for row/panel tints that must stay readable in both themes. */
export const tint = (token: string, percent = 12): string =>
  `color-mix(in srgb, ${token} ${percent}%, transparent)`;
