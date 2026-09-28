/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { z } from 'zod';

// --- Types ---

export type FlagDescriptor = {
  envVar: string;
  schema: z.ZodType;
  typeName: string;
};

export type EnvConfig = Record<string, string | undefined>;

// --- Schema helpers ---

/**
 * Boolean flag. Parses 'true'/'false' strings from env vars.
 * @param envVar - The environment variable name (e.g., 'VITE_FF_MY_FLAG')
 * @param defaultValue - Default value if env var is not set (default: false)
 */
export const booleanFlag = (envVar: string, defaultValue: boolean = false): FlagDescriptor => ({
  envVar,
  typeName: 'boolean',
  schema: z
    .string()
    .optional()
    .transform((val) => (val ? val.toLowerCase() === 'true' : defaultValue)),
});

export const PREVIEW = 'preview' as const;
export type PreviewFlagValue = true | typeof PREVIEW | false;

/**
 * Preview-aware flag. Parses 'true', 'preview', or 'false' strings from env vars.
 *
 * - `'true'`    → `true`    (feature enabled, badge hidden)
 * - `'preview'` → `'preview'` (feature enabled, "Early Preview" badge shown)
 * - `'false'`   → `false`   (feature disabled)
 *
 * Both `true` and `'preview'` are truthy, so existing route gating and
 * conditional rendering (`if (flag)`) work without changes.
 *
 * @param envVar - The environment variable name (e.g., 'VITE_FF_MY_FLAG')
 * @param defaultValue - Default value if env var is not set (default: false)
 */
export const previewFlag = (
  envVar: string,
  defaultValue: PreviewFlagValue = false
): FlagDescriptor => ({
  envVar,
  typeName: 'preview',
  schema: z
    .string()
    .optional()
    .transform((val): PreviewFlagValue => {
      if (!val) return defaultValue;
      const lower = val.toLowerCase();
      if (lower === PREVIEW) return PREVIEW;
      return lower === 'true';
    }),
});

/**
 * String flag. Returns the env var value as-is.
 * @param envVar - The environment variable name
 * @param defaultValue - Default value if env var is not set (omit for required flag)
 */
export const stringFlag = (envVar: string, defaultValue?: string): FlagDescriptor => ({
  envVar,
  typeName: 'string',
  schema:
    defaultValue !== undefined
      ? z
          .string()
          .optional()
          .transform((val) => val ?? defaultValue)
      : z.string(),
});

/**
 * List flag. Splits a comma-separated env var into a trimmed, non-empty string array.
 * @param envVar - The environment variable name
 * @param defaultValue - Default value if env var is not set (default: [])
 */
export const listFlag = (envVar: string, defaultValue: string[] = []): FlagDescriptor => ({
  envVar,
  typeName: 'list',
  schema: z
    .string()
    .optional()
    .transform((val) =>
      val
        ? val
            .split(',')
            .map((item) => item.trim())
            .filter((item) => item.length > 0)
        : defaultValue
    ),
});

/**
 * Number flag. Coerces env var string to number.
 * Uses z.coerce.number() for proper validation (invalid strings fail instead of becoming NaN).
 * @param envVar - The environment variable name
 * @param defaultValue - Default value if env var is not set (omit for required flag)
 */
export const numberFlag = (envVar: string, defaultValue?: number): FlagDescriptor => ({
  envVar,
  typeName: 'number',
  schema:
    defaultValue !== undefined
      ? z.coerce
          .number()
          .optional()
          .transform((val) => val ?? defaultValue)
      : z.coerce.number(),
});

// --- Parser ---

/**
 * Parse feature flags from environment config.
 * @param definitions - Flag definitions object
 * @param env - Environment config (usually import.meta.env)
 * @returns Parsed feature flags object
 * @throws Error if any required flags are missing
 */
export const parseFlags = <T>(definitions: Record<string, FlagDescriptor>, env: EnvConfig): T => {
  const result: Record<string, unknown> = {};
  const errors: string[] = [];

  for (const [key, { envVar, schema, typeName }] of Object.entries(definitions)) {
    const raw = env[envVar];
    const parsed = schema.safeParse(raw);

    if (parsed.success) {
      result[key] = parsed.data;
    } else {
      errors.push(`${key}: ${envVar} (${typeName})`);
    }
  }

  if (errors.length > 0) {
    throw new Error(
      `Missing required feature flags:\n  - ${errors.join('\n  - ')}\n` +
        `Add them to your .env file or provide default values.`
    );
  }

  return result as T;
};
