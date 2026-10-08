// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Parsers shared by the manifest forms, which collect lists and maps as single comma-separated fields. */

/** Split a comma-separated field into a trimmed, non-empty list. */
export const splitList = (value: string | undefined): string[] =>
  (value ?? '')
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean);

/** Parse `KEY=VALUE` pairs; only the first `=` splits, so values may contain `=`. */
export const parseEnvPairs = (value: string | undefined): Record<string, string> =>
  Object.fromEntries(
    splitList(value)
      .map((entry) => {
        const at = entry.indexOf('=');
        return at > 0 ? [entry.slice(0, at).trim(), entry.slice(at + 1).trim()] : null;
      })
      .filter((pair): pair is [string, string] => pair !== null)
  );
