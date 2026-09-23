// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

type DateRangeLike = { start?: string; end?: string };

const isDateRangeLike = (value: unknown): value is DateRangeLike =>
  typeof value === 'object' &&
  value !== null &&
  !Array.isArray(value) &&
  Object.keys(value).every((k) => k === 'start' || k === 'end');

/**
 * Converts a plain search state object into a JSON string formatted for the
 * NeMo Helix entities API `search` query parameter.
 *
 * Operator inference by value type:
 *   - string  → { $like: value }
 *   - { start?, end? } → { $gte: start, $lte: end }
 *
 * Returns `undefined` when the resulting object would be empty.
 */
export const buildApiSearchParam = (
  search: Record<string, unknown> | undefined
): string | undefined => {
  if (!search) return undefined;

  const result: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(search)) {
    if (typeof value === 'string') {
      result[key] = { $like: value };
    } else if (isDateRangeLike(value)) {
      const dateFilter: Record<string, string> = {};
      if (value.start) dateFilter['$gte'] = value.start;
      if (value.end) dateFilter['$lte'] = value.end;
      if (Object.keys(dateFilter).length > 0) result[key] = dateFilter;
    }
  }

  return Object.keys(result).length > 0 ? JSON.stringify(result) : undefined;
};

/**
 * Merged two URLSearchParams, so that entries in b override entries in a
 */
export const mergeURLSearchParams = (
  base: URLSearchParams,
  overrides: Record<string, number | string | undefined>
): URLSearchParams => {
  const mergedParams = new URLSearchParams(base);

  for (const [key, value] of Object.entries(overrides)) {
    if (value !== undefined && value !== null && value !== '') {
      mergedParams.set(key, value.toString());
    } else if (mergedParams.has(key)) {
      mergedParams.delete(key);
    }
  }

  return mergedParams;
};

export const convertQueryToList = (query?: Record<string, unknown>): string[] => {
  if (!query) {
    return [];
  }
  return Object.entries(query).map(([key, value]) => {
    const serializedValue =
      typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value ?? '');
    return `${key}=${serializedValue}`;
  });
};
