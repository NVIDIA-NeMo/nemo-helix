// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ListTracesParams, TraceSortField } from '@nemo/sdk/generated/platform/schema';

/**
 * The traces-list request a detail page was opened from, carried in the URL
 * because the detail route does not share the list's search params. Carrying
 * the resolved request rather than the list's UI state lets the replay reuse
 * the React Query entry the table already filled.
 */
export type TraceListQuery = ListTracesParams & {
  page: number;
  page_size: number;
  sort: TraceSortField;
};

const SORT_FIELDS: readonly string[] = ['started_at', '-started_at'];

export const encodeTraceListQuery = (query: TraceListQuery): string => JSON.stringify(query);

const isPositiveInteger = (value: unknown): value is number =>
  typeof value === 'number' && Number.isInteger(value) && value > 0;

/** `null` for anything a replay cannot trust, such as a hand-edited URL. */
export const parseTraceListQuery = (raw: string | null | undefined): TraceListQuery | null => {
  if (!raw) {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    return null;
  }
  const query = parsed as Record<string, unknown>;
  if (
    !isPositiveInteger(query.page) ||
    !isPositiveInteger(query.page_size) ||
    typeof query.sort !== 'string' ||
    !SORT_FIELDS.includes(query.sort)
  ) {
    return null;
  }
  return query as TraceListQuery;
};
