// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { TanstackTable } from '@nemo/common/src/components/DataView/internal';

/** The URL search param `useStudioDataViewState` syncs column filters through. */
export const FILTERS_SEARCH_PARAM = 'filters';

/** The `filters` param value for a set of column filters, as the hook writes it. */
export const encodeColumnFiltersParam = (filters: TanstackTable.ColumnFiltersState): string =>
  encodeURIComponent(JSON.stringify(filters));

const isColumnFilter = (entry: unknown): entry is TanstackTable.ColumnFilter =>
  typeof entry === 'object' &&
  entry !== null &&
  typeof (entry as { id?: unknown }).id === 'string' &&
  'value' in entry;

/** Column filters from a `filters` param value; malformed input and malformed entries are dropped. */
export const decodeColumnFiltersParam = (
  value: string | null
): TanstackTable.ColumnFiltersState => {
  if (!value) return [];
  try {
    const parsed: unknown = JSON.parse(decodeURIComponent(value));
    return Array.isArray(parsed) ? parsed.filter(isColumnFilter) : [];
  } catch {
    return [];
  }
};
