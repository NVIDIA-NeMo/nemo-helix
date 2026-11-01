// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CJobTerminalStatuses } from '@nemo/common/src/constants/query';
import type { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';

import * as DataView from '../components/DataView/internal';
import { JOB_POLLING_INTERVAL_MS } from '../constants';

export const getJobRefetchInterval = (status?: HelixJobStatus): number | false => {
  if (!status || !CJobTerminalStatuses.includes(status)) {
    return JOB_POLLING_INTERVAL_MS;
  }
  return false;
};

/** The fields of a TanStack query result that {@link getRequestStatus} reads. */
export interface RequestStatusQuery {
  error?: unknown;
  isLoading?: boolean;
  isPlaceholderData?: boolean;
}

/**
 * A DataView's `requestStatus` for a query. Shows loading only when there's nothing current to
 * show: the first load, or the previous page's placeholder rows while a new page, search or filter
 * loads. Never use `isFetching` here; it is also true during background refetches and polls, which
 * would blank a table that already has data.
 */
export const getRequestStatus = ({
  error,
  isLoading,
  isPlaceholderData,
}: RequestStatusQuery): DataView.QueryStatus | undefined => {
  if (error) return 'error';
  if (isLoading || isPlaceholderData) return 'loading';
  return undefined;
};

// Helper to get sort string from DataView sorting state
export const getSortParam = (sortingState: DataView.TanstackTable.SortingState) => {
  if (sortingState.length === 0) {
    return '-created_at';
  }
  const { id, desc } = sortingState[0];
  const prefix = desc ? '-' : '';
  return `${prefix}${id}`;
};

/**
 * Maps DataView sorting state to a sort query value when the API only allows specific fields.
 * Table columns often use client-only ids (e.g. model_name); URL bookmarking can also reference invalid ids.
 *
 * Pass a generated sort-field enum as `allowedFieldIds` to get that enum back rather than a bare
 * `string`, so the result drops straight into the SDK params. Descending sort prefixes an allowed
 * field with `-`; the enum is expected to carry both directions, as the generated ones do.
 */
export const getSortParamWithWhitelist = <S extends string = string>(
  sortingState: DataView.TanstackTable.SortingState,
  allowedFieldIds: readonly S[],
  fallbackWhenEmptyOrInvalid: S
): S => {
  if (sortingState.length === 0) {
    return fallbackWhenEmptyOrInvalid;
  }
  const { id, desc } = sortingState[0];
  const idStr = String(id);
  if (!allowedFieldIds.includes(idStr as S)) {
    return fallbackWhenEmptyOrInvalid;
  }
  const prefix = desc ? '-' : '';
  return `${prefix}${idStr}` as S;
};
