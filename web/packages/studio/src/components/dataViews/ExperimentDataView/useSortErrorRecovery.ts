// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { TanstackTable } from '@nemo/common/src/components/DataView/internal';
import type { ListEvaluationsQueryError } from '@nemo/sdk/generated/platform/evaluations';
import { useEffect, useRef } from 'react';

/** The table's sorting state — at most one active column for this single-sort table. */
export type SortingState = TanstackTable.SortingState;

/**
 * HTTP statuses a metrics-backed experiment sort can fail with, mapped to a user-facing message.
 * Sorting by a mean metric (cost, latency, evaluator score) is served from the metrics store, which
 * can reject the request in ways an ordinary column sort never would:
 * - 413: the group has more experiments than can be sorted in one request — narrow the filter.
 * - 503: the metrics store is temporarily unavailable.
 * - 400: unsupported sort field — guarded against by the column whitelist, handled defensively.
 */
export const SORT_ERROR_MESSAGES: Readonly<Record<number, string>> = {
  413: 'Too many experiments to sort by this metric. Narrow your filter and try again.',
  503: 'Sorting is temporarily unavailable while metrics load. Please try again shortly.',
  400: 'This column can’t be sorted.',
};

const isSameSort = (a: SortingState, b: SortingState): boolean =>
  a[0]?.id === b[0]?.id && (a[0]?.desc ?? false) === (b[0]?.desc ?? false);

export interface UseSortErrorRecoveryParams {
  error: ListEvaluationsQueryError | null;
  isSuccess: boolean;
  sortingState: SortingState;
  setSorting: (next: SortingState) => void;
  onError: (message: string) => void;
}

/**
 * Recovers from a failed metric sort rather than leaving the table broken. When the current sort
 * triggers a recoverable error (see {@link SORT_ERROR_MESSAGES}), this surfaces a toast and reverts
 * the sort indicator to the last sort that loaded successfully. The displayed rows stay consistent
 * with the indicator because the previous page is kept (`keepPreviousData`) while a sort is in
 * flight or errored, so the revert is seamless.
 *
 * @returns whether the current error is a recoverable sort error, so the caller can suppress its
 *   page-level error UI for this case (the toast + revert handle it instead).
 */
export const useSortErrorRecovery = ({
  error,
  isSuccess,
  sortingState,
  setSorting,
  onError,
}: UseSortErrorRecoveryParams): boolean => {
  const lastGoodSortRef = useRef<SortingState>(sortingState);
  useEffect(() => {
    if (isSuccess) lastGoodSortRef.current = sortingState;
  }, [isSuccess, sortingState]);

  const status = error?.response?.status;
  const message = status != null ? SORT_ERROR_MESSAGES[status] : undefined;
  const isRecoverableSortError =
    message != null && !isSameSort(sortingState, lastGoodSortRef.current);

  const handledErrorRef = useRef<ListEvaluationsQueryError | null>(null);
  useEffect(() => {
    if (!error) {
      handledErrorRef.current = null;
      return;
    }
    if (!isRecoverableSortError || handledErrorRef.current === error || message == null) return;
    handledErrorRef.current = error;
    onError(message);
    setSorting(lastGoodSortRef.current);
  }, [error, isRecoverableSortError, message, onError, setSorting]);

  return isRecoverableSortError;
};
