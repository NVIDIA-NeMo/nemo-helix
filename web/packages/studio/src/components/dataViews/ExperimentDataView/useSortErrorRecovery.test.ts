// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ListEvaluationsQueryError } from '@nemo/sdk/generated/platform/evaluations';
import {
  SORT_ERROR_MESSAGES,
  type SortingState,
  useSortErrorRecovery,
  type UseSortErrorRecoveryParams,
} from '@studio/components/dataViews/ExperimentDataView/useSortErrorRecovery';
import { renderHook } from '@testing-library/react';

const sort = (id: string, desc = false): SortingState => [{ id, desc }];
// Minimal Axios-shaped error: the hook only reads `response.status`.
const httpError = (status: number): ListEvaluationsQueryError =>
  ({ response: { status } }) as unknown as ListEvaluationsQueryError;

const CREATED_AT = sort('created_at', true);
const LATENCY = sort('latency_ms');
const COST = sort('cost_usd');

const setup = (overrides: Partial<UseSortErrorRecoveryParams> = {}) => {
  const setSorting = vi.fn();
  const onError = vi.fn();
  const initialProps: UseSortErrorRecoveryParams = {
    error: null,
    isSuccess: true,
    sortingState: CREATED_AT,
    setSorting,
    onError,
    ...overrides,
  };
  const { result, rerender } = renderHook(
    (props: UseSortErrorRecoveryParams) => useSortErrorRecovery(props),
    { initialProps }
  );
  return { result, rerender, setSorting, onError, initialProps };
};

describe('useSortErrorRecovery', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it.each([413, 503, 400] as const)(
    'toasts and reverts the sort when a newly applied sort fails with %i',
    (status) => {
      const { result, rerender, setSorting, onError, initialProps } = setup();

      // User applies a metric sort; it is in flight (not yet successful).
      rerender({ ...initialProps, isSuccess: false, sortingState: LATENCY });
      // The metric sort request errors.
      rerender({
        ...initialProps,
        error: httpError(status),
        isSuccess: false,
        sortingState: LATENCY,
      });

      expect(result.current).toBe(true);
      expect(onError).toHaveBeenCalledTimes(1);
      expect(onError).toHaveBeenCalledWith(SORT_ERROR_MESSAGES[status]);
      // Reverts the indicator to the last sort that loaded successfully.
      expect(setSorting).toHaveBeenCalledWith(CREATED_AT);
    }
  );

  it('does not recover from a non-sort error status (e.g. 500)', () => {
    const { result, rerender, setSorting, onError, initialProps } = setup();

    rerender({ ...initialProps, isSuccess: false, sortingState: LATENCY });
    rerender({ ...initialProps, error: httpError(500), isSuccess: false, sortingState: LATENCY });

    expect(result.current).toBe(false);
    expect(onError).not.toHaveBeenCalled();
    expect(setSorting).not.toHaveBeenCalled();
  });

  it('does not recover when the error is on the already-good sort (e.g. initial load), to avoid a loop', () => {
    // Never reached a success: last good sort is the initial sort, and the error is on that same sort.
    const { result, setSorting, onError, rerender, initialProps } = setup({ isSuccess: false });

    rerender({
      ...initialProps,
      error: httpError(503),
      isSuccess: false,
      sortingState: CREATED_AT,
    });

    expect(result.current).toBe(false);
    expect(onError).not.toHaveBeenCalled();
    expect(setSorting).not.toHaveBeenCalled();
  });

  it('reverts to the most recent good sort, not the original', () => {
    const { rerender, setSorting, initialProps } = setup();

    // A second sort loads successfully, becoming the new last-good sort...
    rerender({ ...initialProps, isSuccess: true, sortingState: LATENCY });
    // ...then a third sort is applied and fails.
    rerender({ ...initialProps, isSuccess: false, sortingState: COST });
    rerender({ ...initialProps, error: httpError(413), isSuccess: false, sortingState: COST });

    expect(setSorting).toHaveBeenCalledWith(LATENCY);
  });

  it('handles each error episode once even if other inputs change', () => {
    const { rerender, onError, initialProps } = setup();
    const error = httpError(413);

    rerender({ ...initialProps, isSuccess: false, sortingState: LATENCY });
    rerender({ ...initialProps, error, isSuccess: false, sortingState: LATENCY });
    // Same error object, still on the failed sort, but a new onError reference re-runs the effect.
    const newOnError = vi.fn();
    rerender({
      ...initialProps,
      error,
      isSuccess: false,
      sortingState: LATENCY,
      onError: newOnError,
    });

    expect(onError).toHaveBeenCalledTimes(1);
    expect(newOnError).not.toHaveBeenCalled();
  });

  it('handles a fresh error after the previous one cleared', () => {
    const { rerender, onError, setSorting, initialProps } = setup();

    // First failed sort → handled.
    rerender({ ...initialProps, isSuccess: false, sortingState: LATENCY });
    rerender({ ...initialProps, error: httpError(413), isSuccess: false, sortingState: LATENCY });
    // Revert applied + refetch on the good sort succeeds (error clears).
    rerender({ ...initialProps, error: null, isSuccess: true, sortingState: CREATED_AT });
    // A new sort fails again → handled again.
    rerender({ ...initialProps, isSuccess: false, sortingState: COST });
    rerender({ ...initialProps, error: httpError(503), isSuccess: false, sortingState: COST });

    expect(onError).toHaveBeenCalledTimes(2);
    expect(onError).toHaveBeenNthCalledWith(1, SORT_ERROR_MESSAGES[413]);
    expect(onError).toHaveBeenNthCalledWith(2, SORT_ERROR_MESSAGES[503]);
    expect(setSorting).toHaveBeenLastCalledWith(CREATED_AT);
  });
});
