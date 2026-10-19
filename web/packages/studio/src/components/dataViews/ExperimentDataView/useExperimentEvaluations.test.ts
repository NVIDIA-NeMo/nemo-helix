// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getListEvaluationsQueryKey,
  useListEvaluations,
  usePatchEvaluation,
  usePinEvaluation,
  useUnpinEvaluation,
} from '@nemo/sdk/generated/platform/evaluations';
import {
  type EvaluationRow,
  useExperimentEvaluations,
  type UseExperimentEvaluationsParams,
} from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';
import { renderHook } from '@testing-library/react';

vi.mock('@nemo/common/src/providers/toast/useToast');
vi.mock('@nemo/sdk/generated/platform/evaluations');
// useQueryClient needs a provider; the queries themselves are mocked, so a stub client is enough.
const { invalidateQueries } = vi.hoisted(() => ({ invalidateQueries: vi.fn() }));
vi.mock('@tanstack/react-query', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@tanstack/react-query')>()),
  useQueryClient: () => ({ invalidateQueries }),
}));

const mockUseToast = vi.mocked(useToast);
const mockUseListEvaluations = vi.mocked(useListEvaluations);
const mockUsePinEvaluation = vi.mocked(usePinEvaluation);
const mockUseUnpinEvaluation = vi.mocked(useUnpinEvaluation);
const mockUsePatchEvaluation = vi.mocked(usePatchEvaluation);
const mockGetListEvaluationsQueryKey = vi.mocked(getListEvaluationsQueryKey);

interface Row {
  id: string;
  name: string;
  pinned_at?: string | null;
}

const pin = (name: string): Row => ({ id: name, name, pinned_at: '2026-01-01T00:00:00Z' });
const unp = (name: string): Row => ({ id: name, name, pinned_at: null });

// Minimal stand-in for the SDK query result; the hook only reads `data`, `pagination.total_results`,
// and the loading/fetching/success/error flags.
const queryResult = (rows: Row[], total: number) =>
  ({
    data: { data: rows, pagination: { total_results: total } },
    isLoading: false,
    isFetching: false,
    isSuccess: true,
    error: null,
  }) as unknown as ReturnType<typeof useListEvaluations>;

const mockLists = (
  pinned: { rows: Row[]; total: number },
  unpinned: { rows: Row[]; total: number }
) => {
  mockUseListEvaluations.mockImplementation(((_workspace, params) =>
    (params?.filter as { is_pinned?: boolean } | undefined)?.is_pinned
      ? queryResult(pinned.rows, pinned.total)
      : queryResult(unpinned.rows, unpinned.total)) as typeof useListEvaluations);
};

const baseParams: UseExperimentEvaluationsParams = {
  workspace: 'ws',
  experimentId: 'grp',
  filter: undefined,
  search: '',
  page: 1,
  pageSize: 50,
  sort: '-created_at',
};

describe('useExperimentEvaluations', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseToast.mockReturnValue({ error: vi.fn() } as unknown as ReturnType<typeof useToast>);
    mockUsePinEvaluation.mockReturnValue({
      mutate: vi.fn(),
    } as unknown as ReturnType<typeof usePinEvaluation>);
    mockUseUnpinEvaluation.mockReturnValue({
      mutate: vi.fn(),
    } as unknown as ReturnType<typeof useUnpinEvaluation>);
    mockUsePatchEvaluation.mockReturnValue({
      mutate: vi.fn(),
    } as unknown as ReturnType<typeof usePatchEvaluation>);
  });

  it('paginates over the unpinned set only, so pinned rows do not inflate the page count', () => {
    // 3 pinned + exactly one page (50) of unpinned. Summing the two (the old behavior) gives 53,
    // which would make the table render a phantom 2nd page containing only the pinned rows again.
    mockLists(
      { rows: [pin('p1'), pin('p2'), pin('p3')], total: 3 },
      { rows: Array.from({ length: 50 }, (_unused, i) => unp(`u${i}`)), total: 50 }
    );

    const { result } = renderHook(() => useExperimentEvaluations(baseParams));

    expect(result.current.totalCount).toBe(50);
  });

  it('falls back to the pinned count when nothing is unpinned so a fully-pinned group is not empty', () => {
    mockLists({ rows: [pin('p1'), pin('p2')], total: 2 }, { rows: [], total: 0 });

    const { result } = renderHook(() => useExperimentEvaluations(baseParams));

    expect(result.current.totalCount).toBe(2);
  });

  it('reports a zero count only when both lists are empty', () => {
    mockLists({ rows: [], total: 0 }, { rows: [], total: 0 });

    const { result } = renderHook(() => useExperimentEvaluations(baseParams));

    expect(result.current.totalCount).toBe(0);
  });

  it('lists pinned rows first, then unpinned, dropping an unpinned row already shown as pinned', () => {
    // 'b' appears in both lists (the brief window where the two queries refetch out of step).
    mockLists({ rows: [pin('a'), pin('b')], total: 2 }, { rows: [unp('b'), unp('c')], total: 2 });

    const { result } = renderHook(() => useExperimentEvaluations(baseParams));

    expect(result.current.rows.map((row) => row.name)).toEqual(['a', 'b', 'c']);
  });

  it('fetches the full pinned set in one page and paginates the unpinned set by the caller page size', () => {
    mockLists({ rows: [], total: 0 }, { rows: [], total: 0 });

    renderHook(() => useExperimentEvaluations({ ...baseParams, page: 2, pageSize: 25 }));

    const calls = mockUseListEvaluations.mock.calls;
    const pinnedParams = calls.find(
      (call) => (call[1]?.filter as { is_pinned?: boolean } | undefined)?.is_pinned === true
    )?.[1];
    const unpinnedParams = calls.find(
      (call) => (call[1]?.filter as { is_pinned?: boolean } | undefined)?.is_pinned === false
    )?.[1];

    // Pinned: a single large page (MAX_PINNED_ROWS), pinned-recency order, independent of caller page.
    expect(pinnedParams).toMatchObject({ page: 1, page_size: 100, sort: '-pinned_at' });
    // Unpinned: the caller's page/page_size and sort.
    expect(unpinnedParams).toMatchObject({ page: 2, page_size: 25, sort: '-created_at' });
  });

  it('stays loading until both queries have loaded, not just the faster one', () => {
    // Pinned has returned; unpinned is still on its initial load (no data yet).
    mockUseListEvaluations.mockImplementation(((_workspace, params) =>
      (params?.filter as { is_pinned?: boolean } | undefined)?.is_pinned
        ? queryResult([pin('p')], 1)
        : ({
            data: undefined,
            isLoading: true,
            isFetching: true,
            error: null,
          } as unknown as ReturnType<typeof useListEvaluations>)) as typeof useListEvaluations);

    const { result } = renderHook(() => useExperimentEvaluations(baseParams));

    expect(result.current.isLoading).toBe(true);
  });

  it('clears loading once both queries have responded', () => {
    mockLists({ rows: [pin('p')], total: 1 }, { rows: [unp('u')], total: 1 });

    const { result } = renderHook(() => useExperimentEvaluations(baseParams));

    expect(result.current.isLoading).toBe(false);
  });

  it('reports isSuccess from the unpinned (sortable) query, not the pinned one', () => {
    // Pinned has loaded; the unpinned query (which carries the sort) is still fetching its sort.
    mockUseListEvaluations.mockImplementation(((_workspace, params) =>
      (params?.filter as { is_pinned?: boolean } | undefined)?.is_pinned
        ? queryResult([pin('p')], 1)
        : ({
            data: undefined,
            isLoading: false,
            isFetching: true,
            isSuccess: false,
            error: null,
          } as unknown as ReturnType<typeof useListEvaluations>)) as typeof useListEvaluations);

    const { result } = renderHook(() => useExperimentEvaluations(baseParams));

    expect(result.current.isSuccess).toBe(false);
  });

  it('reports isSuccess once the unpinned query has loaded the current sort', () => {
    mockLists({ rows: [pin('p')], total: 1 }, { rows: [unp('u')], total: 1 });

    const { result } = renderHook(() => useExperimentEvaluations(baseParams));

    expect(result.current.isSuccess).toBe(true);
  });

  it('does not report isSuccess while a new sort is in flight and the previous page is shown as placeholder', () => {
    // The `keepPreviousData` window: status 'success' but isPlaceholderData true. isSuccess must stay
    // false, else sort-error recovery banks the about-to-fail sort and the next 413/503 isn't recovered.
    mockUseListEvaluations.mockImplementation(((_workspace, params) =>
      (params?.filter as { is_pinned?: boolean } | undefined)?.is_pinned
        ? queryResult([pin('p')], 1)
        : ({
            data: { data: [unp('u')], pagination: { total_results: 1 } },
            isLoading: false,
            isFetching: true,
            isSuccess: true,
            isPlaceholderData: true,
            error: null,
          } as unknown as ReturnType<typeof useListEvaluations>)) as typeof useListEvaluations);

    const { result } = renderHook(() => useExperimentEvaluations(baseParams));

    expect(result.current.isSuccess).toBe(false);
  });

  it('scopes pin/unpin invalidation to this group, not the whole workspace', () => {
    mockLists({ rows: [], total: 0 }, { rows: [], total: 0 });

    renderHook(() => useExperimentEvaluations(baseParams));
    // onSuccess is the group-scoped invalidate the hook wires into both mutations.
    const onSuccess = mockUsePinEvaluation.mock.calls[0]?.[0]?.mutation?.onSuccess as
      | (() => void)
      | undefined;
    onSuccess?.();

    expect(mockGetListEvaluationsQueryKey).toHaveBeenCalledWith('ws', {
      filter: { experiment_id: 'grp' },
    });
    expect(invalidateQueries).toHaveBeenCalledTimes(1);
  });

  it('blocks removing an evaluation whose only group is this one, without patching', () => {
    mockLists({ rows: [], total: 0 }, { rows: [], total: 0 });
    const patch = vi.fn();
    mockUsePatchEvaluation.mockReturnValue({
      mutate: patch,
    } as unknown as ReturnType<typeof usePatchEvaluation>);
    const error = vi.fn();
    mockUseToast.mockReturnValue({ error } as unknown as ReturnType<typeof useToast>);

    const { result } = renderHook(() => useExperimentEvaluations(baseParams));
    result.current.removeFromGroup({
      name: 'only-here',
      experiment_ids: ['grp'],
    } as unknown as EvaluationRow);

    expect(patch).not.toHaveBeenCalled();
    expect(error).toHaveBeenCalled();
  });

  it('removes this group from an evaluation that has others, patching the remaining ids', () => {
    mockLists({ rows: [], total: 0 }, { rows: [], total: 0 });
    const patch = vi.fn();
    mockUsePatchEvaluation.mockReturnValue({
      mutate: patch,
    } as unknown as ReturnType<typeof usePatchEvaluation>);

    const { result } = renderHook(() => useExperimentEvaluations(baseParams));
    result.current.removeFromGroup({
      name: 'multi',
      experiment_ids: ['grp', 'other'],
    } as unknown as EvaluationRow);

    expect(patch).toHaveBeenCalledWith({
      workspace: 'ws',
      name: 'multi',
      data: { experiment_ids: ['other'] },
    });
  });
});
