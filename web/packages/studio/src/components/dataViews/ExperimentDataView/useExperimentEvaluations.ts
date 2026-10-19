// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  getListEvaluationsQueryKey,
  type ListEvaluationsQueryError,
  useListEvaluations,
  usePatchEvaluation,
  usePinEvaluation,
  useUnpinEvaluation,
} from '@nemo/sdk/generated/platform/evaluations';
import type {
  EvaluationFilter,
  EvaluationResponse,
  ListEvaluationsParams,
} from '@nemo/sdk/generated/platform/schema';
import { keepPreviousData, useQueryClient } from '@tanstack/react-query';
import { useCallback, useMemo, useRef } from 'react';

/** An API response plus the stable `id` the data view needs. */
export type EvaluationRow = EvaluationResponse & { id: string };
export type ListEvaluationsSortParam = NonNullable<ListEvaluationsParams['sort']>;

const toRows = (experiments: EvaluationResponse[] | undefined): EvaluationRow[] =>
  (experiments ?? []).map((experiment) => ({
    ...experiment,
    id: experiment.id ?? experiment.name ?? '',
  }));

/**
 * Pinned rows are shown in full atop every page rather than paginated, so the whole pin set is
 * fetched in one request. A group's pin set is a small curated list; this bound keeps that request
 * well under the API's max page size (1000) while covering any realistic number of pins.
 */
const MAX_PINNED_ROWS = 100;

export interface UseExperimentEvaluationsParams {
  workspace: string;
  experimentId: string;
  filter: Partial<EvaluationFilter> | undefined;
  search: string;
  page: number;
  pageSize: number;
  /** Omit (undefined) when no column sort is active; the API then defaults to -created_at. */
  sort?: ListEvaluationsSortParam;
}

export interface ExperimentEvaluations {
  /** Pinned rows first (newest-pinned first), then the current page of unpinned rows. */
  rows: EvaluationRow[];
  /** Pins the row if unpinned, unpins it otherwise, then refetches both lists. */
  togglePin: (row: EvaluationRow) => void;
  /** Removes the current group from the evaluation's `experiment_ids`, then refetches. Blocks (with a
   * toast) when this is the evaluation's only group — every evaluation must belong to at least one. */
  removeFromGroup: (row: EvaluationRow) => void;
  /**
   * Row count that drives pagination: the unpinned total. Pinned rows ride atop every page and are
   * not paginated, so they're excluded; falls back to the pinned count when nothing is unpinned so
   * a fully-pinned group still renders one page instead of reading as empty.
   */
  totalCount: number;
  error: ListEvaluationsQueryError | null;
  /** True until both queries have loaded once; ignores background refetches (page changes, pins). */
  isLoading: boolean;
  /** True when either query is fetching */
  isFetching: boolean;
  /**
   * True once the sortable (unpinned) page has loaded successfully for the current sort. Tracks the
   * current sort key specifically — it stays false while a new sort is in flight or has errored — so
   * callers can record the last good sort for sort-error recovery.
   */
  isSuccess: boolean;
}

/**
 * Loads one experiment group's experiments as two queries — the full pinned set (`is_pinned=true`,
 * sorted `-pinned_at`) and the current page of unpinned (`is_pinned=false`) — and concatenates them
 * so pinned rows top every page. Pagination covers only the unpinned set; the pinned set is fetched
 * once and repeated atop every page rather than paginated. A pin/unpin persists through the API,
 * then invalidates both lists so the new state is refetched (no optimistic update).
 */
export function useExperimentEvaluations({
  workspace,
  experimentId,
  filter,
  search,
  page,
  pageSize,
  sort,
}: UseExperimentEvaluationsParams): ExperimentEvaluations {
  const queryClient = useQueryClient();
  const toast = useToast();

  const baseFilter = {
    ...filter,
    ...(search && { name: { $like: search } }),
    // Spread last so the group scope can't be overridden by a user filter.
    experiment_id: experimentId,
  };
  const listQueryOptions = {
    query: {
      placeholderData: keepPreviousData,
      enabled: !!experimentId,
    },
  };

  const {
    data: pinnedResponse,
    isLoading: isPinnedLoading,
    isFetching: isPinnedFetching,
    error: pinnedError,
  } = useListEvaluations(
    workspace,
    {
      page: 1,
      page_size: MAX_PINNED_ROWS,
      sort: '-pinned_at',
      filter: { ...baseFilter, is_pinned: true } as EvaluationFilter,
    },
    listQueryOptions
  );

  const {
    data: unpinnedResponse,
    isLoading: isUnpinnedLoading,
    isFetching: isUnpinnedFetching,
    isSuccess: isUnpinnedSuccess,
    isPlaceholderData: isUnpinnedPlaceholder,
    error: unpinnedError,
  } = useListEvaluations(
    workspace,
    {
      page,
      page_size: pageSize,
      sort,
      filter: { ...baseFilter, is_pinned: false } as EvaluationFilter,
    },
    listQueryOptions
  );

  const pinned = useMemo(() => toRows(pinnedResponse?.data), [pinnedResponse]);
  const unpinned = useMemo(() => toRows(unpinnedResponse?.data), [unpinnedResponse]);

  // Synchronous guard so a row can't fire a second pin/unpin while one is already in flight; the
  // name is cleared once the mutation settles.
  const pendingRef = useRef<Set<string>>(new Set());

  // Scope invalidation to this group's experiment lists (any page/sort/filter, pinned and unpinned)
  // via a partial key match on experiment_id, so a pin/unpin doesn't refetch other groups'
  // lists. Returned (not voided) so the mutation's onSuccess awaits the refetch before onSettled
  // re-enables the row.
  const invalidateList = useCallback(
    () =>
      queryClient.invalidateQueries({
        queryKey: getListEvaluationsQueryKey(workspace, {
          filter: { experiment_id: experimentId },
        }),
      }),
    [queryClient, workspace, experimentId]
  );

  const { mutate: pinEvaluation } = usePinEvaluation({
    mutation: {
      onSuccess: invalidateList,
      onError: () => toast.error('Failed to pin evaluation.'),
      onSettled: (_data, _error, { name }) => {
        pendingRef.current.delete(name);
      },
    },
  });
  const { mutate: unpinEvaluation } = useUnpinEvaluation({
    mutation: {
      onSuccess: invalidateList,
      onError: () => toast.error('Failed to unpin evaluation.'),
      onSettled: (_data, _error, { name }) => {
        pendingRef.current.delete(name);
      },
    },
  });

  const togglePin = useCallback(
    (row: EvaluationRow) => {
      const { name } = row;
      if (pendingRef.current.has(name)) return;
      pendingRef.current.add(name);
      if (row.pinned_at != null) unpinEvaluation({ workspace, name });
      else pinEvaluation({ workspace, name });
    },
    [workspace, pinEvaluation, unpinEvaluation]
  );

  const { mutate: patchEvaluation } = usePatchEvaluation({
    mutation: {
      onSuccess: () => {
        toast.success('Removed the evaluation from this experiment.');
        return invalidateList();
      },
      onError: () => toast.error('Failed to remove the evaluation from this experiment.'),
      onSettled: (_data, _error, { name }) => {
        pendingRef.current.delete(name);
      },
    },
  });

  const removeFromGroup = useCallback(
    (row: EvaluationRow) => {
      const { name } = row;
      if (pendingRef.current.has(name)) return;
      // The API rejects an empty `experiment_ids`, so block removing an evaluation's only group and
      // explain why rather than firing a request that would 400.
      if (row.experiment_ids.length <= 1) {
        toast.error('This evaluation must belong to at least one experiment.');
        return;
      }
      pendingRef.current.add(name);
      patchEvaluation({
        workspace,
        name,
        data: { experiment_ids: row.experiment_ids.filter((id) => id !== experimentId) },
      });
    },
    [workspace, experimentId, patchEvaluation, toast]
  );

  // Pinned first, then the current page of unpinned. Drop any unpinned row already shown as pinned —
  // it can appear in both server lists during the brief window where the two queries refetch out of step.
  const rows = useMemo<EvaluationRow[]>(() => {
    const pinnedNames = new Set(pinned.map((row) => row.name));
    return [...pinned, ...unpinned.filter((row) => !pinnedNames.has(row.name))];
  }, [pinned, unpinned]);

  // Pagination covers the unpinned set only — pinned rows are a fixed header repeated atop every
  // page, so counting them would inflate the page count and add trailing pages that show nothing
  // but the (already-visible) pinned rows. Fall back to the pinned count when nothing is unpinned so
  // a fully-pinned group still renders a single page instead of reading as empty.
  const unpinnedTotal = unpinnedResponse?.pagination?.total_results ?? unpinned.length;
  const totalCount = unpinnedTotal > 0 ? unpinnedTotal : pinned.length;
  const error = pinnedError ?? unpinnedError;
  // react-query's per-query isLoading is true only on first load (keepPreviousData keeps data across
  // refetches, so isPending stays false). OR-ing the two keeps the table in its loading state until
  // both lists have loaded once, rather than clearing as soon as the faster query returns.
  const isLoading = isPinnedLoading || isUnpinnedLoading;
  const isFetching = isPinnedFetching || isUnpinnedFetching;
  // Sort-error recovery keys off the unpinned (sort-carrying) query. Exclude placeholder data so
  // `keepPreviousData`'s in-flight 'success' doesn't bank an about-to-fail sort as the last good one.
  const isSuccess = isUnpinnedSuccess && !isUnpinnedPlaceholder;

  return { rows, togglePin, removeFromGroup, totalCount, error, isLoading, isFetching, isSuccess };
}
