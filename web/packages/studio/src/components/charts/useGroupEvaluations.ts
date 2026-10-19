// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useListEvaluations } from '@nemo/sdk/generated/platform/evaluations';
import type { EvaluationFilter } from '@nemo/sdk/generated/platform/schema';
import type { EvaluationRow } from '@studio/components/dataViews/ExperimentDataView/useExperimentEvaluations';
import { DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import { useMemo } from 'react';

export interface GroupEvaluations {
  rows: EvaluationRow[];
  /** Evaluations in the group, which exceeds `rows.length` when the group is larger than one page. */
  total: number;
  isLoading: boolean;
  isError: boolean;
}

/** Loads every evaluation in a group in one request for the group's charts, reusing the list endpoint
 * (each evaluation already carries the rollup means the charts plot). */
export function useGroupEvaluations(
  workspace: string,
  experimentId: string,
  options?: { enabled?: boolean }
): GroupEvaluations {
  // Disabled by callers that already hold the whole group, to avoid a redundant all-evaluations fetch.
  const enabled = (options?.enabled ?? true) && !!experimentId;
  // Newest first, explicitly rather than by the API's default, so a group too large for one page
  // truncates to a known slice — the most recent evaluations — instead of an arbitrary one.
  const { data, isLoading, isError } = useListEvaluations(
    workspace,
    {
      page: 1,
      page_size: DEFAULT_LARGE_PAGE_SIZE,
      sort: '-created_at',
      filter: { experiment_id: experimentId } as EvaluationFilter,
    },
    { query: { enabled } }
  );

  const rows = useMemo<EvaluationRow[]>(
    () =>
      (data?.data ?? []).map((evaluation) => ({
        ...evaluation,
        id: evaluation.id ?? evaluation.name ?? '',
      })),
    [data]
  );

  return { rows, total: data?.pagination?.total_results ?? rows.length, isLoading, isError };
}
