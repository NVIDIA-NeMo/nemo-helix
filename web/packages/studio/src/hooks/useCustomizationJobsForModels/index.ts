// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withOperators } from '@nemo/common/src/api/filterOperators';
import { useJobsListJobs } from '@nemo/sdk/generated/platform/jobs';
import type {
  HelixJobListSortField,
  HelixJobsListFilter,
} from '@nemo/sdk/generated/platform/schema';
import { JOB_SOURCE } from '@studio/components/dataViews/JobsDataView/constants';
import { useMemo } from 'react';

/**
 * Upper bound on names resolved in one request. A table page holds far fewer rows than
 * this; the cap exists so an unexpectedly large page cannot build an unbounded query.
 */
const MAX_NAMES = 200;

/**
 * Re-runs reuse an output name, so a name can match several jobs. Ask for enough rows
 * that the newest job for every name is present even when names are duplicated.
 */
const PAGE_SIZE_MULTIPLIER = 2;

export interface UseCustomizationJobsForModelsResult {
  /** Output name -> name of the job that produced it. Absent when no job matched. */
  jobsByOutputName: Map<string, string>;
  isLoading: boolean;
}

/**
 * Resolves the originating customization job for many outputs at once.
 *
 * The per-row counterpart, `useCustomizationJobForModel`, cannot be used from a table:
 * row actions are built in a plain callback that may not call hooks. This resolves every
 * visible row in one request instead, the same shape as `useModelDeploymentStatuses`.
 *
 * `names` are output names — a model's own name for a full-weight run, the adapter's name
 * for a LoRA run, since that is what the backend records in `spec.output.name`.
 */
export const useCustomizationJobsForModels = (
  workspace: string,
  names: string[]
): UseCustomizationJobsForModelsResult => {
  // Sorted and de-duplicated so an unchanged page keeps one stable query key rather than
  // re-fetching whenever the table happens to reorder.
  const outputNames = useMemo(
    () => [...new Set(names.filter(Boolean))].sort().slice(0, MAX_NAMES),
    [names]
  );

  const enabled = Boolean(workspace) && outputNames.length > 0;

  const { data, isLoading } = useJobsListJobs(
    workspace,
    {
      page: 1,
      page_size: outputNames.length * PAGE_SIZE_MULTIPLIER,
      sort: '-created_at' as HelixJobListSortField,
      filter: withOperators<HelixJobsListFilter>({
        source: JOB_SOURCE.CUSTOMIZATION,
        // Dotted sub-path with `$in`: the plain-array form serializes as repeated params,
        // which the API matches nothing against.
        ...({ 'spec.output.name': { $in: outputNames } } as Record<string, unknown>),
      }),
    },
    { query: { enabled, staleTime: 5 * 60 * 1000 } }
  );

  const jobsByOutputName = useMemo(() => {
    const map = new Map<string, string>();
    // Newest first, so the first job seen for a name is the one to keep.
    for (const job of data?.data ?? []) {
      const spec = job.spec as { output?: { name?: unknown } } | undefined;
      const outputName = spec?.output?.name;
      if (typeof outputName === 'string' && !map.has(outputName)) {
        map.set(outputName, job.name);
      }
    }
    return map;
  }, [data?.data]);

  return { jobsByOutputName, isLoading: enabled && isLoading };
};
