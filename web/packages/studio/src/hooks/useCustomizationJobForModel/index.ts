// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useJobsListJobs } from '@nemo/sdk/generated/platform/jobs';
import type {
  Adapter,
  ModelEntity,
  HelixJobListSortField,
  HelixJobsListFilter,
} from '@nemo/sdk/generated/platform/schema';
import { JOB_SOURCE } from '@studio/components/dataViews/JobsDataView/constants';

export interface UseCustomizationJobForModelResult {
  jobName: string | undefined;
  isLoading: boolean;
}

/** A model entity or an adapter — anything a customization run registers as its output. */
export type CustomizationOutputEntity = Pick<ModelEntity | Adapter, 'name' | 'workspace'>;

/**
 * Resolves the customization job that produced a model or an adapter, for linking it back
 * to its originating run. Neither entity carries a job reference, so query jobs by
 * `spec.output.name` (set by every customizer backend) — an exact match returning at most
 * one job, newest first when a name was reused across re-runs.
 *
 * Pass the entity the UI is actually describing. A LoRA run registers its output as an
 * adapter on the base model rather than as a Model Entity of its own, and sets
 * `spec.output.name` to the **adapter's** name — so an adapter must be resolved from the
 * adapter itself. Resolving it from the base returns the base's own job, which is a
 * confidently wrong answer rather than a missing one.
 *
 * The query runs in the entity's own workspace rather than the one the UI happens to be
 * showing: output names are unique only within a workspace, and an adapter may live in a
 * different workspace from its base (see `toInferenceModelName`). Link to the result with
 * that same workspace.
 */
export const useCustomizationJobForModel = (
  entity: CustomizationOutputEntity | null | undefined
): UseCustomizationJobForModelResult => {
  const outputName = entity?.name;
  const workspace = entity?.workspace ?? '';
  const enabled = Boolean(outputName) && Boolean(workspace);

  const { data, isLoading } = useJobsListJobs(
    workspace,
    {
      page: 1,
      page_size: 1,
      sort: '-created_at' as HelixJobListSortField,
      filter: {
        source: JOB_SOURCE.CUSTOMIZATION,
        'spec.output.name': outputName,
      } as unknown as HelixJobsListFilter,
    },
    { query: { enabled, staleTime: 5 * 60 * 1000 } }
  );

  // Gated on `enabled` so a disabled hook cannot report a job cached under this key from
  // an earlier render.
  return { jobName: enabled ? data?.data?.[0]?.name : undefined, isLoading: enabled && isLoading };
};
