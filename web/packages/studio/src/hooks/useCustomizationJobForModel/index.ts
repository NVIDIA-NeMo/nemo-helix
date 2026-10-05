// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useJobsListJobs } from '@nemo/sdk/generated/platform/jobs';
import type {
  ModelEntity,
  HelixJobListSortField,
  HelixJobsListFilter,
} from '@nemo/sdk/generated/platform/schema';
import { JOB_SOURCE } from '@studio/components/dataViews/JobsDataView/constants';

export interface UseCustomizationJobForModelResult {
  jobName: string | undefined;
  isLoading: boolean;
}

/**
 * Resolves the customization job that produced a fine-tuned checkpoint, for linking a
 * model back to its originating job. A ModelEntity has no job reference, so query jobs
 * by `spec.output.name` (set by every customizer backend) — an exact match returning at
 * most one job, newest first when a name was reused across re-runs.
 */
export const useCustomizationJobForModel = (
  workspace: string,
  model: ModelEntity | null | undefined
): UseCustomizationJobForModelResult => {
  const modelName = model?.name;
  const enabled = Boolean(modelName);

  const { data, isLoading } = useJobsListJobs(
    workspace,
    {
      page: 1,
      page_size: 1,
      sort: '-created_at' as HelixJobListSortField,
      filter: {
        source: JOB_SOURCE.CUSTOMIZATION,
        'spec.output.name': modelName,
      } as unknown as HelixJobsListFilter,
    },
    { query: { enabled, staleTime: 5 * 60 * 1000 } }
  );

  return { jobName: data?.data?.[0]?.name, isLoading: enabled && isLoading };
};
