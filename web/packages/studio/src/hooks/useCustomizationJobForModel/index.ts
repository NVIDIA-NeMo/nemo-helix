// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getJobsListJobsQueryOptions, useJobsListJobs } from '@nemo/sdk/generated/platform/jobs';
import type {
  ModelEntity,
  HelixJobListSortField,
  HelixJobsListFilter,
  JobsListJobsParams,
} from '@nemo/sdk/generated/platform/schema';
import { JOB_SOURCE } from '@studio/components/dataViews/JobsDataView/constants';
import type { QueryClient } from '@tanstack/react-query';

export interface UseCustomizationJobForModelResult {
  jobName: string | undefined;
  isLoading: boolean;
}

const CUSTOMIZATION_JOB_STALE_TIME_MS = 5 * 60 * 1000;

/**
 * A ModelEntity has no job reference, so query jobs by `spec.output.name` (set by every
 * customizer backend to the output model or adapter name) — an exact match returning at
 * most one job, newest first when a name was reused across re-runs.
 */
const getCustomizationJobForOutputParams = (
  outputName: string | undefined
): JobsListJobsParams => ({
  page: 1,
  page_size: 1,
  sort: '-created_at' as HelixJobListSortField,
  filter: {
    source: JOB_SOURCE.CUSTOMIZATION,
    'spec.output.name': outputName,
  } as unknown as HelixJobsListFilter,
});

/**
 * Imperative counterpart of {@link useCustomizationJobForModel} for callbacks that may
 * not call hooks (e.g. a row-actions menu). Shares the hook's query cache. Takes the
 * output name so it also resolves LoRA adapters, which are named by `spec.output.name`
 * just like full-weight checkpoints.
 */
export const fetchCustomizationJobNameForOutput = async (
  queryClient: QueryClient,
  workspace: string,
  outputName: string
): Promise<string | undefined> => {
  const page = await queryClient.fetchQuery({
    ...getJobsListJobsQueryOptions(workspace, getCustomizationJobForOutputParams(outputName)),
    staleTime: CUSTOMIZATION_JOB_STALE_TIME_MS,
  });
  return page.data?.[0]?.name;
};

/**
 * Resolves the customization job that produced a fine-tuned checkpoint, for linking a
 * model back to its originating job.
 */
export const useCustomizationJobForModel = (
  workspace: string,
  model: ModelEntity | null | undefined
): UseCustomizationJobForModelResult => {
  const modelName = model?.name;
  const enabled = Boolean(modelName);

  const { data, isLoading } = useJobsListJobs(
    workspace,
    getCustomizationJobForOutputParams(modelName),
    { query: { enabled, staleTime: CUSTOMIZATION_JOB_STALE_TIME_MS } }
  );

  return { jobName: data?.data?.[0]?.name, isLoading: enabled && isLoading };
};
