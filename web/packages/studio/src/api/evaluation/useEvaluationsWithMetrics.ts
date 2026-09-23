/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
import {
  EvaluateJob,
  EvaluateJobsPage,
  HelixJobStatus as JobStatus,
  EvaluatorListEvaluateJobsParams,
} from '@nemo/sdk/generated/evaluator/schema';
import { EvaluationApiError } from '@studio/api/evaluation/EvaluationApiError';
import { fetchEvaluationsWithMetrics } from '@studio/api/evaluation/index';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { UseQueryOptions, queryOptions, useQuery } from '@tanstack/react-query';

/** Task result shape appended by fetchEvaluationsWithMetrics. */
export type TaskResultOutput = Record<string, unknown>;

export interface EvaluationJobWithTaskMetrics extends EvaluateJob {
  tasks?: Record<string, TaskResultOutput>;
}

export interface EvaluationJobsWithTaskMetricsPage extends Omit<EvaluateJobsPage, 'data'> {
  data: EvaluationJobWithTaskMetrics[];
}

export interface UseEvaluationsWithMetricsOptions {
  queryOptions?: Omit<
    UseQueryOptions<EvaluationJobsWithTaskMetricsPage, EvaluationApiError>,
    'queryFn' | 'queryKey'
  >;
  query?: EvaluatorListEvaluateJobsParams;
  filters?: {
    tags?: {
      projectId?: string;
      modelId?: string;
    };
    status?: JobStatus;
  };
}

export const getEvaluationsWithMetricsQueryOptions = (
  workspace: string,
  query?: EvaluatorListEvaluateJobsParams
) => {
  return queryOptions<EvaluationJobsWithTaskMetricsPage, EvaluationApiError>({
    queryKey: ['evaluationsWithMetrics', query],
    queryFn: ({ signal }) => fetchEvaluationsWithMetrics({ workspace, signal, query }),
  });
};

export const useEvaluationsWithMetrics = (options?: UseEvaluationsWithMetricsOptions) => {
  let filterQueryParams = {};
  const workspace = useWorkspaceFromPath();
  if (options?.filters) {
    const { tags, status } = options.filters;
    let filter = '';

    const tagFilters = [];
    if (tags?.projectId) {
      tagFilters.push(tags.projectId);
    }
    if (tags?.modelId) {
      tagFilters.push(tags.modelId);
    }
    if (tagFilters.length > 0) {
      const tagFilterString = tagFilters.join('_');
      filter = `tag:${tagFilterString}`;
    }

    if (status) {
      const statusFilterString = `status:${status}`;
      filter = filter ? `${filter},${statusFilterString}` : statusFilterString;
    }

    filterQueryParams = { filter };
  }

  const query = {
    ...options?.query,
    ...filterQueryParams,
  };

  return useQuery({
    ...getEvaluationsWithMetricsQueryOptions(workspace, query),
    ...options?.queryOptions,
  });
};
