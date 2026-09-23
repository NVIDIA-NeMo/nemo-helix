// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NamedEntity } from '@nemo/common/src/namedEntity';
import {
  evaluatorGetEvaluateJob,
  evaluatorListEvaluateJobResults,
  evaluatorListEvaluateJobs,
} from '@nemo/sdk/generated/evaluator/evaluator-plugin-jobs-routes';
import {
  EvaluateJob,
  EvaluateJobsPage,
  HelixJobListResultResponse,
} from '@nemo/sdk/generated/evaluator/schema';
import { useDetailsChartsStore } from '@studio/api/evaluation/details/useDetailsChartsStore';
import { getEvaluationJobModel, isEvaluationJobSucceeded } from '@studio/selectors/evaluationJob';
import { useQuery } from '@tanstack/react-query';

/**
 * Helper to exhaust pagination for fetch for evaluation jobs by workspace
 *
 * @example
 * Example usage in a component:
 * fetchAllEvaluationsByConfig(workspace);
 *
 * @returns {Object} Promise<EvaluateJobsPage>
 */
const fetchAllEvaluationsByConfig = async (
  workspace: string | undefined
): Promise<EvaluateJobsPage> => {
  if (!workspace) {
    return {
      data: [],
    };
  }

  let allEvaluations: EvaluateJobsPage['data'] = [];
  let currentPage = 1;
  let totalPages = 1; // initialize to avoid infinite loop
  const pageSize = 100;

  while (currentPage <= totalPages) {
    const response = await evaluatorListEvaluateJobs(workspace, {
      page_size: pageSize,
      page: currentPage,
      sort: '-created_at',
    });

    const { data, pagination } = response;

    allEvaluations = [...allEvaluations, ...data];

    totalPages = pagination?.total_pages || 1;
    currentPage++;
  }

  return {
    data: allEvaluations,
    pagination: {
      total_results: allEvaluations.length,
      total_pages: Math.ceil(allEvaluations.length / pageSize),
      page: currentPage - 1,
      page_size: pageSize,
      current_page_size: Math.min(
        pageSize,
        Math.max(0, allEvaluations.length - (currentPage - 1) * pageSize)
      ),
    },
  };
};

/**
 * Helper to get all unique models with their associated evaluation jobs by workspace
 * Optionally provide an evaluationId that will be omitted from the list as it is the original
 *
 * @example
 * Example usage in a component:
 * useUniqueModelsByConfig(workspace, id);
 *
 * @returns {Object} <Record<string, EvaluateJob[]>
 */

export const useUniqueModelsByConfig = (workspace: string | undefined, evaluationId?: string) => {
  return useQuery<Record<string, EvaluateJob[]>, Error>({
    queryKey: ['uniqueModelsByConfig', workspace, evaluationId],
    queryFn: async () => {
      const evaluationsByConfig = await fetchAllEvaluationsByConfig(workspace);

      // Group evaluations by model_name
      return evaluationsByConfig.data.reduce(
        (acc, job) => {
          const modelName = getEvaluationJobModel(job);
          // Skip the evaluation which is the focus, or has not succeeded
          if (job.id == evaluationId || !isEvaluationJobSucceeded(job.status)) {
            return acc;
          }

          if (modelName) {
            if (!acc[modelName]) {
              acc[modelName] = [] as EvaluateJob[];
            }
            acc[modelName].push(job);
          }
          return acc;
        },
        {} as Record<string, EvaluateJob[]>
      );
    },
    placeholderData: (prevData) => prevData ?? {},
  });
};

/**
 * Helper to fetch evaluation results for all selected evaluations in the Details Chart store.
 *
 * @example
 * Example usage in a component:
 * useAggregatedEvaluationResults(config);
 *
 * @returns UseQueryResult<HelixJobListResultResponse[], Error>
 */
export const useAggregatedEvaluationResults = (config: NamedEntity) => {
  const { selectedEvaluations } = useDetailsChartsStore();

  return useQuery<HelixJobListResultResponse[], Error>({
    queryKey: ['aggregatedEvaluationResults', config, selectedEvaluations],
    queryFn: async () => {
      if (!config.workspace) {
        return [];
      }
      return await Promise.all(
        selectedEvaluations.map((evaluationId) =>
          evaluatorListEvaluateJobResults(config.workspace!, evaluationId)
        )
      );
    },
    enabled: !!config.workspace && selectedEvaluations.length > 0,
    placeholderData: (prevData) => prevData ?? [],
  });
};

/**
 * Helper to get all evaluation jobs by id - jobs are retrieved from the set of selectedEvaluations
 *
 * @example
 * Example usage in a component:
 * useAggregatedEvaluations(config);
 *
 * @returns {Array} EvaluateJob[]
 */
export const useAggregatedEvaluations = (config: NamedEntity) => {
  const { selectedEvaluations } = useDetailsChartsStore();

  return useQuery<EvaluateJob[], Error>({
    queryKey: ['aggregatedEvaluations', config, selectedEvaluations],
    queryFn: async () => {
      if (!config.workspace) {
        return [];
      }
      return await Promise.all(
        selectedEvaluations.map((evaluationId) =>
          evaluatorGetEvaluateJob(config.workspace!, evaluationId)
        )
      );
    },
    enabled: !!config.workspace && selectedEvaluations.length > 0,
    placeholderData: (prevData) => prevData ?? [],
  });
};
