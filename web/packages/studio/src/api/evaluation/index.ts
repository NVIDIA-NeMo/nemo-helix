// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { evalsListEvaluateJobs } from '@nemo/sdk/generated/evals/evals-plugin-jobs-routes';
import type { EvalsListEvaluateJobsParams } from '@nemo/sdk/generated/evals/schema';
import type { EvaluationJobWithTaskMetrics } from '@studio/api/evaluation/useEvaluationsWithMetrics';

export interface FetchEvaluationsWithMetricsOptions {
  workspace: string;
  query?: EvalsListEvaluateJobsParams;
  signal?: AbortSignal;
}

export const fetchEvaluationsWithMetrics = async ({
  workspace,
  query,
  signal,
}: FetchEvaluationsWithMetricsOptions) => {
  const evaluations = await evalsListEvaluateJobs(workspace, query, signal);
  if (evaluations.data?.length > 0) {
    const evaluationsWithMetrics = await Promise.all(
      evaluations.data.map(async (evaluation) => {
        try {
          if (!evaluation.workspace) {
            return evaluation;
          }
          if (!evaluation.name) {
            return evaluation;
          }

          const evaluationWithMetrics: EvaluationJobWithTaskMetrics = { ...evaluation, tasks: {} };
          return evaluationWithMetrics;
        } catch {
          return evaluation;
        }
      })
    );
    return { ...evaluations, data: evaluationsWithMetrics };
  }
  return evaluations;
};
