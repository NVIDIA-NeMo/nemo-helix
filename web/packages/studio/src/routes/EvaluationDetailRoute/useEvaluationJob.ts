// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_POLLING_INTERVAL_LONG } from '@nemo/common/src/constants';
import { HelixJobTerminalStatuses } from '@nemo/common/src/constants/query';
import type { EvaluationResponse, HelixJobResponse } from '@nemo/sdk/generated/platform/schema';
import { fetchEvaluatorJobs } from '@studio/api/evaluation/evaluator-jobs';
import {
  type EvalJobRow,
  evalDurationMs,
  publishedEvaluationName,
  toEvalJobRow,
} from '@studio/api/evaluation/utils';
import { useQuery } from '@tanstack/react-query';

/** A run that reached Intake carries its recorded duration and its ingested test cases. Before that
 *  the evaluation exists but holds nothing that says how the run went. */
const isPublished = (evaluation?: EvaluationResponse): boolean =>
  evaluation != null &&
  (evalDurationMs(evaluation.metadata) !== undefined || (evaluation.test_case_count ?? 0) > 0);

/** The evaluator job that produced one evaluation, and the status to show for it.
 *
 *  `EvaluationResponse.status` is a producer-defined free-form field that nothing in this pipeline
 *  writes, so the job is the only place a run's outcome is recorded; a published run whose job has
 *  been pruned falls back to completed, which publishing guarantees.
 *
 *  Jobs carry no evaluation filter, so the job is found by scanning evaluator jobs newest-first and
 *  stopping at the first match. */
export const useEvaluationJob = (
  workspace: string,
  evaluationName: string,
  evaluation?: EvaluationResponse
): { job?: EvalJobRow; status?: string } => {
  const published = isPublished(evaluation);

  const { data: job } = useQuery({
    queryKey: ['evaluation-job', workspace, evaluationName] as const,
    queryFn: async ({ signal }) => {
      const matches = (candidate: HelixJobResponse) =>
        publishedEvaluationName(candidate) === evaluationName;
      const jobs = await fetchEvaluatorJobs(workspace, signal, (all) => all.some(matches));
      const found = jobs.find(matches);
      return found ? toEvalJobRow(found) : null;
    },
    enabled: !!workspace && !!evaluationName,
    refetchInterval: (query) => {
      if (published) return false;
      const status = query.state.data?.status;
      const settled = HelixJobTerminalStatuses.some((terminal) => terminal === status);
      return settled ? false : JOB_POLLING_INTERVAL_LONG;
    },
  });

  return {
    job: job ?? undefined,
    status: job?.status ?? (published ? 'completed' : undefined),
  };
};
