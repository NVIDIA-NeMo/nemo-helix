// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { jsonFilter, withOperators } from '@nemo/common/src/api/filterOperators';
import { jobsListJobs } from '@nemo/sdk/generated/platform/jobs';
import type {
  HelixJobListSortField,
  HelixJobResponse,
  HelixJobsListFilter,
} from '@nemo/sdk/generated/platform/schema';

const EVALUATOR_JOB_SOURCES = ['nemo-evals', 'nemo-evals.agent-evaluate'] as const;

const PAGE_SIZE = 50;

export const fetchEvaluatorJobs = async (
  workspace: string,
  signal: AbortSignal,
  shouldStop?: (accumulated: HelixJobResponse[]) => boolean,
  agentName?: string
): Promise<HelixJobResponse[]> => {
  const all: HelixJobResponse[] = [];
  let page = 1;
  while (true) {
    const res = await jobsListJobs(
      workspace,
      {
        page,
        page_size: PAGE_SIZE,
        sort: '-created_at' as HelixJobListSortField,
        filter: agentName
          ? jsonFilter<HelixJobsListFilter>({
              $and: [
                { source: { $in: [...EVALUATOR_JOB_SOURCES] } },
                {
                  $or: [
                    {
                      'spec.target.kind': { $eq: 'agent' },
                      'spec.target.agent.name': { $in: [agentName, `${workspace}/${agentName}`] },
                    },
                    {
                      'spec.target.kind': { $in: ['fabric', 'harbor', 'gym'] },
                      'spec.target.source.agent': { $in: [agentName, `${workspace}/${agentName}`] },
                    },
                    // Dataset-driven `evaluate` jobs take the bare endpoint as their target, with
                    // no `kind` to match on.
                    {
                      'spec.target.format': { $eq: 'generic' },
                      'spec.target.name': { $in: [agentName, `${workspace}/${agentName}`] },
                    },
                  ],
                },
              ],
            })
          : withOperators<HelixJobsListFilter>({
              source: { $in: [...EVALUATOR_JOB_SOURCES] },
            }),
      },
      signal
    );
    const batch = res?.data ?? [];
    all.push(...batch);
    if (batch.length < PAGE_SIZE) break;
    if (shouldStop?.(all)) break;
    page++;
  }
  return all;
};

/** The newest evaluator job that published to `evaluationName`, matched server-side on its spec. */
export const fetchEvaluationJob = async (
  workspace: string,
  evaluationName: string,
  signal: AbortSignal
): Promise<HelixJobResponse | undefined> => {
  const res = await jobsListJobs(
    workspace,
    {
      page_size: 1,
      sort: '-created_at' as HelixJobListSortField,
      filter: jsonFilter<HelixJobsListFilter>({
        source: { $in: [...EVALUATOR_JOB_SOURCES] },
        'spec.publication.intake.evaluation_id': { $eq: evaluationName },
      }),
    },
    signal
  );
  return res?.data?.[0];
};
