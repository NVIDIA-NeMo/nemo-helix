// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { withOperators } from '@nemo/common/src/api/filterOperators';
import { jobsListJobs } from '@nemo/sdk/generated/platform/jobs';
import type {
  HelixJobListSortField,
  HelixJobResponse,
  HelixJobsListFilter,
} from '@nemo/sdk/generated/platform/schema';

const EVALUATOR_JOB_SOURCES = ['nemo-evaluator', 'nemo-evaluator.agent-evaluate'] as const;

const PAGE_SIZE = 50;

export const fetchEvaluatorJobs = async (
  workspace: string,
  signal: AbortSignal,
  shouldStop?: (accumulated: HelixJobResponse[]) => boolean
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
        filter: withOperators<HelixJobsListFilter>({
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
