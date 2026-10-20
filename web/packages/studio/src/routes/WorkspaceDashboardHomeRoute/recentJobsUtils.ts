// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { jsonFilter } from '@nemo/common/src/api/filterOperators';
import { HelixJobTerminalStatuses } from '@nemo/common/src/constants/query';
import { utcToLocalDate } from '@nemo/common/src/utils/date';
import {
  type HelixJobResponse,
  type HelixJobsListFilter,
  HelixJobStatus,
} from '@nemo/sdk/generated/platform/schema';
import { getHiddenJobSources } from '@studio/components/dataViews/JobsDataView/constants';
import { IN_PROGRESS_JOB_STATUSES } from '@studio/constants/platformJobs';

/** Jobs in progress, or finished in the last 48 hours (`updated_at` stands in for finish time). */
export const buildFilter = (finishedSince: string): HelixJobsListFilter =>
  jsonFilter<HelixJobsListFilter>({
    $and: [
      { source: { $nin: getHiddenJobSources() } },
      {
        $or: [
          { status: { $in: IN_PROGRESS_JOB_STATUSES } },
          {
            $and: [
              { status: { $in: HelixJobTerminalStatuses } },
              { updated_at: { $gte: finishedSince } },
            ],
          },
        ],
      },
    ],
  });

export const isRecentJob = (job: HelixJobResponse, finishedSinceMs: number): boolean => {
  if (IN_PROGRESS_JOB_STATUSES.includes(job.status)) {
    return true;
  }
  const finishedAt = utcToLocalDate(job.updated_at);
  return finishedAt !== undefined && finishedAt.getTime() >= finishedSinceMs;
};

export const isJobRunning = (status: HelixJobStatus): boolean =>
  IN_PROGRESS_JOB_STATUSES.includes(status) && status !== HelixJobStatus.paused;
