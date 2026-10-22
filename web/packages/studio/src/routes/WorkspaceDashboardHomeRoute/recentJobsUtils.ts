// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { jsonFilter } from '@nemo/common/src/api/filterOperators';
import { HelixJobTerminalStatuses } from '@nemo/common/src/constants/query';
import { type HelixJobsListFilter, HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { getHiddenJobSources } from '@studio/components/dataViews/JobsDataView/constants';

/** Jobs in progress, or finished in the last 48 hours (`updated_at` stands in for finish time). */
export const buildFilter = (finishedSince: string): HelixJobsListFilter =>
  jsonFilter<HelixJobsListFilter>({
    $and: [
      { source: { $nin: getHiddenJobSources() } },
      {
        $or: [
          { status: { $nin: HelixJobTerminalStatuses } },
          { updated_at: { $gte: finishedSince } },
        ],
      },
    ],
  });

export const isJobRunning = (status: HelixJobStatus): boolean =>
  !HelixJobTerminalStatuses.includes(status) && status !== HelixJobStatus.paused;
