// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import type { ExecuteJob, ExecuteJobsPage } from '@nemo/sdk/generated/agents/schema';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema/HelixJobStatus';
import { DEFAULT_QUERY_RETRY_COUNT } from '@studio/api/queryClient';
import { LIST_POLL_MS, runPollInterval } from '@studio/api/useLatestAnalysisRun';
import { mockAnalysisJob } from '@studio/mocks/handlers/insights';

const failedFetchAttempts = DEFAULT_QUERY_RETRY_COUNT + 1;

const pageOf = (...jobs: ExecuteJob[]): ExecuteJobsPage => ({ data: jobs });

describe('runPollInterval', () => {
  it('polls an in-flight run at the job interval', () => {
    expect(runPollInterval(pageOf(mockAnalysisJob('agent', HelixJobStatus.active)), 0)).toBe(
      JOB_POLLING_INTERVAL_MS
    );
  });

  it('polls a queued run at the job interval', () => {
    expect(runPollInterval(pageOf(mockAnalysisJob('agent', HelixJobStatus.queued)), 0)).toBe(
      JOB_POLLING_INTERVAL_MS
    );
  });

  it('backs off to the list interval when a refetch fails', () => {
    expect(
      runPollInterval(pageOf(mockAnalysisJob('agent', HelixJobStatus.active)), failedFetchAttempts)
    ).toBe(LIST_POLL_MS);
  });

  it('keeps polling at the list interval after a run finishes, to notice new runs', () => {
    expect(runPollInterval(pageOf(mockAnalysisJob('agent', HelixJobStatus.completed)), 0)).toBe(
      LIST_POLL_MS
    );
  });

  it('polls at the list interval when the agent has no runs yet', () => {
    expect(runPollInterval(pageOf(), 0)).toBe(LIST_POLL_MS);
    expect(runPollInterval(undefined, 0)).toBe(LIST_POLL_MS);
  });
});
