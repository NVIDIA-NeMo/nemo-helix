// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { DEFAULT_QUERY_RETRY_COUNT } from '@studio/api/queryClient';
import { LIST_POLL_MS, runPollInterval } from '@studio/api/useLatestAnalysisRun';
import { mockAnalysisRunWithJob } from '@studio/mocks/handlers/insights';

const failedFetchAttempts = DEFAULT_QUERY_RETRY_COUNT + 1;

describe('runPollInterval', () => {
  it('polls an in-flight run at the job interval', () => {
    expect(runPollInterval(mockAnalysisRunWithJob('agent', HelixJobStatus.active), 0)).toBe(
      JOB_POLLING_INTERVAL_MS
    );
  });

  it('backs off to the list interval instead of stopping when a refetch fails', () => {
    expect(
      runPollInterval(mockAnalysisRunWithJob('agent', HelixJobStatus.active), failedFetchAttempts)
    ).toBe(LIST_POLL_MS);
  });

  it('stops polling a finished run even when its last refetch failed', () => {
    expect(
      runPollInterval(
        mockAnalysisRunWithJob('agent', HelixJobStatus.completed),
        failedFetchAttempts
      )
    ).toBe(false);
  });

  it('does not poll before the run has loaded', () => {
    expect(runPollInterval(undefined, failedFetchAttempts)).toBe(false);
  });
});
