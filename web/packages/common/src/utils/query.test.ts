// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { CJobTerminalStatuses } from '@nemo/common/src/constants/query';
import { getJobRefetchInterval } from '@nemo/common/src/utils/query';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema/HelixJobStatus';

describe('getJobRefetchInterval', () => {
  it('should return JOB_POLLING_INTERVAL_MS if status is undefined', () => {
    expect(getJobRefetchInterval(undefined)).toBe(JOB_POLLING_INTERVAL_MS);
  });
  it.each([...CJobTerminalStatuses])('should return false if status is %s', (status) => {
    expect(getJobRefetchInterval(status)).toBe(false);
  });
  it('should return JOB_POLLING_INTERVAL_MS if status is not terminal', () => {
    // Platform uses 'active' instead of 'running'
    expect(getJobRefetchInterval(HelixJobStatus.active)).toBe(JOB_POLLING_INTERVAL_MS);
    expect(getJobRefetchInterval(HelixJobStatus.pending)).toBe(JOB_POLLING_INTERVAL_MS);
    expect(getJobRefetchInterval(HelixJobStatus.created)).toBe(JOB_POLLING_INTERVAL_MS);
    expect(getJobRefetchInterval(HelixJobStatus.cancelling)).toBe(JOB_POLLING_INTERVAL_MS);
  });
});
