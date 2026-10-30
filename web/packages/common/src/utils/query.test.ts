// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';

import { getJobRefetchInterval, getRequestStatus } from './query';
import { JOB_POLLING_INTERVAL_MS } from '../constants';
import { CJobTerminalStatuses } from '../constants/query';

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

describe('getRequestStatus', () => {
  it('reports an error over everything else', () => {
    expect(getRequestStatus({ error: new Error('x'), isLoading: true })).toBe('error');
  });

  it('is loading on the first load and while showing placeholder rows', () => {
    expect(getRequestStatus({ isLoading: true })).toBe('loading');
    expect(getRequestStatus({ isPlaceholderData: true })).toBe('loading');
  });

  it('is not loading during a background refetch of loaded data', () => {
    expect(getRequestStatus({ isLoading: false, isPlaceholderData: false })).toBeUndefined();
  });
});
