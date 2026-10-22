// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { isJobRunning } from '@studio/routes/WorkspaceDashboardHomeRoute/recentJobsUtils';

describe('isJobRunning', () => {
  it.each([HelixJobStatus.created, HelixJobStatus.active, HelixJobStatus.pausing])(
    'counts %s as running',
    (status) => {
      expect(isJobRunning(status)).toBe(true);
    }
  );

  it.each([HelixJobStatus.paused, HelixJobStatus.completed, HelixJobStatus.error])(
    'does not count %s as running',
    (status) => {
      expect(isJobRunning(status)).toBe(false);
    }
  );

  it('counts a status newer than this bundle as running', () => {
    expect(isJobRunning('queued' as HelixJobStatus)).toBe(true);
  });
});
