// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { STATUS_FILTER_OPTIONS } from '@studio/constants/platformJobs';

describe('STATUS_FILTER_OPTIONS', () => {
  it('offers every job status', () => {
    expect(STATUS_FILTER_OPTIONS.map(({ value }) => value).sort()).toEqual(
      Object.values(HelixJobStatus).sort()
    );
  });

  it('labels the queued status', () => {
    expect(STATUS_FILTER_OPTIONS).toContainEqual({ value: HelixJobStatus.queued, label: 'Queued' });
  });
});
