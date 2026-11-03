// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { getTrainingMetricsEmptyState } from '@studio/util/trainingMetricsEmptyState';

describe('getTrainingMetricsEmptyState', () => {
  it.each([
    HelixJobStatus.created,
    HelixJobStatus.queued,
    HelixJobStatus.pending,
    HelixJobStatus.active,
  ])('says metrics are not available yet while the job is %s', (status) => {
    expect(getTrainingMetricsEmptyState(status).title).toBe(
      'Training metrics are not available yet'
    );
  });

  it('names the metric in the copy', () => {
    expect(getTrainingMetricsEmptyState(HelixJobStatus.queued, 'Validation').title).toBe(
      'Validation metrics are not available yet'
    );
  });

  it('falls back to the generic copy without a status', () => {
    expect(getTrainingMetricsEmptyState(undefined).description).toBe(
      'No metrics have been reported for this chart.'
    );
  });
});
