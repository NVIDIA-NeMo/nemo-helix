// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';

/** Empty chart copy describes telemetry, never the input dataset. */
export function getTrainingMetricsEmptyState(
  status?: HelixJobStatus,
  metric = 'Training'
): { title: string; description: string } {
  switch (status) {
    case HelixJobStatus.created:
    case HelixJobStatus.pending:
    case HelixJobStatus.active:
    case HelixJobStatus.resuming:
      return {
        title: `${metric} metrics are not available yet`,
        description:
          'Metrics should appear after training starts and the first steps are reported.',
      };
    case HelixJobStatus.completed:
      return {
        title: `No ${metric.toLowerCase()} metrics were reported`,
        description: 'This job completed without recording metrics for this chart.',
      };
    case HelixJobStatus.error:
      return {
        title: `${metric} metrics are unavailable`,
        description:
          'This job failed before reporting metrics for this chart. Check the job logs for details.',
      };
    case HelixJobStatus.cancelled:
    case HelixJobStatus.cancelling:
      return {
        title: `${metric} metrics are unavailable`,
        description: 'This job was stopped before reporting metrics for this chart.',
      };
    case HelixJobStatus.paused:
    case HelixJobStatus.pausing:
      return {
        title: `${metric} metrics are not available yet`,
        description:
          'This job is paused. Metrics should appear after training resumes and steps are reported.',
      };
    default:
      return {
        title: `${metric} metrics are unavailable`,
        description: 'No metrics have been reported for this chart.',
      };
  }
}
