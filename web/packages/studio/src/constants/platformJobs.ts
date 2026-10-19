// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixJobTerminalStatuses } from '@nemo/common/src/constants/query';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';

export const STATUS_FILTER_OPTIONS = [
  { value: HelixJobStatus.created, label: 'Created' },
  { value: HelixJobStatus.pending, label: 'Pending' },
  { value: HelixJobStatus.active, label: 'Active' },
  { value: HelixJobStatus.completed, label: 'Completed' },
  { value: HelixJobStatus.error, label: 'Error' },
  { value: HelixJobStatus.cancelled, label: 'Cancelled' },
  { value: HelixJobStatus.cancelling, label: 'Cancelling' },
  { value: HelixJobStatus.paused, label: 'Paused' },
  { value: HelixJobStatus.pausing, label: 'Pausing' },
  { value: HelixJobStatus.resuming, label: 'Resuming' },
];

/** Every status outside `HelixJobTerminalStatuses`, so new statuses count as in progress. */
export const IN_PROGRESS_JOB_STATUSES: readonly HelixJobStatus[] = Object.values(
  HelixJobStatus
).filter((status) => !HelixJobTerminalStatuses.includes(status));
