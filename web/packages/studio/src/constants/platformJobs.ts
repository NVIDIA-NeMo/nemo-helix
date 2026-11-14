// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema/HelixJobStatus';

export const getDeleteJobsDescription = (count: number) =>
  count === 1
    ? "The job is removed from this workspace and can't be restored."
    : "The jobs are removed from this workspace and can't be restored.";

export const STATUS_FILTER_OPTIONS = [
  { value: HelixJobStatus.created, label: 'Created' },
  { value: HelixJobStatus.queued, label: 'Queued' },
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
