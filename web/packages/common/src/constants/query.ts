// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';

// Customizer uses Platform SDK status
export const CJobCancellableStatuses: HelixJobStatus[] = [
  HelixJobStatus.created,
  HelixJobStatus.pending,
  HelixJobStatus.active, // was 'running'
];
export const CJobLaunchableStatuses: HelixJobStatus[] = [HelixJobStatus.completed];

export const CJobTerminalStatuses: HelixJobStatus[] = [
  HelixJobStatus.completed,
  HelixJobStatus.error, // was 'failed'
  HelixJobStatus.cancelled,
];
export const HelixJobTerminalStatuses: HelixJobStatus[] = [
  HelixJobStatus.completed,
  HelixJobStatus.cancelled,
  HelixJobStatus.error,
];
