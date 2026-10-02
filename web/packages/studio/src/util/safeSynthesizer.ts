// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';

export const TERMINAL_STATUSES: readonly string[] = [
  HelixJobStatus.completed,
  HelixJobStatus.cancelled,
  HelixJobStatus.error,
];
export const SUCCESSFUL_STATUS: HelixJobStatus = HelixJobStatus.completed;

export const isJobTerminated = (status: string | undefined) => {
  if (!status) return false;
  return TERMINAL_STATUSES.includes(status);
};

export const isJobSuccessful = (status: string | undefined) => {
  if (!status) return false;
  return status === SUCCESSFUL_STATUS;
};
