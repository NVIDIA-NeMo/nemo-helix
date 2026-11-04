// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { BadgeStatus } from '@nemo/common/src/components/StatusBadge';
import type { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';

const RUN_BADGES: Partial<Record<HelixJobStatus, { status: BadgeStatus; label: string }>> = {
  created: { status: 'pending', label: 'Queued' },
  pending: { status: 'pending', label: 'Queued' },
  active: { status: 'active', label: 'Running' },
  error: { status: 'error', label: 'Failed' },
};

const NOT_SUBMITTED = { status: 'unavailable', label: 'Not submitted' } as const;

/** The badge for an analysis run: its job's status, or "Not submitted" when no job landed. */
export const analysisRunBadge = (
  submitted: boolean,
  status?: HelixJobStatus
): { status: BadgeStatus | string | undefined; label?: string } =>
  submitted ? ((status && RUN_BADGES[status]) ?? { status, label: undefined }) : NOT_SUBMITTED;
