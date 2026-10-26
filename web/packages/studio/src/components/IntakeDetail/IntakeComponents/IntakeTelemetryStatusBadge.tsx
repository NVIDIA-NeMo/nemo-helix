// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatusBadge, type StatusConfigEntry } from '@nemo/common/src/components/StatusBadge';
import { Cancel, CheckCircle, HelpCircle, CloseCircle } from '@nemo/icons/icons';
import type { SpanStatus } from '@nemo/sdk/generated/platform/schema';

const STATUS_CONFIG: Record<SpanStatus, StatusConfigEntry> = {
  success: { label: 'Completed', color: 'green', icon: CheckCircle },
  error: { label: 'Error', color: 'red', icon: CloseCircle },
  cancelled: { label: 'Cancelled', color: 'yellow', icon: Cancel },
  unknown: { label: 'Unknown', color: 'gray', icon: HelpCircle },
};

export interface IntakeTelemetryStatusBadgeProps {
  status: SpanStatus | undefined;
}

export const IntakeTelemetryStatusBadge = ({ status }: IntakeTelemetryStatusBadgeProps) => (
  <StatusBadge status={status} statusConfig={STATUS_CONFIG} fallback={STATUS_CONFIG.unknown} />
);
