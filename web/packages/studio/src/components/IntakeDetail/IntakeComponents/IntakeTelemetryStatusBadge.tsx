// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatusBadge, type StatusConfigEntry } from '@nemo/common/src/components/StatusBadge';
import type { SpanStatus } from '@nemo/sdk/generated/platform/schema';
import { Ban, CircleCheck, CircleHelp, CircleX } from 'lucide-react';

const STATUS_CONFIG: Record<SpanStatus, StatusConfigEntry> = {
  success: { label: 'Completed', color: 'green', icon: CircleCheck },
  error: { label: 'Error', color: 'red', icon: CircleX },
  cancelled: { label: 'Cancelled', color: 'yellow', icon: Ban },
  unknown: { label: 'Unknown', color: 'gray', icon: CircleHelp },
};

export interface IntakeTelemetryStatusBadgeProps {
  status: SpanStatus | undefined;
}

export const IntakeTelemetryStatusBadge = ({ status }: IntakeTelemetryStatusBadgeProps) => (
  <StatusBadge status={status} statusConfig={STATUS_CONFIG} fallback={STATUS_CONFIG.unknown} />
);
