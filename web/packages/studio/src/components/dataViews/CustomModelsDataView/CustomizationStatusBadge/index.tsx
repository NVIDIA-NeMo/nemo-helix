// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatusBadge, type StatusConfigEntry } from '@nemo/common/src/components/StatusBadge';
import type { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { getFormattedCustomizationStatus } from '@studio/util/customizations';
import type { FC } from 'react';

const STATUS_CONFIG: Record<string, StatusConfigEntry> = {
  cancelled: { label: 'Cancelled', color: 'red' },
  failed: { label: 'Failed', color: 'red' },
  created: { label: 'Created', color: 'blue' },
  running: { label: 'Running', color: 'blue' },
  pending: { label: 'Pending', color: 'yellow' },
  completed: { label: 'Completed', color: 'green' },
};

interface Props {
  status: HelixJobStatus | string;
  progressPercent?: number;
}

// TODO: Rename this to JobStatusBadge
export const CustomizationStatusBadge: FC<Props> = ({ status, progressPercent }) => (
  <StatusBadge
    status={status}
    statusConfig={STATUS_CONFIG}
    label={getFormattedCustomizationStatus(status, progressPercent)}
  />
);
