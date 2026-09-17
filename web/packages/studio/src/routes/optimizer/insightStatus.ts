// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { InsightStatus } from '@nemo/sdk/generated/insights/schema';
import type { Button, Tag } from '@nvidia/foundations-react-core';
import type { ComponentProps } from 'react';

type TagColor = ComponentProps<typeof Tag>['color'];

/** Tag color per insight status. Exhaustive over InsightStatus. */
export const INSIGHT_STATUS_COLOR: Record<InsightStatus, TagColor> = {
  open: 'blue',
  resolved: 'green',
  deleted: 'gray',
};

export const insightStatusColor = (status: InsightStatus): TagColor =>
  INSIGHT_STATUS_COLOR[status] ?? 'gray';

/** A status-change action rendered as a button on the insight page. */
export interface InsightAction {
  label: string;
  /** Status the button transitions the insight to. */
  target: InsightStatus;
  kind: 'primary' | 'secondary';
  /** Optional button color; 'brand' renders the green primary CTA. */
  color?: ComponentProps<typeof Button>['color'];
}

// Canonical order: secondary actions first, Run experiment (green brand CTA) on the right.
const OPEN: InsightAction = {
  label: 'Run experiment',
  target: 'open',
  kind: 'primary',
  color: 'brand',
};
const DELETE: InsightAction = { label: 'Delete', target: 'deleted', kind: 'secondary' };
const RESOLVE: InsightAction = { label: 'Resolve', target: 'resolved', kind: 'primary' };

export const insightActions = (status: InsightStatus): InsightAction[] => {
  return status === 'open' ? [DELETE, RESOLVE] : [OPEN];
};
