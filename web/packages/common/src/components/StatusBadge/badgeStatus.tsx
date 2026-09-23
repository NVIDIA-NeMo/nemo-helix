// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { BadgeProps as KuiBadgeProps } from '@nvidia/foundations-react-core';
import {
  CircleCheck,
  CircleX,
  RefreshCw,
  Ban,
  CircleHelp,
  TriangleAlert,
  Pause,
} from 'lucide-react';
import { type ComponentType, type SVGProps } from 'react';

export interface BadgeProps {
  label: string;
  icon: ComponentType<SVGProps<SVGSVGElement>>;
  color?: Exclude<KuiBadgeProps['color'], null>;
}

export interface StatusConfigEntry {
  label: string;
  color: Exclude<KuiBadgeProps['color'], null>;
  icon?: ComponentType<SVGProps<SVGSVGElement>>;
}

export type BadgeStatus<T = HelixJobStatus> =
  | Exclude<T, undefined>
  | 'error'
  | 'active'
  | 'in_progress'
  | 'unavailable'
  | 'ready'
  | 'unknown'
  | 'default'
  | 'starting'
  | 'running'
  | 'failed'
  | 'deleting'
  | 'deleted'
  | 'lost';

export const badgeStatus: Record<BadgeStatus | 'default', BadgeProps> = {
  created: {
    label: 'Created',
    color: 'teal',
    icon: CircleCheck,
  },
  completed: {
    label: 'Completed',
    color: 'green',
    icon: CircleCheck,
  },
  error: {
    label: 'Error',
    color: 'red',
    icon: CircleX,
  },
  pending: {
    label: 'Pending',
    color: 'gray',
    icon: RefreshCw,
  },
  active: {
    label: 'Active',
    color: 'blue',
    icon: RefreshCw,
  },
  in_progress: {
    label: 'In Progress',
    color: 'blue',
    icon: RefreshCw,
  },
  cancelling: {
    label: 'Cancelling',
    color: 'yellow',
    icon: Ban,
  },
  cancelled: {
    label: 'Cancelled',
    color: 'yellow',
    icon: Ban,
  },
  unavailable: {
    label: 'Unavailable',
    color: 'gray',
    icon: TriangleAlert,
  },
  default: {
    label: 'Unknown',
    color: 'purple',
    icon: CircleHelp,
  },
  // What does it even mean for the job to be "ready" to be used..
  ready: {
    label: 'Ready',
    color: 'green',
    icon: CircleCheck,
  },
  unknown: {
    label: 'Unknown',
    color: 'purple',
    icon: CircleHelp,
  },
  paused: {
    label: 'Paused',
    color: 'yellow',
    icon: Pause,
  },
  pausing: {
    label: 'Pausing',
    color: 'yellow',
    icon: Pause,
  },
  resuming: {
    label: 'Resuming',
    color: 'yellow',
    icon: RefreshCw,
  },
  starting: {
    label: 'Starting',
    color: 'yellow',
    icon: RefreshCw,
  },
  running: {
    label: 'Running',
    color: 'green',
    icon: CircleCheck,
  },
  failed: {
    label: 'Failed',
    color: 'red',
    icon: CircleX,
  },
  deleting: {
    label: 'Deleting',
    color: 'yellow',
    icon: Ban,
  },
  deleted: {
    label: 'Deleted',
    color: 'gray',
    icon: CircleX,
  },
  lost: {
    label: 'Lost',
    color: 'red',
    icon: TriangleAlert,
  },
};
