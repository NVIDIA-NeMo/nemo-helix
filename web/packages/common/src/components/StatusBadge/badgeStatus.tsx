// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  CheckCircle,
  CloseCircle,
  Refresh,
  Cancel,
  HelpCircle,
  Warning,
  Pause,
} from '@nemo/icons/icons';
import { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { BadgeProps as KuiBadgeProps } from '@nvidia/foundations-react-core';
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
    icon: CheckCircle,
  },
  completed: {
    label: 'Completed',
    color: 'green',
    icon: CheckCircle,
  },
  error: {
    label: 'Error',
    color: 'red',
    icon: CloseCircle,
  },
  queued: {
    label: 'Queued',
    color: 'gray',
    icon: Refresh,
  },
  pending: {
    label: 'Pending',
    color: 'gray',
    icon: Refresh,
  },
  active: {
    label: 'Active',
    color: 'blue',
    icon: Refresh,
  },
  in_progress: {
    label: 'In Progress',
    color: 'blue',
    icon: Refresh,
  },
  cancelling: {
    label: 'Cancelling',
    color: 'yellow',
    icon: Cancel,
  },
  cancelled: {
    label: 'Cancelled',
    color: 'yellow',
    icon: Cancel,
  },
  unavailable: {
    label: 'Unavailable',
    color: 'gray',
    icon: Warning,
  },
  default: {
    label: 'Unknown',
    color: 'purple',
    icon: HelpCircle,
  },
  // What does it even mean for the job to be "ready" to be used..
  ready: {
    label: 'Ready',
    color: 'green',
    icon: CheckCircle,
  },
  unknown: {
    label: 'Unknown',
    color: 'purple',
    icon: HelpCircle,
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
    icon: Refresh,
  },
  starting: {
    label: 'Starting',
    color: 'yellow',
    icon: Refresh,
  },
  running: {
    label: 'Running',
    color: 'green',
    icon: CheckCircle,
  },
  failed: {
    label: 'Failed',
    color: 'red',
    icon: CloseCircle,
  },
  deleting: {
    label: 'Deleting',
    color: 'yellow',
    icon: Cancel,
  },
  deleted: {
    label: 'Deleted',
    color: 'gray',
    icon: CloseCircle,
  },
  lost: {
    label: 'Lost',
    color: 'red',
    icon: Warning,
  },
};
