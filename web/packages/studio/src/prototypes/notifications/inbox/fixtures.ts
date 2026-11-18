// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NAV_HREFS } from '@studio/prototypes/notifications/fixtures';
import type { InboxItem } from '@studio/prototypes/notifications/inbox/types';
import { toInboxItem } from '@studio/prototypes/notifications/inbox/utils';
import {
  DATASET_UPLOADED,
  EVALUATION_FAILED,
  GUARDRAIL_DEGRADED,
  JOB_COMPLETED,
} from '@studio/prototypes/notifications/toasts/fixtures';

const HOUR = 60;
const DAY = 24 * HOUR;

export const SAMPLE_INBOX: InboxItem[] = [
  toInboxItem(EVALUATION_FAILED, 'evaluation-results', { minutesAgo: 4 }),
  toInboxItem(
    {
      status: 'info',
      heading: 'Routing suggestion for support-agent',
      description: 'Route 60% of traffic to nemotron-nano-9b. Projected cost −38%.',
      link: { label: 'Review', href: `${NAV_HREFS['agents-group']}/support-agent?tab=suggestions` },
    },
    'agents-group',
    { minutesAgo: 38 }
  ),
  toInboxItem(JOB_COMPLETED, 'custom-models', { minutesAgo: 2 * HOUR + 10 }),
  toInboxItem(GUARDRAIL_DEGRADED, 'guardrails', { minutesAgo: 5 * HOUR, read: true }),
  toInboxItem(DATASET_UPLOADED, 'dataset-list', { minutesAgo: DAY + 2 * HOUR, read: true }),
  toInboxItem(
    {
      status: 'success',
      heading: 'Synthetic data generation finished',
      description: '128 records generated for support-intents.',
      link: { label: 'View job', href: `${NAV_HREFS['data-designer']}/support-intents` },
    },
    'data-designer',
    { minutesAgo: DAY + 5 * HOUR, read: true }
  ),
  toInboxItem(
    {
      status: 'error',
      heading: 'Deployment failed',
      description: 'nemotron-nano-9b could not be scheduled: no GPUs available.',
      link: { label: 'View deployment', href: `${NAV_HREFS.deployments}/nemotron-nano-9b` },
    },
    'deployments',
    { minutesAgo: 3 * DAY, read: true }
  ),
];
