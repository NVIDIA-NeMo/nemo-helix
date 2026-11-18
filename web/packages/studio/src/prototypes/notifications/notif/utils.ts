// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { InboxItem } from '@studio/prototypes/notifications/inbox/types';
import type { NotifEvent } from '@studio/prototypes/notifications/notif/types';
import type { NotifyOptions } from '@studio/prototypes/notifications/toasts/types';
import type { NotificationSeverity } from '@studio/prototypes/notifications/types';

export const NOTIF_PLUGIN_NAME = 'notif';

export const notifStreamPath = (workspace: string): string =>
  `/apis/notif/v2/workspaces/${encodeURIComponent(workspace)}/events/stream`;

const SEVERITIES: ReadonlySet<string> = new Set<NotificationSeverity>([
  'info',
  'success',
  'warning',
  'error',
]);

const SOURCE_LABELS: Record<string, string> = {
  agents: 'Agents',
  customization: 'Fine-tuning',
  customizer: 'Fine-tuning',
  'data-designer': 'Data Designer',
  datasets: 'Datasets',
  deployments: 'Deployments',
  evals: 'Evaluations',
  evaluator: 'Evaluations',
  files: 'Filesets',
  guardrails: 'Guardrails',
  jobs: 'Jobs',
  models: 'Models',
};

const DEFAULT_LINK_LABEL = 'View details';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export const parseNotifEvent = (raw: string): NotifEvent | undefined => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return undefined;
  }
  if (
    !isRecord(parsed) ||
    typeof parsed.id !== 'string' ||
    typeof parsed.type !== 'string' ||
    typeof parsed.title !== 'string' ||
    typeof parsed.created_at !== 'string' ||
    typeof parsed.severity !== 'string' ||
    !SEVERITIES.has(parsed.severity)
  ) {
    return undefined;
  }
  return {
    ...parsed,
    data: isRecord(parsed.data) ? parsed.data : {},
  } as NotifEvent;
};

const sourceLabelFor = (type: string): string => {
  const prefix = type.split('.')[0];
  const label = SOURCE_LABELS[prefix];
  if (label) {
    return label;
  }
  const words = prefix.replace(/[-_]/g, ' ');
  return words.charAt(0).toUpperCase() + words.slice(1);
};

const linkLabelFor = (data: Record<string, unknown>): string =>
  typeof data.link_label === 'string' && data.link_label.length > 0
    ? data.link_label
    : DEFAULT_LINK_LABEL;

export const notifEventToInboxItem = (event: NotifEvent): InboxItem => ({
  id: event.id,
  severity: event.severity,
  title: event.title,
  description: event.message ?? undefined,
  source: sourceLabelFor(event.type),
  createdAt: event.created_at,
  link: event.link ? { label: linkLabelFor(event.data), href: event.link } : undefined,
});

export const inboxItemToNotifyOptions = (item: InboxItem): NotifyOptions => ({
  status: item.severity,
  heading: item.title,
  description: item.description,
  link: item.link,
});
