// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { InboxItem } from '@studio/prototypes/notifications/inbox/types';
import type { NotifEvent } from '@studio/prototypes/notifications/notif/types';
import type { NotifyOptions } from '@studio/prototypes/notifications/toasts/types';
import type { NotificationSeverity } from '@studio/prototypes/notifications/types';

export const NOTIF_PLUGIN_NAME = 'notif';

export const notifEventsPath = (workspace: string): string =>
  `/apis/notif/v2/workspaces/${encodeURIComponent(workspace)}/events`;

export const notifStreamPath = (workspace: string): string =>
  `${notifEventsPath(workspace)}/stream`;

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
  openshell: 'OpenShell',
};

const DEFAULT_LINK_LABEL = 'View details';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export const toNotifEvent = (value: unknown): NotifEvent | undefined => {
  if (
    !isRecord(value) ||
    typeof value.id !== 'string' ||
    typeof value.type !== 'string' ||
    typeof value.title !== 'string' ||
    typeof value.severity !== 'string' ||
    !SEVERITIES.has(value.severity)
  ) {
    return undefined;
  }
  return {
    ...value,
    created_at: typeof value.created_at === 'string' ? value.created_at : null,
    read: value.read === true,
    data: isRecord(value.data) ? value.data : {},
  } as NotifEvent;
};

export const parseNotifEvent = (raw: string): NotifEvent | undefined => {
  try {
    return toNotifEvent(JSON.parse(raw));
  } catch {
    return undefined;
  }
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
  createdAt: event.created_at ?? new Date().toISOString(),
  link: event.link ? { label: linkLabelFor(event.data), href: event.link } : undefined,
  read: event.read,
});

export const inboxItemToNotifyOptions = (item: InboxItem): NotifyOptions => ({
  status: item.severity,
  heading: item.title,
  description: item.description,
  link: item.link,
});
