// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NAV_LABELS, type NavId } from '@studio/prototypes/notifications/fixtures';
import type {
  InboxGroup,
  InboxGroupLabel,
  InboxItem,
} from '@studio/prototypes/notifications/inbox/types';
import type { NotifyOptions } from '@studio/prototypes/notifications/toasts/types';
import type { NavNotification, NotificationSeverity } from '@studio/prototypes/notifications/types';
import { rollupNotifications } from '@studio/prototypes/notifications/utils';

const MINUTE_MS = 60_000;
const DAY_MS = 24 * 60 * MINUTE_MS;

export const SEVERITY_LABEL: Record<NotificationSeverity, string> = {
  info: 'Info',
  success: 'Success',
  warning: 'Warning',
  error: 'Error',
};

const startOfDay = (date: Date): number =>
  new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();

const groupLabelFor = (createdAt: string, today: number): InboxGroupLabel => {
  const day = startOfDay(new Date(createdAt));
  if (day >= today) {
    return 'Today';
  }
  if (day >= today - DAY_MS) {
    return 'Yesterday';
  }
  return 'Earlier';
};

export const groupInboxItems = (items: InboxItem[], now: Date = new Date()): InboxGroup[] => {
  const today = startOfDay(now);
  return items.reduce<InboxGroup[]>((groups, item) => {
    const label = groupLabelFor(item.createdAt, today);
    const last = groups.at(-1);
    if (last?.label === label) {
      return [...groups.slice(0, -1), { label, items: [...last.items, item] }];
    }
    return [...groups, { label, items: [item] }];
  }, []);
};

export const summarizeInbox = (items: InboxItem[]): NavNotification | undefined =>
  rollupNotifications(items.map((item) => ({ count: 1, severity: item.severity })));

interface ToInboxItemOptions {
  minutesAgo?: number;
}

export const toInboxItem = (
  options: NotifyOptions,
  sourceId: NavId,
  { minutesAgo = 0 }: ToInboxItemOptions = {}
): InboxItem => ({
  id: `inbox-${crypto.randomUUID()}`,
  severity: options.status,
  title: options.heading,
  description: options.description,
  link: options.link,
  source: NAV_LABELS[sourceId],
  createdAt: new Date(Date.now() - minutesAgo * MINUTE_MS).toISOString(),
});
