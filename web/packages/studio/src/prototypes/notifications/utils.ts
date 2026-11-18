// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  NavNotification,
  NotificationSeverity,
  ProtoNavGroup,
  ProtoNavGroupInput,
  ProtoNavInputItem,
  ProtoNavItem,
} from '@studio/prototypes/notifications/types';

export const MAX_BADGE_COUNT = 99;

export const SEVERITY_COLOR = {
  info: 'blue',
  success: 'green',
  warning: 'yellow',
  error: 'red',
} as const satisfies Record<NotificationSeverity, 'blue' | 'green' | 'yellow' | 'red'>;

const SEVERITY_RANK: Record<NotificationSeverity, number> = {
  info: 0,
  success: 1,
  warning: 2,
  error: 3,
};

export const hasNotification = (
  notification: NavNotification | undefined
): notification is NavNotification => notification !== undefined && notification.count > 0;

export const formatNotificationCount = (count: number): string =>
  count > MAX_BADGE_COUNT ? `${MAX_BADGE_COUNT}+` : String(count);

export const describeNotification = ({ count }: NavNotification): string =>
  `${count} unread notification${count === 1 ? '' : 's'}`;

export const rollupNotifications = (
  notifications: (NavNotification | undefined)[]
): NavNotification | undefined =>
  notifications.filter(hasNotification).reduce<NavNotification | undefined>(
    (total, next) =>
      total === undefined
        ? next
        : {
            count: total.count + next.count,
            severity:
              SEVERITY_RANK[next.severity] > SEVERITY_RANK[total.severity]
                ? next.severity
                : total.severity,
          },
    undefined
  );

const isGroup = (item: ProtoNavInputItem): item is ProtoNavGroupInput =>
  Array.isArray((item as { items?: unknown }).items);

export const toProtoGroups = (items: ProtoNavInputItem[]): ProtoNavGroup[] =>
  items.map((entry) =>
    isGroup(entry) ? { groupLabel: entry.group, items: entry.items } : { items: [entry] }
  );

export const flattenProtoForRail = (items: ProtoNavItem[]): ProtoNavItem[] =>
  items.flatMap((item) => [
    ...(item.href !== undefined ? [{ ...item, subItems: undefined }] : []),
    ...(item.subItems ?? []).filter((sub) => sub.href !== undefined),
  ]);
