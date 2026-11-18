// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NavId } from '@studio/prototypes/notifications/fixtures';
import type { NotificationLink } from '@studio/prototypes/notifications/toasts/types';
import type { NavNotification, NotificationSeverity } from '@studio/prototypes/notifications/types';
import type { ReactNode } from 'react';

export interface InboxItem {
  readonly id: string;
  readonly severity: NotificationSeverity;
  readonly title: string;
  readonly description?: ReactNode;
  readonly sourceId: NavId;
  readonly createdAt: string;
  readonly link?: NotificationLink;
  readonly read: boolean;
}

export type InboxGroupLabel = 'Today' | 'Yesterday' | 'Earlier';

export interface InboxGroup {
  readonly label: InboxGroupLabel;
  readonly items: InboxItem[];
}

export interface InboxState {
  items: InboxItem[];
  unread: NavNotification | undefined;
  add: (item: InboxItem) => void;
  markRead: (id: string) => void;
  markAllRead: () => void;
  dismiss: (id: string) => void;
}
