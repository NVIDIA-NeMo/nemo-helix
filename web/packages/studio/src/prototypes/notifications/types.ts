// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NavItem, NavSubItem } from '@studio/components/Layouts/NavigationDrawer/types';

export type NotificationSeverity = 'info' | 'success' | 'warning' | 'error';

export interface NavNotification {
  readonly count: number;
  readonly severity: NotificationSeverity;
}

export interface ProtoNavSubItem extends NavSubItem {
  notification?: NavNotification;
}

export interface ProtoNavItem extends Omit<NavItem, 'subItems'> {
  notification?: NavNotification;
  subItems?: ProtoNavSubItem[];
}

export interface ProtoNavGroupInput {
  group?: string;
  items: ProtoNavItem[];
}

export type ProtoNavInputItem = ProtoNavItem | ProtoNavGroupInput;

export interface ProtoNavGroup {
  groupLabel?: string;
  items: ProtoNavItem[];
}
