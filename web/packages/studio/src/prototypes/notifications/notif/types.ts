// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NotificationSeverity } from '@studio/prototypes/notifications/types';

export interface NotifEvent {
  readonly id: string;
  readonly type: string;
  readonly workspace: string;
  readonly title: string;
  readonly message?: string | null;
  readonly severity: NotificationSeverity;
  readonly link?: string | null;
  readonly data: Record<string, unknown>;
  readonly created_at: string | null;
  readonly created_by?: string | null;
  readonly read: boolean;
}

export interface NotifEventList {
  readonly data: NotifEvent[];
  readonly unread_count: number;
}

export interface NotifListParams {
  readonly limit: number;
  readonly unreadOnly: boolean;
}
