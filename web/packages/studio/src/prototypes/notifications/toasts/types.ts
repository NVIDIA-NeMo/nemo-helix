// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NotificationSeverity } from '@studio/prototypes/notifications/types';
import type { ReactNode } from 'react';

export interface NotificationLink {
  label: string;
  href: string;
}

export interface NotificationAction {
  label: string;
  onAction: () => void | Promise<void>;
}

export interface NotificationDecision {
  accept: NotificationAction;
  decline: NotificationAction;
}

export interface NotifyOptions {
  status: NotificationSeverity;
  heading: string;
  description?: ReactNode;
  link?: NotificationLink;
  decision?: NotificationDecision;
  durationMs?: number | false;
  sourceId?: string;
}

export interface BriefToastOptions {
  status: NotificationSeverity;
  message: string;
  durationMs?: number | false;
}

export type NotificationEntry =
  | ({ id: string; kind: 'notification' } & NotifyOptions)
  | ({ id: string; kind: 'toast' } & BriefToastOptions);

export interface ProtoNotificationsContextValue {
  notify: (options: NotifyOptions) => string;
  toast: (options: BriefToastOptions) => string;
  dismiss: (id: string) => void;
}
