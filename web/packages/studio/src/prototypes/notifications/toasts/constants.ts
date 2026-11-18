// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NotificationSeverity } from '@studio/prototypes/notifications/types';

export const NOTIFICATION_DISMISS_MS: Record<NotificationSeverity, number | false> = {
  info: 6000,
  success: 6000,
  warning: 10000,
  error: false,
};

export const BRIEF_TOAST_DISMISS_MS = 3000;

export const MAX_VISIBLE_NOTIFICATIONS = 3;
