// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ProtoNotificationsContextValue } from '@studio/prototypes/notifications/toasts/types';
import { createContext, useContext } from 'react';

export const ProtoNotificationsContext = createContext<ProtoNotificationsContextValue | null>(null);

export const useProtoNotifications = (): ProtoNotificationsContextValue => {
  const context = useContext(ProtoNotificationsContext);
  if (!context) {
    throw new Error('useProtoNotifications must be used within a ProtoNotificationProvider');
  }
  return context;
};
