// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatusIndicator } from '@nvidia/foundations-react-core';
import type { NavNotification } from '@studio/prototypes/notifications/types';
import {
  SEVERITY_COLOR,
  describeNotification,
  hasNotification,
} from '@studio/prototypes/notifications/utils';
import type { FC, ReactNode } from 'react';

interface NavNotificationDotProps {
  notification?: NavNotification;
  children: ReactNode;
}

export const NavNotificationDot: FC<NavNotificationDotProps> = ({ notification, children }) => (
  <span className="relative inline-flex">
    {children}
    {hasNotification(notification) && (
      <>
        <StatusIndicator
          size="small"
          color={SEVERITY_COLOR[notification.severity]}
          className="absolute -right-1 -top-1"
          data-testid="nav-notification-dot"
        />
        <span className="sr-only">{describeNotification(notification)}</span>
      </>
    )}
  </span>
);
