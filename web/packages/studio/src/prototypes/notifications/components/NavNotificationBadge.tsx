// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge } from '@nvidia/foundations-react-core';
import type { NavNotification } from '@studio/prototypes/notifications/types';
import {
  SEVERITY_COLOR,
  describeNotification,
  formatNotificationCount,
  hasNotification,
} from '@studio/prototypes/notifications/utils';
import cn from 'classnames';
import type { FC } from 'react';

interface NavNotificationBadgeProps {
  notification?: NavNotification;
  className?: string;
}

export const NavNotificationBadge: FC<NavNotificationBadgeProps> = ({
  notification,
  className,
}) => {
  if (!hasNotification(notification)) {
    return null;
  }

  return (
    <Badge
      kind="solid"
      size="small"
      color={SEVERITY_COLOR[notification.severity]}
      className={cn('ml-auto shrink-0 tabular-nums', className)}
      data-testid="nav-notification-badge"
    >
      <span aria-hidden>{formatNotificationCount(notification.count)}</span>
      <span className="sr-only">{describeNotification(notification)}</span>
    </Badge>
  );
};
