// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { ROUTES } from '@studio/constants/routes';
import { lazy } from 'react';
import type { RouteObject } from 'react-router';

const NotificationsRoute = lazy(() =>
  import('@studio/prototypes/notifications/notif/NotificationsRoute').then((module) => ({
    default: module.NotificationsRoute,
  }))
);

export const notificationRoutes: RouteObject[] = [
  {
    path: ROUTES.workspace.notifications,
    element: <NotificationsRoute />,
    errorElement: <RouteErrorPanel title="Notifications" />,
  },
];
