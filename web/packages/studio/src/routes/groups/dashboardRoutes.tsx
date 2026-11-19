// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { LayoutGrid } from '@nemo/icons/icons';
import { DASHBOARD_ROUTE_ENABLED, DASHBOARD_SANDBOX_ENABLED } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { iconColorClass } from '@studio/routes/constants';
import { gateDashboardRoutes, getWorkspaceDashboardRoute } from '@studio/routes/utils';
import { lazy, type ReactNode } from 'react';
import type { RouteObject } from 'react-router';

const WorkspaceDashboardRoute = lazy(() =>
  import('@studio/routes/WorkspaceDashboardRoute').then((module) => ({
    default: module.WorkspaceDashboardRoute,
  }))
);
const WorkspaceDashboardHomeRoute = lazy(() =>
  import('@studio/routes/WorkspaceDashboardHomeRoute').then((module) => ({
    default: module.WorkspaceDashboardHomeRoute,
  }))
);

/**
 * Picks which page renders at `/workspaces/:workspace/dashboard`. The assistant is the top-bar
 * pop-out only (there is no full-page assistant route), so ASSISTANT_STUDIO_ENABLED does not
 * change the dashboard page.
 */
const getDashboardElement = (): ReactNode => {
  if (DASHBOARD_SANDBOX_ENABLED) return <WorkspaceDashboardHomeRoute />;
  return <WorkspaceDashboardRoute />;
};

export const dashboardRoutes: RouteObject[] = gateDashboardRoutes([
  {
    path: ROUTES.workspace.dashboard,
    element: getDashboardElement(),
    errorElement: <RouteErrorPanel title="Workspace" />,
  },
]);

export const getDashboardSideNavItems = (workspace: string) =>
  DASHBOARD_ROUTE_ENABLED
    ? [
        {
          id: 'dashboard',
          slotIcon: <LayoutGrid className={iconColorClass} />,
          slotLabel: 'Dashboard',
          href: getWorkspaceDashboardRoute(workspace),
        },
      ]
    : [];
