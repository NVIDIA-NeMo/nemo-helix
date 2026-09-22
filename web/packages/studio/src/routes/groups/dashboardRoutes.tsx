// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import {
  ASSISTANT_STUDIO_ENABLED,
  DASHBOARD_ROUTE_ENABLED,
  DASHBOARD_SANDBOX_ENABLED,
} from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { iconColorClass } from '@studio/routes/constants';
import {
  gateAssistantStudioRoutes,
  gateDashboardRoutes,
  getWorkspaceDashboardRoute,
} from '@studio/routes/utils';
import { LayoutDashboard } from 'lucide-react';
import { lazy, type ReactNode } from 'react';
import type { RouteObject } from 'react-router';

const DashboardLandingRoute = lazy(() =>
  import('@studio/routes/DashboardLandingRoute').then((module) => ({
    default: module.DashboardLandingRoute,
  }))
);
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
const AssistantChatRoute = lazy(() =>
  import('@studio/routes/agents/AssistantChatRoute').then((module) => ({
    default: module.AssistantChatRoute,
  }))
);

/**
 * Picks which page renders at `/workspaces/:workspace/dashboard`; DASHBOARD_SANDBOX_ENABLED wins
 * over ASSISTANT_STUDIO_ENABLED since the new dashboard replaces every prior landing.
 */
const getDashboardElement = (): ReactNode => {
  if (DASHBOARD_SANDBOX_ENABLED) return <WorkspaceDashboardHomeRoute />;
  if (ASSISTANT_STUDIO_ENABLED) return <DashboardLandingRoute />;
  return <WorkspaceDashboardRoute />;
};

export const dashboardRoutes: RouteObject[] = gateDashboardRoutes([
  {
    path: ROUTES.workspace.dashboard,
    element: getDashboardElement(),
    errorElement: <RouteErrorPanel title="Workspace" />,
  },
  ...gateAssistantStudioRoutes([
    {
      path: ROUTES.workspace.assistantChat,
      element: <AssistantChatRoute />,
      errorElement: <RouteErrorPanel title="NeMo Assistant" />,
    },
  ]),
]);

export const getDashboardSideNavItems = (workspace: string) =>
  DASHBOARD_ROUTE_ENABLED
    ? [
        {
          id: 'dashboard',
          slotIcon: <LayoutDashboard className={iconColorClass} />,
          slotLabel: 'Dashboard',
          href: getWorkspaceDashboardRoute(workspace),
        },
      ]
    : [];
