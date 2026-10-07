// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { Loading } from '@studio/components/Layouts/Loading';
import { ROUTES } from '@studio/constants/routes';
import { PluginProvider } from '@studio/plugins/PluginProvider';
import { PluginRenderer } from '@studio/plugins/PluginRenderer';
import {
  agentRoutes,
  anonymizerRoutes,
  baseModelsRoutes,
  customizationRoutes,
  dashboardRoutes,
  dataDesignerRoutes,
  deploymentRoutes,
  evaluationRoutes,
  experimentRoutes,
  filesetRoutes,
  guardrailsRoutes,
  inferenceProviderRoutes,
  virtualModelsRoutes,
  intakeRoutes,
  jobRoutes,
  memberRoutes,
  modelCompareRoutes,
  optimizerRoutes,
  safeSynthesizerRoutes,
  secretsRoutes,
  settingsRoutes,
} from '@studio/routes/groups';
import { PageLayout } from '@studio/routes/PageLayout';
import { RootLayout } from '@studio/routes/RootLayout';
import { RootRedirect } from '@studio/routes/RootRedirect';
import { lazy, Suspense } from 'react';
import { Outlet, type RouteObject } from 'react-router';

const NoMatchRoute = lazy(() =>
  import('@studio/routes/NoMatchRoute').then((module) => ({ default: module.NoMatchRoute }))
);
const AuthSuccessRoute = lazy(() =>
  import('@studio/routes/AuthSuccessRoute').then((m) => ({
    default: m.AuthSuccessRoute,
  }))
);
const WorkspaceIndexRoute = lazy(() =>
  import('@studio/routes/WorkspaceIndexRoute').then((module) => ({
    default: module.WorkspaceIndexRoute,
  }))
);
const WorkspaceSideNav = lazy(() =>
  import('@studio/routes/WorkspaceLayout/WorkspaceSideNav').then((module) => ({
    default: module.WorkspaceSideNav,
  }))
);

export const routes: RouteObject[] = [
  {
    path: '/health',
    element: <>OK</>,
  },
  {
    element: <RootLayout />,
    errorElement: <ErrorMessage height="100vh" />,
    children: [
      {
        path: ROUTES.auth.success,
        element: <AuthSuccessRoute />,
      },
      {
        element: <PageLayout />,
        children: [
          {
            path: '/',
            element: <RootRedirect />,
          },
          {
            path: '/workspaces',
            element: <RootRedirect />,
          },
          {
            path: '*',
            element: <NoMatchRoute />,
          },
        ],
      },
      {
        path: ROUTES.workspace.index,
        element: (
          <PluginProvider>
            <PageLayout sideNav={(collapsed) => <WorkspaceSideNav collapsed={collapsed} />} />
          </PluginProvider>
        ),
        children: [
          {
            path: ROUTES.workspace.index,
            element: <WorkspaceIndexRoute />,
          },
          {
            element: (
              // Suspense queries will show loader in panel area
              <Suspense fallback={<Loading description="Loading..." />}>
                <Outlet />
              </Suspense>
            ),
            errorElement: <RouteErrorPanel title="Entity Store" />,
            children: [
              ...dashboardRoutes,
              ...baseModelsRoutes,
              ...filesetRoutes,
              ...secretsRoutes,
              ...guardrailsRoutes,
              ...inferenceProviderRoutes,
              ...virtualModelsRoutes,
              ...deploymentRoutes,
              ...evaluationRoutes,
              ...experimentRoutes,
              ...customizationRoutes,
              ...jobRoutes,
              ...intakeRoutes,
              ...optimizerRoutes,
              ...safeSynthesizerRoutes,
              ...dataDesignerRoutes,
              ...anonymizerRoutes,
              ...agentRoutes,
              {
                // The /* suffix allows the plugin to own sub-paths via its own internal router.
                path: `${ROUTES.workspace.plugin}/*`,
                element: <PluginRenderer />,
                errorElement: <RouteErrorPanel title="Plugin" />,
              },
              ...settingsRoutes,
              ...modelCompareRoutes,
              ...memberRoutes,
            ],
          },
        ],
      },
      {
        path: '*',
        element: <NoMatchRoute />,
      },
    ],
  },
];
