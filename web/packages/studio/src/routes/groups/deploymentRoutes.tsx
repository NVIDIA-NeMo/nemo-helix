// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { DEPLOYMENTS_ENABLED } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { iconColorClass } from '@studio/routes/constants';
import { gateDeploymentsRoutes, getWorkspaceDeploymentsRoute } from '@studio/routes/utils';
import { lazy } from 'react';
import type { RouteObject } from 'react-router';

const DeploymentsListRoute =
  DEPLOYMENTS_ENABLED &&
  lazy(() =>
    import('@studio/routes/DeploymentsListRoute').then((module) => ({
      default: module.DeploymentsListRoute,
    }))
  );

const NewDeploymentRoute =
  DEPLOYMENTS_ENABLED &&
  lazy(() =>
    import('@studio/routes/NewDeploymentRoute').then((module) => ({
      default: module.NewDeploymentRoute,
    }))
  );

export const deploymentRoutes: RouteObject[] = gateDeploymentsRoutes([
  {
    path: ROUTES.workspace.deployments,
    element: DeploymentsListRoute ? <DeploymentsListRoute /> : null,
    errorElement: <RouteErrorPanel title="Deployments" />,
  },
  {
    path: ROUTES.workspace.deploymentsNew,
    element: NewDeploymentRoute ? <NewDeploymentRoute /> : null,
    errorElement: <RouteErrorPanel title="Create Deployment" />,
  },
  {
    path: ROUTES.workspace.deploymentsDeployment,
    element: DeploymentsListRoute ? <DeploymentsListRoute /> : null,
    errorElement: <RouteErrorPanel title="Deployments" />,
  },
]);

const NavIcon = ENTITY_ICONS.deployments;

export const getDeploymentSideNavItems = (workspace: string) =>
  DEPLOYMENTS_ENABLED
    ? [
        {
          id: 'deployments',
          slotIcon: <NavIcon className={iconColorClass} />,
          slotLabel: 'Deployments',
          href: getWorkspaceDeploymentsRoute(workspace),
        },
      ]
    : [];
