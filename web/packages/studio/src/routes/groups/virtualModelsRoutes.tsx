// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { VIRTUAL_MODELS_ROUTE_ENABLED } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { iconColorClass } from '@studio/routes/constants';
import { gateVirtualModelsRoutes, getWorkspaceVirtualModelsRoute } from '@studio/routes/utils';
import { lazy } from 'react';
import { Navigate, type RouteObject } from 'react-router';

const VirtualModelsListRoute = lazy(() =>
  import('@studio/routes/VirtualModelsListRoute').then((module) => ({
    default: module.VirtualModelsListRoute,
  }))
);

const VirtualModelDetailRoute = lazy(() =>
  import('@studio/routes/virtualModels/VirtualModelDetailRoute').then((m) => ({
    default: m.VirtualModelDetailRoute,
  }))
);

const VirtualModelDetailsTab = lazy(() =>
  import('@studio/routes/virtualModels/VirtualModelDetailsTab').then((m) => ({
    default: m.VirtualModelDetailsTab,
  }))
);

const VirtualModelChatTab = lazy(() =>
  import('@studio/routes/virtualModels/VirtualModelChatTab').then((m) => ({
    default: m.VirtualModelChatTab,
  }))
);

export const virtualModelsRoutes: RouteObject[] = gateVirtualModelsRoutes([
  {
    path: ROUTES.workspace.virtualModels,
    element: <VirtualModelsListRoute />,
    errorElement: <RouteErrorPanel title="Virtual Models" />,
  },
  {
    path: ROUTES.workspace.virtualModelDetail,
    element: <VirtualModelDetailRoute />,
    errorElement: <RouteErrorPanel title="Virtual Models" />,
    children: [
      {
        index: true,
        element: <Navigate to="details" replace />,
      },
      {
        path: ROUTES.workspace.virtualModelDetails,
        element: <VirtualModelDetailsTab />,
      },
      {
        path: ROUTES.workspace.virtualModelChat,
        element: <VirtualModelChatTab />,
      },
    ],
  },
]);

const NavIcon = ENTITY_ICONS.virtualModels;

export const getVirtualModelsSideNavItems = (workspace: string) =>
  VIRTUAL_MODELS_ROUTE_ENABLED
    ? [
        {
          id: 'virtual-models',
          slotIcon: <NavIcon className={iconColorClass} />,
          slotLabel: 'Virtual Models',
          href: getWorkspaceVirtualModelsRoute(workspace),
        },
      ]
    : [];
