// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { MODEL_DETAILS_ENABLED } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { iconColorClass } from '@studio/routes/constants';
import { gateModelDetailsRoutes, getWorkspaceBaseModelsRoute } from '@studio/routes/utils';
import { lazy } from 'react';
import type { RouteObject } from 'react-router';

const BaseModelsIcon = ENTITY_ICONS.baseModels;

const WorkspaceBaseModelsRoute = lazy(() =>
  import('@studio/routes/WorkspaceBaseModelsRoute').then((module) => ({
    default: module.WorkspaceBaseModelsRoute,
  }))
);
const ModelDetailRoute = lazy(() =>
  import('@studio/routes/ModelDetailRoute').then((module) => ({
    default: module.ModelDetailRoute,
  }))
);

export const baseModelsRoutes: RouteObject[] = [
  {
    element: <WorkspaceBaseModelsRoute />,
    children: [
      { path: ROUTES.workspace.baseModels, element: null },
      ...(MODEL_DETAILS_ENABLED ? [] : [{ path: ROUTES.workspace.baseModelsModel, element: null }]),
    ],
  },
  ...gateModelDetailsRoutes({
    path: ROUTES.workspace.baseModelsModel,
    element: <ModelDetailRoute parent="catalog" />,
    errorElement: <RouteErrorPanel title="Model" />,
  }),
];

export const getBaseModelsSideNavItems = (workspace: string) => [
  {
    id: 'base-models',
    slotIcon: <BaseModelsIcon className={iconColorClass} />,
    slotLabel: 'Model Catalog',
    href: getWorkspaceBaseModelsRoute(workspace),
  },
];
