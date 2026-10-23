// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Library } from '@nemo/icons/icons';
import { ROUTES } from '@studio/constants/routes';
import { iconColorClass } from '@studio/routes/constants';
import { getWorkspaceBaseModelsRoute } from '@studio/routes/utils';
import { lazy } from 'react';
import type { RouteObject } from 'react-router';

const WorkspaceBaseModelsRoute = lazy(() =>
  import('@studio/routes/WorkspaceBaseModelsRoute').then((module) => ({
    default: module.WorkspaceBaseModelsRoute,
  }))
);

export const baseModelsRoutes: RouteObject[] = [
  {
    element: <WorkspaceBaseModelsRoute />,
    children: [
      { path: ROUTES.workspace.baseModels, element: null },
      { path: ROUTES.workspace.baseModelsModel, element: null },
    ],
  },
];

export const getBaseModelsSideNavItems = (workspace: string) => [
  {
    id: 'base-models',
    slotIcon: <Library className={iconColorClass} />,
    slotLabel: 'Model Catalog',
    href: getWorkspaceBaseModelsRoute(workspace),
  },
];
