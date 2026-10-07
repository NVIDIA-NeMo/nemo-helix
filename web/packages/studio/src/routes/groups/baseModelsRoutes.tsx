// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTES } from '@studio/constants/routes';
import { iconColorClass } from '@studio/routes/constants';
import { getWorkspaceBaseModelsRoute } from '@studio/routes/utils';
import { LibraryBig } from 'lucide-react';
import { lazy } from 'react';
import type { RouteObject } from 'react-router';

const WorkspaceBaseModelsRoute = lazy(() =>
  import('@studio/routes/WorkspaceBaseModelsRoute').then((module) => ({
    default: module.WorkspaceBaseModelsRoute,
  }))
);

export const baseModelsRoutes: RouteObject[] = [
  {
    path: ROUTES.workspace.baseModels,
    element: <WorkspaceBaseModelsRoute />,
  },
  {
    path: ROUTES.workspace.baseModelsModel,
    element: <WorkspaceBaseModelsRoute />,
  },
];

export const getBaseModelsSideNavItems = (workspace: string) => [
  {
    id: 'base-models',
    slotIcon: <LibraryBig className={iconColorClass} />,
    slotLabel: 'Model Catalog',
    href: getWorkspaceBaseModelsRoute(workspace),
  },
];
