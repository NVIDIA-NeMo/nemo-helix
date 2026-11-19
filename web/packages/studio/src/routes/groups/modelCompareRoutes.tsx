// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { Play } from '@nemo/icons/icons';
import { ROUTES } from '@studio/constants/routes';
import { iconColorClass } from '@studio/routes/constants';
import { getModelCompareRoute } from '@studio/routes/utils';
import { lazy } from 'react';
import type { RouteObject } from 'react-router';

const ModelCompareRoute = lazy(() =>
  import('@studio/routes/ModelCompareRoute').then((module) => ({
    default: module.ModelCompareRoute,
  }))
);

export const modelCompareRoutes: RouteObject[] = [
  {
    path: ROUTES.workspace.modelCompare,
    element: <ModelCompareRoute />,
    errorElement: <RouteErrorPanel title="Chat" />,
  },
];

export const getModelCompareSideNavItems = (workspace: string) => [
  {
    id: 'playground',
    slotIcon: <Play className={iconColorClass} />,
    slotLabel: 'Playground',
    href: getModelCompareRoute(workspace),
  },
];
