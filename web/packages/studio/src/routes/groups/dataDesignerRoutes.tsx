// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { ROUTES } from '@studio/constants/routes';
import { iconColorClass } from '@studio/routes/constants';
import { getDataDesignerJobListRoute } from '@studio/routes/utils';
import { lazy } from 'react';
import type { RouteObject } from 'react-router';

const DataDesignerJobListRoute = lazy(() =>
  import('@studio/routes/DataDesignerJobListRoute').then((m) => ({
    default: m.DataDesignerJobListRoute,
  }))
);
const DataDesignerJobDetailsRoute = lazy(() =>
  import('@studio/routes/DataDesignerJobDetailsRoute').then((m) => ({
    default: m.DataDesignerJobDetailsRoute,
  }))
);
const NewDataDesignerJobRoute = lazy(() =>
  import('@studio/routes/NewDataDesignerJobRoute').then((m) => ({
    default: m.NewDataDesignerJobRoute,
  }))
);
const DataDesignerJobBuildRoute = lazy(() =>
  import('@studio/routes/DataDesignerJobBuildRoute').then((m) => ({
    default: m.DataDesignerJobBuildRoute,
  }))
);
const LegacyNewDataDesignerJobRoute = lazy(() =>
  import('@studio/routes/LegacyNewDataDesignerJobRoute').then((m) => ({
    default: m.LegacyNewDataDesignerJobRoute,
  }))
);

export const dataDesignerRoutes: RouteObject[] = [
  {
    path: ROUTES.workspace.dataDesignerJobList,
    element: <DataDesignerJobListRoute />,
    errorElement: <RouteErrorPanel title="Data Designer" />,
  },
  {
    path: ROUTES.workspace.dataDesignerJobDetails,
    element: <DataDesignerJobDetailsRoute />,
    errorElement: <RouteErrorPanel title="Data Designer" />,
  },
  {
    path: ROUTES.workspace.dataDesignerJobNew,
    element: <NewDataDesignerJobRoute />,
    errorElement: <RouteErrorPanel title="Data Designer" />,
  },
  {
    path: ROUTES.workspace.dataDesignerJobBuild,
    element: <DataDesignerJobBuildRoute />,
    errorElement: <RouteErrorPanel title="Data Designer" />,
  },
  {
    path: ROUTES.workspace.dataDesignerJobNewLegacy,
    element: <LegacyNewDataDesignerJobRoute />,
    errorElement: <RouteErrorPanel title="Data Designer" />,
  },
];

const NavIcon = ENTITY_ICONS.dataDesignerJobs;

export const getDataDesignerSideNavItems = (workspace: string) => [
  {
    id: 'data-designer',
    slotIcon: <NavIcon className={iconColorClass} />,
    slotLabel: 'Data Designer',
    href: getDataDesignerJobListRoute(workspace),
  },
];
