// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { ROUTES } from '@studio/constants/routes';
import { iconColorClass } from '@studio/routes/constants';
import { LegacyFilesetRedirect } from '@studio/routes/FilesetDetailRoute/LegacyFilesetRedirect';
import { getWorkspaceFilesetsRoute } from '@studio/routes/utils';
import { lazy } from 'react';
import type { RouteObject } from 'react-router';

const FilesetNewRoute = lazy(() =>
  import('@studio/routes/FilesetNewRoute').then((module) => ({ default: module.FilesetNewRoute }))
);
const FilesetListRoute = lazy(() =>
  import('@studio/routes/FilesetListRoute').then((module) => ({ default: module.FilesetListRoute }))
);
const FilesetDetailRoute = lazy(() =>
  import('@studio/routes/FilesetDetailRoute').then((module) => ({
    default: module.FilesetDetailRoute,
  }))
);

export const filesetRoutes: RouteObject[] = [
  {
    path: ROUTES.workspace.filesets,
    element: <FilesetListRoute />,
    errorElement: <RouteErrorPanel title="Filesets" />,
    children: [
      {
        path: ROUTES.workspace.filesetNew,
        element: <FilesetNewRoute />,
      },
    ],
  },
  {
    path: ROUTES.workspace.filesetDetail,
    element: <FilesetDetailRoute />,
    errorElement: <RouteErrorPanel title="Fileset" />,
  },
  {
    path: ROUTES.workspace.filesetDetails,
    element: <LegacyFilesetRedirect />,
  },
  {
    path: ROUTES.workspace.filesetFile,
    element: <LegacyFilesetRedirect />,
  },
];

const NavIcon = ENTITY_ICONS.filesets;

export const getFilesetSideNavItems = (workspace: string) => [
  {
    id: 'datasets',
    slotIcon: <NavIcon className={iconColorClass} />,
    slotLabel: 'Filesets',
    href: getWorkspaceFilesetsRoute(workspace),
  },
];
