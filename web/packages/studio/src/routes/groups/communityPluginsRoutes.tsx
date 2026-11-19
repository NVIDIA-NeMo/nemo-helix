// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { ROUTES } from '@studio/constants/routes';
import { lazy } from 'react';
import type { RouteObject } from 'react-router';

const CommunityPluginsRoute = lazy(() =>
  import('@studio/routes/CommunityPluginsRoute').then((module) => ({
    default: module.CommunityPluginsRoute,
  }))
);

export const communityPluginsRoutes: RouteObject[] = [
  {
    path: ROUTES.workspace.communityPlugins,
    element: <CommunityPluginsRoute />,
    errorElement: <RouteErrorPanel title="Community Plugins" />,
  },
];
