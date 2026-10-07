// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { ROUTES } from '@studio/constants/routes';
import { iconColorClass } from '@studio/routes/constants';
import { getWorkspaceSettingsRoute } from '@studio/routes/utils';
import { Settings } from 'lucide-react';
import { lazy } from 'react';
import type { RouteObject } from 'react-router';

const WorkspaceSettingsRoute = lazy(() =>
  import('@studio/routes/WorkspaceSettingsRoute').then((module) => ({
    default: module.WorkspaceSettingsRoute,
  }))
);

export const settingsRoutes: RouteObject[] = [
  {
    path: ROUTES.workspace.settings,
    element: <WorkspaceSettingsRoute />,
    errorElement: <RouteErrorPanel title="Settings" />,
  },
];

export const getSettingsSideNavItems = (workspace: string) => [
  {
    id: 'settings',
    slotIcon: <Settings className={iconColorClass} />,
    slotLabel: 'Settings',
    href: getWorkspaceSettingsRoute(workspace),
  },
];
