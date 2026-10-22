// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { Cog } from '@nvidia/react-gui-icons/icons';
import { SETTINGS_ENABLED } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { iconColorClass } from '@studio/routes/constants';
import { gateSettingsRoutes, getWorkspaceSettingsRoute } from '@studio/routes/utils';
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
    errorElement: <RouteErrorPanel title="Cog" />,
  },
];

export const getSettingsSideNavItems = (workspace: string) =>
  SETTINGS_ENABLED
    ? [
        {
          id: 'settings',
          slotIcon: <Cog className={iconColorClass} />,
          slotLabel: 'Settings',
          href: getWorkspaceSettingsRoute(workspace),
        },
      ]
    : [];
