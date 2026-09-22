// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { CUSTOMIZER_ENABLED } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { iconColorClass } from '@studio/routes/constants';
import {
  gateCustomizationRoutes,
  getWorkspaceCustomizationJobListRoute,
} from '@studio/routes/utils';
import { lazy } from 'react';
import type { RouteObject } from 'react-router';

const NewCustomizationRoute = lazy(() =>
  import('@studio/routes/NewCustomizationRoute/index').then((module) => ({
    default: module.NewCustomizationRoute,
  }))
);
const CustomizationJobListRoute = lazy(() =>
  import('@studio/routes/CustomizationJobListRoute').then((module) => ({
    default: module.CustomizationJobListRoute,
  }))
);
const CustomizationJobDetailsRoute = lazy(() =>
  import('@studio/routes/CustomizationJobDetailsRoute').then((module) => ({
    default: module.CustomizationJobDetailsRoute,
  }))
);

export const customizationRoutes: RouteObject[] = gateCustomizationRoutes([
  {
    path: ROUTES.workspace.newCustomizationJob,
    element: <NewCustomizationRoute />,
    errorElement: <RouteErrorPanel title="Customizer" />,
  },
  {
    path: ROUTES.workspace.customizationJobList,
    element: <CustomizationJobListRoute />,
    errorElement: <RouteErrorPanel title="Customizer" />,
  },
  {
    path: ROUTES.workspace.customizationJobDetails,
    element: <CustomizationJobDetailsRoute />,
    errorElement: <RouteErrorPanel title="Customizer" />,
  },
]);

const NavIcon = ENTITY_ICONS.customModels;

export const getCustomizationSideNavItems = (workspace: string) =>
  CUSTOMIZER_ENABLED
    ? [
        {
          id: 'custom-models',
          slotIcon: <NavIcon className={iconColorClass} />,
          slotLabel: 'Fine-tuning',
          href: getWorkspaceCustomizationJobListRoute(workspace),
        },
      ]
    : [];
