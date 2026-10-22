// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { Play } from '@nvidia/react-gui-icons/icons';
import { MODEL_COMPARE_ENABLED } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { iconColorClass } from '@studio/routes/constants';
import { gateModelCompareRoutes, getModelCompareRoute } from '@studio/routes/utils';
import { lazy } from 'react';
import type { RouteObject } from 'react-router';

const ModelCompareRoute =
  MODEL_COMPARE_ENABLED &&
  lazy(() =>
    import('@studio/routes/ModelCompareRoute').then((module) => ({
      default: module.ModelCompareRoute,
    }))
  );

export const modelCompareRoutes: RouteObject[] = gateModelCompareRoutes([
  {
    path: ROUTES.workspace.modelCompare,
    element: ModelCompareRoute ? <ModelCompareRoute /> : null,
    errorElement: <RouteErrorPanel title="Chat" />,
  },
]);

export const getModelCompareSideNavItems = (workspace: string) =>
  MODEL_COMPARE_ENABLED
    ? [
        {
          id: 'playground',
          slotIcon: <Play className={iconColorClass} />,
          slotLabel: 'Playground',
          href: getModelCompareRoute(workspace),
        },
      ]
    : [];
