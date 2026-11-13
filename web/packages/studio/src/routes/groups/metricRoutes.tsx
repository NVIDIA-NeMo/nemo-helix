// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { EVALS_ENABLED, METRICS_ENABLED } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { iconColorClass } from '@studio/routes/constants';
import { gateMetricsRoutes, getMetricsRoute } from '@studio/routes/utils';
import { lazy } from 'react';
import type { RouteObject } from 'react-router';

const MetricListRoute = lazy(() =>
  import('@studio/routes/MetricListRoute').then((module) => ({ default: module.MetricListRoute }))
);
const MetricDetailRoute = lazy(() =>
  import('@studio/routes/MetricDetailRoute').then((module) => ({
    default: module.MetricDetailRoute,
  }))
);

export const metricRoutes: RouteObject[] = gateMetricsRoutes([
  {
    path: ROUTES.workspace.metrics,
    element: <MetricListRoute />,
    errorElement: <RouteErrorPanel title="Metrics" />,
  },
  {
    path: ROUTES.workspace.metricDetail,
    element: <MetricDetailRoute />,
    errorElement: <RouteErrorPanel title="Metric" />,
  },
]);

const NavIcon = ENTITY_ICONS.metrics;

export const getMetricSideNavItems = (workspace: string) =>
  EVALS_ENABLED && METRICS_ENABLED
    ? [
        {
          id: 'metrics',
          slotIcon: <NavIcon className={iconColorClass} />,
          slotLabel: 'Metrics',
          href: getMetricsRoute(workspace),
        },
      ]
    : [];
