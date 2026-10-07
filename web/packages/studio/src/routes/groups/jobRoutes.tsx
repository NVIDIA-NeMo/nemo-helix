// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { ROUTES } from '@studio/constants/routes';
import { iconColorClass } from '@studio/routes/constants';
import { getWorkspaceJobsRoute } from '@studio/routes/utils';
import { lazy } from 'react';
import type { RouteObject } from 'react-router';

const JobsRoute = lazy(() =>
  import('@studio/routes/JobsRoute').then((module) => ({
    default: module.JobsRoute,
  }))
);
const JobDetailRoute = lazy(() =>
  import('@studio/routes/JobDetailRoute').then((module) => ({
    default: module.JobDetailRoute,
  }))
);

export const jobRoutes: RouteObject[] = [
  {
    path: ROUTES.workspace.jobs,
    element: <JobsRoute />,
    errorElement: <RouteErrorPanel title="Jobs" />,
  },
  {
    path: ROUTES.workspace.jobDetail,
    element: <JobDetailRoute />,
    errorElement: <RouteErrorPanel title="Job Details" />,
  },
];

const NavIcon = ENTITY_ICONS.jobs;

export const getJobSideNavItems = (workspace: string) => [
  {
    id: 'jobs',
    slotIcon: <NavIcon className={iconColorClass} />,
    slotLabel: 'Jobs',
    href: getWorkspaceJobsRoute(workspace),
  },
];
