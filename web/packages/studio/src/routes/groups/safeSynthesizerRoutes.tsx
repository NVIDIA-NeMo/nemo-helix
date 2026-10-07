// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { ROUTES } from '@studio/constants/routes';
import { iconColorClass } from '@studio/routes/constants';
import { getWorkspaceSafeSynthesizerRoute } from '@studio/routes/utils';
import { FC, lazy } from 'react';
import type { RouteObject } from 'react-router';

const SafeSynthesizerListRoute = lazy(() =>
  import('@studio/routes/SafeSynthesizerListRoute').then((m) => ({
    default: m.SafeSynthesizerListRoute as FC,
  }))
);
const SafeSynthesizerNewRoute = lazy(() =>
  import('@studio/routes/SafeSynthesizerNewRoute').then((m) => ({
    default: m.SafeSynthesizerNewRoute as FC,
  }))
);
const GenerateJobDetailsRoute = lazy(() =>
  import('@studio/routes/SafeSynthesizerJobDetailsRoute').then((m) => ({
    default: m.GenerateJobDetailsRoute as FC,
  }))
);
const GenerateJobReportRoute = lazy(() =>
  import('@studio/routes/SafeSynthesizerJobReportRoute').then((m) => ({
    default: m.GenerateJobReportRoute as FC,
  }))
);

export const safeSynthesizerRoutes: RouteObject[] = [
  {
    path: ROUTES.workspace.safeSynthesizer,
    element: <SafeSynthesizerListRoute />,
    errorElement: <RouteErrorPanel title="Safe Synthesizer" />,
  },
  {
    path: ROUTES.workspace.safeSynthesizerNew,
    element: <SafeSynthesizerNewRoute />,
    errorElement: <RouteErrorPanel title="Safe Synthesizer" />,
  },
  {
    path: ROUTES.workspace.safeSynthesizerJob,
    element: <GenerateJobDetailsRoute />,
    errorElement: <RouteErrorPanel title="Safe Synthesizer" />,
  },
  {
    path: ROUTES.workspace.safeSynthesizerJobReport,
    element: <GenerateJobReportRoute />,
    errorElement: <RouteErrorPanel title="Safe Synthesizer" />,
  },
];

const NavIcon = ENTITY_ICONS.safeSynthesizerJobs;

export const getSafeSynthesizerSideNavItems = (workspace: string) => [
  {
    id: 'safeSynthesizer',
    slotIcon: <NavIcon className={iconColorClass} />,
    slotLabel: 'Safe Synthesizer',
    href: getWorkspaceSafeSynthesizerRoute(workspace),
  },
];
