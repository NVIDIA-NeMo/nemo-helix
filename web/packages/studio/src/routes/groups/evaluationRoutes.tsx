// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { EVALUATOR_ENABLED } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { iconColorClass } from '@studio/routes/constants';
import {
  gateEvaluationBenchmarksRoutes,
  gateEvaluationRoutes,
  getEvaluationResultsRoute,
} from '@studio/routes/utils';
import { lazy } from 'react';
import { Navigate, type RouteObject } from 'react-router';

const EvaluationLayout = lazy(() =>
  import('@studio/routes/evaluation/EvaluationLayout').then((module) => ({
    default: module.EvaluationLayout,
  }))
);
const EvaluationResultsLayout = lazy(() =>
  import('@studio/routes/evaluation/EvaluationResultsLayout').then((module) => ({
    default: module.EvaluationResultsLayout,
  }))
);
const EvaluationResultsRoute = lazy(() =>
  import('@studio/routes/evaluation/EvaluationResultsRoute').then((module) => ({
    default: module.EvaluationResultsRoute,
  }))
);
const EvaluationResultDetailsRoute = lazy(() =>
  import('@studio/routes/evaluation/EvaluationResultDetailsRoute').then((module) => ({
    default: module.EvaluationResultDetailsRoute,
  }))
);
const EvaluationNewRoute = lazy(() =>
  import('@studio/routes/evaluation/EvaluationNewRoute').then((module) => ({
    default: module.EvaluationNewRoute,
  }))
);

export const evaluationRoutes: RouteObject[] = gateEvaluationRoutes([
  {
    path: ROUTES.workspace.evaluation,
    element: <EvaluationLayout />,
    errorElement: <RouteErrorPanel title="Evaluator" />,
    children: [
      {
        index: true,
        element: <Navigate to="results" replace />,
      },
      ...gateEvaluationBenchmarksRoutes([]),
    ],
  },
  {
    path: ROUTES.workspace.evaluationNew,
    element: <EvaluationNewRoute />,
    errorElement: <RouteErrorPanel title="Evaluator" />,
  },
  {
    path: ROUTES.workspace.evaluationResultDetails,
    element: <EvaluationResultDetailsRoute />,
    errorElement: <RouteErrorPanel title="Evaluator" />,
  },
  {
    path: ROUTES.workspace.evaluationResults,
    element: <EvaluationResultsLayout />,
    errorElement: <RouteErrorPanel title="Evaluator" />,
    children: [
      {
        index: true,
        element: <EvaluationResultsRoute />,
      },
    ],
  },
]);

const NavIcon = ENTITY_ICONS.evaluationResults;

export const getEvaluationSideNavItems = (workspace: string) =>
  EVALUATOR_ENABLED
    ? [
        {
          id: 'evaluation-results',
          slotIcon: <NavIcon className={iconColorClass} />,
          // Qualified: the rail hoists this out of Models, next to the agent evaluations link.
          slotLabel: 'Model Evaluations',
          href: getEvaluationResultsRoute(workspace),
        },
      ]
    : [];
