// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { EVALS_ENABLED, TASKSETS_ENABLED } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { iconColorClass } from '@studio/routes/constants';
import {
  gateEvaluationBenchmarksRoutes,
  gateEvaluationRoutes,
  gateModelEvaluationFormRoutes,
  gateTasksetRoutes,
  getEvaluationResultsRoute,
  getTasksetsRoute,
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

const TasksetListRoute = lazy(() =>
  import('@studio/routes/evaluation/tasksets/TasksetListRoute').then((module) => ({
    default: module.TasksetListRoute,
  }))
);
const TasksetDetailRoute = lazy(() =>
  import('@studio/routes/evaluation/tasksets/TasksetDetailRoute').then((module) => ({
    default: module.TasksetDetailRoute,
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
  ...gateModelEvaluationFormRoutes({
    path: ROUTES.workspace.evaluationNew,
    element: <EvaluationNewRoute />,
    errorElement: <RouteErrorPanel title="Evaluator" />,
  }),
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
  ...gateTasksetRoutes([
    {
      path: ROUTES.workspace.tasksets,
      element: <TasksetListRoute />,
      errorElement: <RouteErrorPanel title="Tasksets" />,
    },
    {
      path: ROUTES.workspace.tasksetDetail,
      element: <TasksetDetailRoute />,
      errorElement: <RouteErrorPanel title="Taskset" />,
    },
  ]),
]);

const NavIcon = ENTITY_ICONS.evaluationResults;
const TasksetsIcon = ENTITY_ICONS.tasksets;

export const getEvaluationSideNavItems = (workspace: string) =>
  EVALS_ENABLED
    ? [
        {
          id: 'evaluation-results',
          slotIcon: <NavIcon className={iconColorClass} />,
          slotLabel: 'Evaluations',
          href: getEvaluationResultsRoute(workspace),
        },
      ]
    : [];

/** Tasksets sit in the Evaluations nav group, beside the Experiments they feed. */
export const getTasksetSideNavItems = (workspace: string) =>
  EVALS_ENABLED && TASKSETS_ENABLED
    ? [
        {
          id: 'tasksets',
          slotIcon: <TasksetsIcon className={iconColorClass} />,
          slotLabel: 'Tasksets',
          href: getTasksetsRoute(workspace),
        },
      ]
    : [];
