// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouteErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { AGENT_OPTIMIZATIONS_ENABLED, MONITOR_ENABLED } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { iconColorClass } from '@studio/routes/constants';
import { getAgentMonitorRoute } from '@studio/routes/utils';
import { lazy } from 'react';
import type { RouteObject } from 'react-router';

const AgentsListRoute = lazy(() =>
  import('@studio/routes/agents/AgentsListRoute').then((m) => ({
    default: m.AgentsListRoute,
  }))
);
const AgentDetailRoute = lazy(() =>
  import('@studio/routes/agents/AgentDetailRoute').then((m) => ({
    default: m.AgentDetailRoute,
  }))
);
const AgentMonitorRoute = lazy(() =>
  import('@studio/routes/agents/AgentMonitorRoute').then((m) => ({
    default: m.AgentMonitorRoute,
  }))
);
const AgentEvaluationDetailRoute = lazy(() =>
  import('@studio/routes/agents/AgentEvaluationsRoute').then((m) => ({
    default: m.AgentEvaluationDetailRoute,
  }))
);
const AgentOptimizationDetailRoute =
  AGENT_OPTIMIZATIONS_ENABLED &&
  lazy(() =>
    import('@studio/routes/agents/AgentOptimizationDetailRoute/index').then((m) => ({
      default: m.AgentOptimizationDetailRoute,
    }))
  );

export const agentRoutes: RouteObject[] = [
  {
    path: ROUTES.workspace.agentsList,
    element: <AgentsListRoute />,
    errorElement: <RouteErrorPanel title="Agents" />,
  },
  ...(MONITOR_ENABLED
    ? [
        {
          path: ROUTES.workspace.agentMonitor,
          element: <AgentMonitorRoute />,
          errorElement: <RouteErrorPanel title="Monitor" />,
        },
      ]
    : []),
  {
    path: ROUTES.workspace.agentEvaluationDetail,
    element: <AgentEvaluationDetailRoute />,
    errorElement: <RouteErrorPanel title="Agent Evaluation" />,
  },
  ...(AgentOptimizationDetailRoute
    ? [
        {
          path: ROUTES.workspace.agentOptimizationDetail,
          element: <AgentOptimizationDetailRoute />,
          errorElement: <RouteErrorPanel title="Agent Optimization" />,
        },
      ]
    : []),
  {
    path: ROUTES.workspace.agentDetail,
    element: <AgentDetailRoute />,
    errorElement: <RouteErrorPanel title="Agent details" />,
  },
];

const NavIcon = ENTITY_ICONS.agentMonitorRuns;

export const getAgentSideNavItems = (workspace: string) =>
  MONITOR_ENABLED
    ? [
        {
          id: 'agent-monitor',
          slotIcon: <NavIcon className={iconColorClass} />,
          slotLabel: 'Monitor',
          href: getAgentMonitorRoute(workspace),
        },
      ]
    : [];
