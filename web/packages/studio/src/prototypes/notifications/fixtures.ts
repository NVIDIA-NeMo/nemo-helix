// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { Apps, Cog } from '@nemo/icons/icons';
import type {
  NavNotification,
  ProtoNavInputItem,
  ProtoNavItem,
  ProtoNavSubItem,
} from '@studio/prototypes/notifications/types';
import { iconColorClass } from '@studio/routes/constants';
import { createElement } from 'react';

const WORKSPACE_ROOT = '/workspaces/default';

export const NAV_HREFS = {
  dashboard: WORKSPACE_ROOT,
  optimizer: `${WORKSPACE_ROOT}/insights`,
  traces: `${WORKSPACE_ROOT}/traces`,
  'agents-group': `${WORKSPACE_ROOT}/agents`,
  'agent-monitor': `${WORKSPACE_ROOT}/agents/monitor`,
  'base-models': `${WORKSPACE_ROOT}/models`,
  'evaluation-results': `${WORKSPACE_ROOT}/evaluations`,
  'custom-models': `${WORKSPACE_ROOT}/fine-tune`,
  playground: `${WORKSPACE_ROOT}/playground`,
  deployments: `${WORKSPACE_ROOT}/deployments`,
  'virtual-models': `${WORKSPACE_ROOT}/virtual-models`,
  experiment: `${WORKSPACE_ROOT}/experiments`,
  tasksets: `${WORKSPACE_ROOT}/tasksets`,
  'dataset-list': `${WORKSPACE_ROOT}/datasets`,
  anonymizer: `${WORKSPACE_ROOT}/anonymizer`,
  'data-designer': `${WORKSPACE_ROOT}/data-designer`,
  safeSynthesizer: `${WORKSPACE_ROOT}/safe-synthesizer`,
  guardrails: `${WORKSPACE_ROOT}/guardrails`,
  datasets: `${WORKSPACE_ROOT}/filesets`,
  jobs: `${WORKSPACE_ROOT}/jobs`,
  settings: `${WORKSPACE_ROOT}/settings`,
} as const;

export type NavId = keyof typeof NAV_HREFS | 'models-group';

export type NotificationMap = Partial<Record<NavId, NavNotification>>;

export const NAV_LABELS: Record<NavId, string> = {
  dashboard: 'Dashboard',
  optimizer: 'Insights',
  traces: 'Traces',
  'agents-group': 'Agents',
  'agent-monitor': 'Monitor',
  'models-group': 'Models',
  'base-models': 'Model Catalog',
  'evaluation-results': 'Evaluations',
  'custom-models': 'Fine-tuning',
  playground: 'Playground',
  deployments: 'Deployments',
  'virtual-models': 'Virtual Models',
  experiment: 'Experiments',
  tasksets: 'Tasksets',
  'dataset-list': 'Datasets',
  anonymizer: 'Anonymizer',
  'data-designer': 'Data Designer',
  safeSynthesizer: 'Safe Synthesizer',
  guardrails: 'Guardrails',
  datasets: 'Filesets',
  jobs: 'Jobs',
  settings: 'Settings',
};

const ICONS: Partial<Record<NavId, typeof Apps>> = {
  dashboard: Apps,
  optimizer: ENTITY_ICONS.optimizerInsights,
  traces: ENTITY_ICONS.telemetryTraces,
  'agents-group': ENTITY_ICONS.agents,
  'agent-monitor': ENTITY_ICONS.agentMonitorRuns,
  'models-group': ENTITY_ICONS.baseModels,
  'base-models': ENTITY_ICONS.baseModels,
  'evaluation-results': ENTITY_ICONS.evaluationResults,
  'custom-models': ENTITY_ICONS.customModels,
  playground: ENTITY_ICONS.evalComparison,
  deployments: ENTITY_ICONS.deployments,
  'virtual-models': ENTITY_ICONS.virtualModels,
  experiment: ENTITY_ICONS.experiments,
  tasksets: ENTITY_ICONS.tasksets,
  'dataset-list': ENTITY_ICONS.datasets,
  anonymizer: ENTITY_ICONS.anonymizerJobs,
  'data-designer': ENTITY_ICONS.dataDesignerJobs,
  safeSynthesizer: ENTITY_ICONS.safeSynthesizerJobs,
  guardrails: ENTITY_ICONS.guardrails,
  datasets: ENTITY_ICONS.filesets,
  jobs: ENTITY_ICONS.jobs,
  settings: Cog,
};

interface BuildOptions {
  parentsOpen?: boolean;
}

export const buildStudioNav = (
  notifications: NotificationMap,
  { parentsOpen = true }: BuildOptions = {}
): ProtoNavInputItem[] => {
  const node = (id: NavId): ProtoNavItem & ProtoNavSubItem => {
    const Icon = ICONS[id];
    return {
      id,
      slotLabel: NAV_LABELS[id],
      slotIcon: Icon ? createElement(Icon, { className: iconColorClass }) : undefined,
      href: id === 'models-group' ? undefined : NAV_HREFS[id],
      notification: notifications[id],
    };
  };

  const subItem = (id: NavId): ProtoNavSubItem => node(id);

  return [
    node('dashboard'),
    { group: 'Observability', items: [node('optimizer'), node('traces')] },
    {
      group: 'Components',
      items: [
        {
          ...node('agents-group'),
          defaultOpen: parentsOpen,
          subItems: [subItem('agent-monitor')],
        },
        {
          ...node('models-group'),
          defaultOpen: parentsOpen,
          subItems: [
            subItem('base-models'),
            subItem('evaluation-results'),
            subItem('custom-models'),
            subItem('playground'),
            subItem('deployments'),
            subItem('virtual-models'),
          ],
        },
      ],
    },
    { group: 'Evaluations', items: [node('experiment'), node('tasksets')] },
    {
      group: 'Data',
      items: [
        node('dataset-list'),
        node('anonymizer'),
        node('data-designer'),
        node('safeSynthesizer'),
      ],
    },
    { group: 'Governance', items: [node('guardrails')] },
    { group: 'System', items: [node('datasets'), node('jobs'), node('settings')] },
  ];
};

export const SAMPLE_NOTIFICATIONS: NotificationMap = {
  'custom-models': { count: 2, severity: 'success' },
  'evaluation-results': { count: 1, severity: 'error' },
  'agents-group': { count: 3, severity: 'info' },
  optimizer: { count: 4, severity: 'info' },
  'data-designer': { count: 128, severity: 'success' },
  guardrails: { count: 1, severity: 'warning' },
  jobs: { count: 7, severity: 'error' },
};
