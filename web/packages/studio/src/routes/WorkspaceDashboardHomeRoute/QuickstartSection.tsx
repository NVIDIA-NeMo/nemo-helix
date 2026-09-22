// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { Button, Flex, Grid, Stack, Text } from '@nvidia/foundations-react-core';
import {
  AGENTS_ENABLED,
  CUSTOMIZER_ENABLED,
  DATASETS_ENABLED,
  DEPLOYMENTS_ENABLED,
  EVALUATOR_ENABLED,
  EXPERIMENT_ENABLED,
  INTAKE_ENABLED,
  OPTIMIZER_ENABLED,
} from '@studio/constants/environment';
import {
  getAgentsListRoute,
  getEvaluationResultsRoute,
  getExperimentRoute,
  getIntakeTracesRoute,
  getNewCustomizationJobRoute,
  getNewFilesetRoute,
  getOptimizerRoute,
  getWorkspaceNewDeploymentRoute,
} from '@studio/routes/utils';
import { getQuickstartDismissedKey } from '@studio/routes/WorkspaceDashboardHomeRoute/quickstartDismissedStorage';
import {
  DashboardPanel,
  type DashboardPanelAction,
} from '@studio/routes/WorkspaceDashboardRoute/DashboardPanel';
import { useLocalStorage } from '@studio/util/hooks/useLocalStorage';
import { Upload, X } from 'lucide-react';
import type { FC, ReactNode } from 'react';

export interface QuickstartSectionProps {
  workspace: string;
  /**
   * Called right before the section unmounts (i.e. while the dismiss button
   * still has focus), so a caller can move focus somewhere sensible instead
   * of letting the browser drop it to `<body>`.
   */
  onDismiss?: () => void;
  /**
   * Each defaults to the live feature flag. Override in Storybook to preview
   * how the section degrades as individual panels/actions are disabled.
   */
  agentsEnabled?: boolean;
  evaluatorEnabled?: boolean;
  optimizerEnabled?: boolean;
  intakeEnabled?: boolean;
  experimentEnabled?: boolean;
  customizerEnabled?: boolean;
  datasetsEnabled?: boolean;
  deploymentsEnabled?: boolean;
}

interface QuickstartAction extends DashboardPanelAction {
  readonly enabled: boolean;
}

interface QuickstartPanelConfig {
  readonly key: string;
  readonly enabled: boolean;
  readonly icon: ReactNode;
  readonly title: string;
  readonly description: string;
  readonly actions: readonly QuickstartAction[];
}

export const QuickstartSection: FC<QuickstartSectionProps> = ({
  workspace,
  onDismiss,
  agentsEnabled = AGENTS_ENABLED,
  evaluatorEnabled = EVALUATOR_ENABLED,
  optimizerEnabled = OPTIMIZER_ENABLED,
  intakeEnabled = INTAKE_ENABLED,
  experimentEnabled = EXPERIMENT_ENABLED,
  customizerEnabled = CUSTOMIZER_ENABLED,
  datasetsEnabled = DATASETS_ENABLED,
  deploymentsEnabled = DEPLOYMENTS_ENABLED,
}) => {
  const [dismissed, setDismissed] = useLocalStorage<boolean>(getQuickstartDismissedKey(workspace));

  if (dismissed) {
    return null;
  }

  const panels: QuickstartPanelConfig[] = [
    {
      key: 'connect-agent',
      enabled: agentsEnabled,
      icon: <ENTITY_ICONS.agents className="w-8 h-8" />,
      title: 'Connect an Agent',
      description: 'Register an agent you already run and route it through the gateway.',
      actions: [
        // Implied by the panel's own `enabled` above, not an independent gate.
        {
          enabled: true,
          icon: <Upload className="size-4" />,
          label: 'Upload an Agent',
          href: getAgentsListRoute(workspace),
        },
        {
          enabled: evaluatorEnabled,
          icon: <ENTITY_ICONS.evaluationResults className="size-4" />,
          label: 'Evaluate Performance',
          href: getEvaluationResultsRoute(workspace),
        },
        {
          enabled: optimizerEnabled,
          icon: <ENTITY_ICONS.agentOptimizations className="size-4" />,
          label: 'Optimize',
          href: getOptimizerRoute(workspace),
        },
      ],
    },
    {
      key: 'observability',
      enabled: intakeEnabled,
      icon: <ENTITY_ICONS.telemetryTraces className="w-8 h-8" />,
      title: 'Observability',
      description: 'Send existing traces to see how your agent behaves in production.',
      actions: [
        // Implied by the panel's own `enabled` above, not an independent gate.
        {
          enabled: true,
          icon: <Upload className="size-4" />,
          label: 'Import Traces',
          href: getIntakeTracesRoute(workspace),
        },
        {
          enabled: optimizerEnabled,
          icon: <ENTITY_ICONS.optimizerInsights className="size-4" />,
          label: 'Generate Insights',
          href: getOptimizerRoute(workspace),
        },
        {
          enabled: experimentEnabled,
          icon: <ENTITY_ICONS.experiments className="size-4" />,
          label: 'Experiment with Test Cases',
          href: getExperimentRoute(workspace),
        },
      ],
    },
    {
      key: 'customize-models',
      enabled: customizerEnabled,
      icon: <ENTITY_ICONS.customModels className="w-8 h-8" />,
      title: 'Customize Models',
      description:
        'Improve agents by training custom models on your data to achieve measurable results.',
      actions: [
        {
          enabled: datasetsEnabled,
          icon: <ENTITY_ICONS.datasets className="size-4" />,
          label: 'Upload Datasets',
          href: getNewFilesetRoute(workspace),
        },
        // Implied by the panel's own `enabled` above, not an independent gate.
        {
          enabled: true,
          icon: <ENTITY_ICONS.baseModels className="size-4" />,
          label: 'Fine-tune Base Models',
          href: getNewCustomizationJobRoute(workspace),
        },
        {
          enabled: deploymentsEnabled,
          icon: <ENTITY_ICONS.deployments className="size-4" />,
          label: 'Deploy Models',
          href: getWorkspaceNewDeploymentRoute(workspace),
        },
      ],
    },
  ];

  const visiblePanels = panels
    .filter((panel) => panel.enabled)
    .map((panel) => ({
      ...panel,
      actions: panel.actions.filter((action) => action.enabled),
    }));

  if (visiblePanels.length === 0) {
    return null;
  }

  const handleDismiss = () => {
    // Move focus off this section's own button before it unmounts, so the
    // browser doesn't drop keyboard/screen-reader focus to <body>.
    onDismiss?.();
    setDismissed(true);
  };

  return (
    <Stack gap="density-lg">
      <Stack gap="density-xxs">
        <Flex align="center" justify="between" className="w-full">
          <Text kind="title/md">Quickstart</Text>
          <Button
            kind="tertiary"
            color="neutral"
            size="small"
            aria-label="Dismiss Quickstart"
            onClick={handleDismiss}
          >
            <X className="size-4" />
          </Button>
        </Flex>
        <Text kind="body/regular/sm" className="text-secondary">
          Choose a workflow to connect an agent, import traces, or customize models—then follow the
          guided steps to get started.
        </Text>
      </Stack>
      <Grid colMinWidth="320px" gap="density-xl">
        {visiblePanels.map((panel) => (
          <DashboardPanel
            key={panel.key}
            icon={panel.icon}
            title={panel.title}
            description={panel.description}
            actions={panel.actions}
          />
        ))}
      </Grid>
    </Stack>
  );
};
