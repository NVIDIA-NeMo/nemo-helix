// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Button,
  Divider,
  Flex,
  Panel,
  SegmentedControl,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { AGENTS_ENABLED } from '@studio/constants/environment';
import { getAgentDetailRoute } from '@studio/routes/utils';
import { QuickstartAgentRow } from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartPanel/QuickstartAgentRow';
import {
  buildQuickstartSteps,
  type QuickstartAgent,
  type QuickstartFeatures,
  type QuickstartStep,
  type QuickstartView,
} from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartPanel/quickstartContent';
import { QuickstartStepRow } from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartPanel/QuickstartStepRow';
import { useState, type FC } from 'react';

export type {
  QuickstartAction,
  QuickstartAgent,
  QuickstartFeatures,
  QuickstartStep,
  QuickstartView,
} from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartPanel/quickstartContent';

/**
 * Each flag defaults to the live value. Override in Storybook or a test to see how the
 * panel degrades as the destinations behind individual steps stop being registered.
 */
export interface QuickstartPanelProps extends QuickstartFeatures {
  workspace: string;
  /** Absent means no sample agent is installed, and the section is hidden entirely. */
  agent?: QuickstartAgent;
  steps?: readonly QuickstartStep[];
  /** Seeds the tab on first render; the panel owns the selection from then on. */
  defaultView?: QuickstartView;
  onViewChange?: (view: QuickstartView) => void;
  footerText?: string;
  footerActionLabel?: string;
  /**
   * Which shared workspace to switch to depends on the account, so the destination is
   * the caller's. Without this the footer is not rendered at all.
   */
  onSwitchWorkspace?: () => void;
  /**
   * Opens the Run Evaluation modal. Without a handler the step shows only "View results"
   * rather than a second button pointing at the same place.
   */
  onRunEvaluation?: () => void;
  onOptimize?: () => void;
  /**
   * Every step but one points into the agents route group, as does the agent row itself,
   * so without it there is no panel left to render.
   */
  agentsEnabled?: boolean;
}

/**
 * Onboarding panel for the sandbox dashboard: the sample agent, then four things to do
 * with it, as Studio actions or CLI commands. Presentational; the heading lives on the route.
 */
export const QuickstartPanel: FC<QuickstartPanelProps> = ({
  workspace,
  agent,
  steps,
  defaultView = 'studio',
  onViewChange,
  footerText = 'Ready to start with your own assets?',
  footerActionLabel = 'Switch to Shared Workspace',
  onSwitchWorkspace,
  onRunEvaluation,
  onOptimize,
  agentsEnabled = AGENTS_ENABLED,
  intakeEnabled,
  agentOptimizationsEnabled,
}) => {
  const [view, setView] = useState<QuickstartView>(defaultView);

  if (!agent || !agentsEnabled) {
    return null;
  }

  const resolvedSteps =
    steps ??
    buildQuickstartSteps({
      workspace,
      agent,
      onRunEvaluation,
      onOptimize,
      intakeEnabled,
      agentOptimizationsEnabled,
    });

  // Reachable once steps are flag-gated: the chrome alone is an agent row and a switcher
  // toggling between two empty lists.
  if (resolvedSteps.length === 0) {
    return null;
  }

  const handleViewChange = (next: string) => {
    const nextView: QuickstartView = next === 'cli' ? 'cli' : 'studio';
    setView(nextView);
    onViewChange?.(nextView);
  };

  return (
    <Panel>
      {/* `.nv-panel-content` is not a flex column, so the gap has to come from here. */}
      <Stack gap="density-2xl" className="w-full">
        <Stack gap="density-xl" className="w-full">
          <QuickstartAgentRow agent={agent} href={getAgentDetailRoute(workspace, agent.name)} />

          <SegmentedControl
            size="small"
            className="w-full"
            aria-label="Quickstart view"
            value={view}
            onValueChange={handleViewChange}
            items={[
              { value: 'studio', children: 'NeMo Studio' },
              { value: 'cli', children: 'NeMo CLI' },
            ]}
          />

          {/* `role="list"` is deliberate: WebKit drops list semantics under the
              `list-style: none` Tailwind's preflight applies. jsx-a11y calls it redundant. */}
          <ul role="list" className="flex w-full flex-col">
            {resolvedSteps.map((step, index) => (
              <li key={step.id} className="w-full">
                <QuickstartStepRow
                  step={step}
                  view={view}
                  isLast={index === resolvedSteps.length - 1}
                />
              </li>
            ))}
          </ul>
        </Stack>

        {/* A button that silently does nothing is still focusable and still announced. */}
        {onSwitchWorkspace && (
          <>
            <Divider />
            {/* Not `slotFooter` — that justifies to the end, and this is centred. `wrap`
                because the label cannot shrink on a narrow panel. */}
            <Flex align="center" justify="center" wrap="wrap" gap="density-xl" className="w-full">
              <Text kind="body/regular/md">{footerText}</Text>
              <Button color="neutral" kind="tertiary" size="small" onClick={onSwitchWorkspace}>
                {footerActionLabel}
              </Button>
            </Flex>
          </>
        )}
      </Stack>
    </Panel>
  );
};
