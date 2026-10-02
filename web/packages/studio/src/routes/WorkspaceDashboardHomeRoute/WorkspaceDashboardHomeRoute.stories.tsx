// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { GradientBackground } from '@nemo/common/src/components/GradientBackground';
import { PageHeader, Stack } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import { QuickstartSection } from '@studio/routes/WorkspaceDashboardHomeRoute/QuickstartSection';
import { StatTileRow } from '@studio/routes/WorkspaceDashboardHomeRoute/StatTileRow';
import { http, HttpResponse } from 'msw';
import type { FC } from 'react';

/**
 * Every StatTile/Quickstart action fetches a `total_results` count from a
 * different service — this generic handler answers any of them so the story
 * can focus on flag behavior instead of per-endpoint fixtures.
 */
const genericTotalResultsHandler = http.get('*', () =>
  HttpResponse.json({
    data: [],
    pagination: { page: 1, page_size: 1, total_pages: 1, total_results: 7 },
  })
);

interface DashboardFlagsArgs {
  agentsEnabled: boolean;
  optimizerEnabled: boolean;
  evaluatorEnabled: boolean;
  experimentEnabled: boolean;
  customizerEnabled: boolean;
  intakeEnabled: boolean;
  datasetsEnabled: boolean;
  deploymentsEnabled: boolean;
}

/**
 * Mirrors `WorkspaceDashboardHomeRoute`, but with every feature flag lifted to
 * a prop so Storybook controls can drive them — the real route always reads
 * the live flags from `@studio/constants/environment` instead.
 */
const DashboardFlagsPreview: FC<DashboardFlagsArgs> = ({
  agentsEnabled,
  optimizerEnabled,
  evaluatorEnabled,
  experimentEnabled,
  customizerEnabled,
  intakeEnabled,
  datasetsEnabled,
  deploymentsEnabled,
}) => (
  <GradientBackground>
    <Stack gap="density-3xl" padding="density-2xl" className="relative">
      <PageHeader slotHeading="Dashboard" />
      <StatTileRow
        workspace="acme"
        agentsEnabled={agentsEnabled}
        optimizerEnabled={optimizerEnabled}
        evaluatorEnabled={evaluatorEnabled}
        experimentEnabled={experimentEnabled}
        customizerEnabled={customizerEnabled}
      />
      <QuickstartSection
        workspace="acme"
        agentsEnabled={agentsEnabled}
        evaluatorEnabled={evaluatorEnabled}
        optimizerEnabled={optimizerEnabled}
        intakeEnabled={intakeEnabled}
        experimentEnabled={experimentEnabled}
        customizerEnabled={customizerEnabled}
        datasetsEnabled={datasetsEnabled}
        deploymentsEnabled={deploymentsEnabled}
      />
    </Stack>
  </GradientBackground>
);

const meta = {
  title: 'Routes/WorkspaceDashboardHomeRoute/Feature Flags',
  component: DashboardFlagsPreview,
  parameters: {
    msw: { handlers: [genericTotalResultsHandler] },
  },
  argTypes: {
    agentsEnabled: { control: 'boolean', name: 'AGENTS_ENABLED' },
    optimizerEnabled: { control: 'boolean', name: 'OPTIMIZER_ENABLED' },
    evaluatorEnabled: { control: 'boolean', name: 'EVALUATOR_ENABLED' },
    experimentEnabled: { control: 'boolean', name: 'EXPERIMENT_ENABLED' },
    customizerEnabled: { control: 'boolean', name: 'CUSTOMIZER_ENABLED' },
    intakeEnabled: { control: 'boolean', name: 'INTAKE_ENABLED' },
    datasetsEnabled: { control: 'boolean', name: 'DATASETS_ENABLED' },
    deploymentsEnabled: { control: 'boolean', name: 'DEPLOYMENTS_ENABLED' },
  },
} satisfies Meta<typeof DashboardFlagsPreview>;

export default meta;
type Story = StoryObj<typeof meta>;

const ALL_ENABLED: DashboardFlagsArgs = {
  agentsEnabled: true,
  optimizerEnabled: true,
  evaluatorEnabled: true,
  experimentEnabled: true,
  customizerEnabled: true,
  intakeEnabled: true,
  datasetsEnabled: true,
  deploymentsEnabled: true,
};

/** Toggle any control below to see how the dashboard reacts. */
export const AllFlagsEnabled: Story = {
  args: ALL_ENABLED,
};

export const ExperimentsDisabled: Story = {
  args: { ...ALL_ENABLED, experimentEnabled: false },
};

export const DeploymentsDisabled: Story = {
  args: { ...ALL_ENABLED, deploymentsEnabled: false },
};

/** Only Agents is on — every other StatTile and Quickstart panel/action disappears. */
export const AgentsOnly: Story = {
  args: {
    agentsEnabled: true,
    optimizerEnabled: false,
    evaluatorEnabled: false,
    experimentEnabled: false,
    customizerEnabled: false,
    intakeEnabled: false,
    datasetsEnabled: false,
    deploymentsEnabled: false,
  },
};

/** Every flag off — StatTile row and Quickstart section both collapse to nothing. */
export const AllFlagsDisabled: Story = {
  args: {
    agentsEnabled: false,
    optimizerEnabled: false,
    evaluatorEnabled: false,
    experimentEnabled: false,
    customizerEnabled: false,
    intakeEnabled: false,
    datasetsEnabled: false,
    deploymentsEnabled: false,
  },
};
