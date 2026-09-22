// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatTile } from '@nemo/common/src/components/StatTile';
import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { Grid } from '@nvidia/foundations-react-core';
import {
  AGENTS_ENABLED,
  CUSTOMIZER_ENABLED,
  EVALUATOR_ENABLED,
  EXPERIMENT_ENABLED,
  OPTIMIZER_ENABLED,
} from '@studio/constants/environment';
import {
  getAgentsListRoute,
  getCustomizationJobListRoute,
  getEvaluationResultsRoute,
  getExperimentRoute,
  getOptimizerRoute,
} from '@studio/routes/utils';
import {
  useDashboardStatCounts,
  type DashboardStatCount,
} from '@studio/routes/WorkspaceDashboardHomeRoute/useDashboardStatCounts';
import type { FC, ReactNode } from 'react';

export interface StatTileRowProps {
  workspace: string;
  /**
   * Each defaults to the live feature flag. Override in Storybook to preview
   * how the row degrades as individual sections are disabled.
   */
  agentsEnabled?: boolean;
  optimizerEnabled?: boolean;
  evaluatorEnabled?: boolean;
  experimentEnabled?: boolean;
  customizerEnabled?: boolean;
}

interface StatTileConfig {
  readonly key: string;
  readonly enabled: boolean;
  readonly label: string;
  readonly icon: ReactNode;
  readonly to: string;
  readonly count: DashboardStatCount;
}

// A failed fetch must never look like a confirmed "0" — both show the same
// placeholder, and StatTile's border tints red for the error case.
const formatCount = ({ isLoading, isError, value }: DashboardStatCount) =>
  isLoading || isError ? '—' : String(value ?? 0);

// Tiles are ~200px wide by design (see the Figma spec); `grow` (the Grid
// default) stretches them to fill any leftover row width, so a disabled tile
// never leaves an empty trailing column at any screen size.
const TILE_MIN_WIDTH = '200px';

export const StatTileRow: FC<StatTileRowProps> = ({
  workspace,
  agentsEnabled = AGENTS_ENABLED,
  optimizerEnabled = OPTIMIZER_ENABLED,
  evaluatorEnabled = EVALUATOR_ENABLED,
  experimentEnabled = EXPERIMENT_ENABLED,
  customizerEnabled = CUSTOMIZER_ENABLED,
}) => {
  const counts = useDashboardStatCounts(workspace, {
    agents: agentsEnabled,
    insights: optimizerEnabled,
    testCases: evaluatorEnabled,
    experiments: experimentEnabled,
    customModels: customizerEnabled,
  });

  const tiles: StatTileConfig[] = [
    {
      key: 'agents',
      enabled: agentsEnabled,
      label: 'Agents',
      icon: <ENTITY_ICONS.agents className="size-4" />,
      to: getAgentsListRoute(workspace),
      count: counts.agents,
    },
    {
      key: 'insights',
      enabled: optimizerEnabled,
      label: 'Insights',
      icon: <ENTITY_ICONS.optimizerInsights className="size-4" />,
      to: getOptimizerRoute(workspace),
      count: counts.insights,
    },
    {
      key: 'test-cases',
      enabled: evaluatorEnabled,
      label: 'Test Cases',
      icon: <ENTITY_ICONS.evaluationResults className="size-4" />,
      to: getEvaluationResultsRoute(workspace),
      count: counts.testCases,
    },
    {
      key: 'experiments',
      enabled: experimentEnabled,
      label: 'Experiments',
      icon: <ENTITY_ICONS.experiments className="size-4" />,
      to: getExperimentRoute(workspace),
      count: counts.experiments,
    },
    {
      key: 'custom-models',
      enabled: customizerEnabled,
      label: 'Custom Models',
      icon: <ENTITY_ICONS.customModels className="size-4" />,
      to: getCustomizationJobListRoute(workspace),
      count: counts.customModels,
    },
  ];

  const visibleTiles = tiles.filter((tile) => tile.enabled);
  if (visibleTiles.length === 0) {
    return null;
  }

  return (
    <Grid colMinWidth={TILE_MIN_WIDTH} gap="density-md">
      {visibleTiles.map((tile) => (
        <StatTile
          key={tile.key}
          variant="actionable"
          label={tile.label}
          value={formatCount(tile.count)}
          icon={tile.icon}
          to={tile.to}
          status={tile.count.isError ? 'error' : undefined}
        />
      ))}
    </Grid>
  );
};
