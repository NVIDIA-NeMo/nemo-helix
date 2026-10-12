// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { Anchor, Badge, Button, Flex, Text } from '@nvidia/foundations-react-core';
import { EM_DASH } from '@studio/components/dataViews/OptimizationJobsDataView/utils';
import { ROUTING_STRATEGIES } from '@studio/routes/agents/AgentDetailRoute/optimizations/SwitchyardOptimizationForm/formValues';
import {
  routeThreshold,
  type SwitchyardRoute,
} from '@studio/routes/agents/AgentOptimizationDetailRoute/switchyard/switchyardResults';
import { getVirtualModelDetailRoute } from '@studio/routes/utils';
import { type ComponentProps, useCallback } from 'react';
import { Link } from 'react-router';

type MakeRouteColumns = NonNullable<
  ComponentProps<typeof StudioDataView<SwitchyardRoute>>['makeColumns']
>;

export const routingStrategyTitle = (configType: string): string =>
  ROUTING_STRATEGIES.find((strategy) => strategy.id === configType)?.title ?? configType;

/** `Capable model share 0.8` — the strategy's one knob, labelled as the run form labels it. */
export const routeSettingText = (route: SwitchyardRoute): string => {
  const value = routeThreshold(route);
  const label = ROUTING_STRATEGIES.find((strategy) => strategy.id === route.configType)?.threshold
    .label;
  if (value === undefined || !label) return EM_DASH;
  return `${label} ${value.toLocaleString(undefined, { maximumFractionDigits: 3 })}`;
};

interface UseRouteColumnsOptions {
  workspace: string;
  testingVirtualModel?: string;
  onDeploy?: (route: SwitchyardRoute) => void;
}

export const useRouteColumns = ({
  workspace,
  testingVirtualModel,
  onDeploy,
}: UseRouteColumnsOptions): MakeRouteColumns =>
  useCallback<MakeRouteColumns>(
    ({ accessor, display }) => [
      accessor('virtualModel', {
        id: 'virtualModel',
        header: 'Virtual model',
        size: 240,
        enableSorting: true,
        cell: ({ row }) => (
          <Flex align="center" gap="density-sm" className="min-w-0">
            <Anchor asChild>
              <Link
                to={getVirtualModelDetailRoute(workspace, row.original.virtualModel)}
                className="truncate"
                title={row.original.virtualModel}
              >
                {row.original.virtualModel}
              </Link>
            </Anchor>
            {row.original.virtualModel === testingVirtualModel && (
              <Badge kind="outline" color="blue">
                Testing
              </Badge>
            )}
          </Flex>
        ),
      }),
      accessor('configType', {
        id: 'configType',
        header: 'Routing strategy',
        size: 160,
        enableSorting: true,
        cell: ({ row }) => <Text>{routingStrategyTitle(row.original.configType)}</Text>,
      }),
      accessor('capable', {
        id: 'capable',
        header: 'Capable model',
        size: 200,
        enableSorting: true,
        cell: ({ row }) => (
          <Text className="truncate" title={row.original.capable}>
            {row.original.capable}
          </Text>
        ),
      }),
      accessor('efficient', {
        id: 'efficient',
        header: 'Efficient model',
        size: 200,
        enableSorting: true,
        cell: ({ row }) => (
          <Text className="truncate" title={row.original.efficient}>
            {row.original.efficient}
          </Text>
        ),
      }),
      display({
        id: 'setting',
        header: 'Setting',
        size: 200,
        cell: ({ row }) => (
          <Text className="tabular-nums">
            {routeSettingText(row.original)}
            {row.original.judge && row.original.judge !== row.original.capable && (
              <Text kind="body/regular/sm" className="text-secondary block truncate">
                Judge {row.original.judge}
              </Text>
            )}
          </Text>
        ),
      }),
      display({
        id: 'actions',
        header: 'Actions',
        size: 180,
        cell: ({ row }) => (
          <Flex gap="density-sm">
            <Button
              kind="secondary"
              size="small"
              disabled={!onDeploy || !row.original.agentConfigPath}
              onClick={() => onDeploy?.(row.original)}
              aria-label={`Deploy ${row.original.virtualModel}`}
            >
              Deploy
            </Button>
          </Flex>
        ),
      }),
    ],
    [workspace, testingVirtualModel, onDeploy]
  );
