// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChevronLeft } from '@nemo/icons/icons';
import { useAgentOptimizationListStrategies } from '@nemo/sdk/generated/agent-optimization/agent-optimization';
import { Banner, Block, Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { StartOptionCards } from '@studio/components/StartOptions/StartOptionCards';
import { CONTENT_WIDTH } from '@studio/components/StartOptions/tile';
import {
  strategyOptions,
  UPLOAD_CONFIG_OPTION_ID,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizationStrategySelect/constants';
import type { OptimizationStrategySelectProps } from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizationStrategySelect/types';
import type { FC } from 'react';

/**
 * The "how do you want to set it up?" step between the studies table and whatever creates one.
 *
 * The cards are the create flows' own (`StartPage`), but laid out for a tab rather than a page:
 * the agent's header already sits above it, so the heading and way back match the form's.
 * There is nothing to fill in after picking, so a card goes straight to its strategy rather than
 * waiting on a Continue footer. Nothing is pre-selected, so every card fires on click.
 *
 * The tiles are the platform's installed strategies, so a tile only opens when the job it would
 * submit can actually run, and a newly installed strategy shows up without a Studio change.
 */
export const OptimizationStrategySelect: FC<OptimizationStrategySelectProps> = ({
  agentName,
  onBack,
  onSelect,
  onUploadConfig,
}) => {
  const { data, isError } = useAgentOptimizationListStrategies();
  const strategies = data?.data ?? [];

  const select = (name: string) => {
    if (name === UPLOAD_CONFIG_OPTION_ID) {
      onUploadConfig();
      return;
    }
    const strategy = strategies.find((candidate) => candidate.name === name);
    if (strategy) onSelect(strategy);
  };

  return (
    <Stack className="h-full min-h-0 w-full">
      <Stack gap="density-xl" className="shrink-0 pb-6">
        <Button kind="tertiary" className="w-fit px-0" onClick={onBack}>
          <ChevronLeft className="size-4" aria-hidden />
          Optimizations
        </Button>

        <Stack gap="density-sm">
          <Text kind="title/sm">New optimization</Text>
          <Text kind="body/regular/sm" color="secondary">
            How do you want to set up the study{agentName ? ` on ${agentName}` : ''}?
          </Text>
        </Stack>
      </Stack>

      <Block className="min-h-0 flex-1 overflow-auto pb-6 [scrollbar-gutter:stable]">
        <Flex justify="center" className="w-full">
          <Stack className={CONTENT_WIDTH}>
            {isError ? (
              <Banner status="error" kind="inline">
                <Text kind="body/regular/sm">
                  Could not load the optimization strategies installed on this platform.
                </Text>
              </Banner>
            ) : (
              <StartOptionCards
                options={strategyOptions(strategies)}
                value={null}
                onChange={select}
              />
            )}
          </Stack>
        </Flex>
      </Block>
    </Stack>
  );
};
