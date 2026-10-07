// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Block, Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { StartFooter } from '@studio/components/StartOptions/StartFooter';
import { StartOptionCards } from '@studio/components/StartOptions/StartOptionCards';
import { CONTENT_WIDTH } from '@studio/components/StartOptions/tile';
import {
  DEFAULT_STRATEGY,
  STRATEGY_OPTIONS,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizationStrategySelect/constants';
import type {
  OptimizationStrategyId,
  OptimizationStrategySelectProps,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizationStrategySelect/types';
import { ChevronLeft } from 'lucide-react';
import { type FC, useState } from 'react';

/**
 * The "how do you want to set it up?" step between the studies table and whatever creates one.
 *
 * The cards and footer are the create flows' own (`StartPage`), but laid out for a tab rather than
 * a page: the agent's header already sits above it, so the heading and way back match the form's.
 * The cards scroll and the footer stays put, so the list can grow without pushing Continue away.
 */
export const OptimizationStrategySelect: FC<OptimizationStrategySelectProps> = ({
  agentName,
  onBack,
  onContinue,
}) => {
  const [selectedId, setSelectedId] = useState<OptimizationStrategyId>(DEFAULT_STRATEGY);

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
            <StartOptionCards
              options={STRATEGY_OPTIONS}
              value={selectedId}
              onChange={(id) => setSelectedId(id as OptimizationStrategyId)}
            />
          </Stack>
        </Flex>
      </Block>

      <StartFooter
        canContinue
        onContinue={() => onContinue(selectedId)}
        attributes={{ FlexContainer: { className: 'bg-transparent' } }}
      />
    </Stack>
  );
};
