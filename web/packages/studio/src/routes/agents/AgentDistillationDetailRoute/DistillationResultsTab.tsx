// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Banner,
  Grid,
  GridItem,
  SegmentedControl,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import type { WorkflowStatus } from '@studio/routes/agents/AgentDetailRoute/distillation/distillationApiTypes';
import type { MockArmResult } from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationResults';
import { DistillationArmCard } from '@studio/routes/agents/AgentDistillationDetailRoute/DistillationArmCard';
import { DistillationRelativeBars } from '@studio/routes/agents/AgentDistillationDetailRoute/DistillationRelativeBars';
import { DistillationScoreboard } from '@studio/routes/agents/AgentDistillationDetailRoute/DistillationScoreboard';
import { type FC, useState } from 'react';

type ComparisonView = 'table' | 'cards';

const VIEW_ITEMS = [
  { value: 'table', children: 'Table' },
  { value: 'cards', children: 'Cards' },
];

export interface DistillationResultsTabProps {
  status: WorkflowStatus;
  arms: MockArmResult[] | null;
}

/** The workflow's evaluations side by side. Cost is the platform's estimate from token usage. */
export const DistillationResultsTab: FC<DistillationResultsTabProps> = ({ status, arms }) => {
  const [view, setView] = useState<ComparisonView>('table');

  if (status !== 'completed') {
    return (
      <Text kind="body/regular/md" className="text-secondary" role="status">
        Results appear once every evaluation has finished.
      </Text>
    );
  }
  if (!arms) {
    return (
      <Text kind="body/regular/md" className="text-secondary" role="status">
        The evaluations finished without complete score, token and latency figures, so there is
        nothing to compare.
      </Text>
    );
  }

  const baseline = arms[0];
  const distilled = arms[arms.length - 1];
  const modelSwap = arms.find((arm) => arm.id === 'base');
  const scoreKept = Math.round((distilled.score / baseline.score) * 100);
  const costShare =
    distilled.estimatedCostPer1kTasks === null || baseline.estimatedCostPer1kTasks === null
      ? null
      : Math.round((distilled.estimatedCostPer1kTasks / baseline.estimatedCostPer1kTasks) * 100);

  return (
    <Stack gap="density-2xl">
      <Banner kind="inline" status="success">
        The distilled model keeps {scoreKept}% of the baseline score
        {costShare === null ? '' : ` at ${costShare}% of the cost`}
        {modelSwap && modelSwap.id !== baseline.id
          ? `, and beats a plain model swap by ${(distilled.score - modelSwap.score).toFixed(2)} points`
          : ''}
        .
      </Banner>
      <SegmentedControl
        className="w-fit"
        aria-label="Comparison view"
        value={view}
        onValueChange={(next) => setView(next === 'cards' ? 'cards' : 'table')}
        items={VIEW_ITEMS}
      />
      {view === 'table' ? (
        <DistillationScoreboard arms={arms} />
      ) : (
        <Grid cols={{ base: 1, xl: arms.length }} gap="density-3xl">
          {arms.map((arm) => (
            <GridItem key={arm.id}>
              <DistillationArmCard
                arm={arm}
                baseline={baseline}
                highlighted={arm.id === distilled.id}
              />
            </GridItem>
          ))}
        </Grid>
      )}
      <DistillationRelativeBars arms={arms} />
      <Text kind="body/regular/xs" className="text-secondary">
        Cost is estimated from tokens per task at each model's list price. A model with no listed
        price shows no cost.
      </Text>
    </Stack>
  );
};
