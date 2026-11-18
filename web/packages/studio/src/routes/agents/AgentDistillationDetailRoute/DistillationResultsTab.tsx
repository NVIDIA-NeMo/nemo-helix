// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Banner, Grid, GridItem, Stack, Text } from '@nvidia/foundations-react-core';
import {
  buildMockResults,
  HELD_OUT_TASKS,
  ORIGINAL_PRICE,
  STUDENT_PRICE,
} from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationResults';
import type { MockWorkflow } from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationWorkflow';
import { DistillationArmCard } from '@studio/routes/agents/AgentDistillationDetailRoute/DistillationArmCard';
import type { FC } from 'react';

export interface DistillationResultsTabProps {
  workflow: MockWorkflow;
}

/** The three held-out evaluations side by side, with cost estimated from token usage. */
export const DistillationResultsTab: FC<DistillationResultsTabProps> = ({ workflow }) => {
  if (workflow.status !== 'completed') {
    return (
      <Text kind="body/regular/md" className="text-secondary" role="status">
        Results appear once all three evaluations finish.
      </Text>
    );
  }

  const [original, base, distilled] = buildMockResults(workflow.name, workflow.studentModel);
  const scoreKept = Math.round((distilled.score / original.score) * 100);
  const costShare = Math.round(
    (distilled.estimatedCostPer1kTasks / original.estimatedCostPer1kTasks) * 100
  );

  return (
    <Stack gap="density-2xl">
      <Banner kind="inline" status="success">
        The distilled model keeps {scoreKept}% of the original score at {costShare}% of the cost,
        and beats the untrained small model by {(distilled.score - base.score).toFixed(2)} points.
      </Banner>
      <Grid cols={3} gap="density-lg">
        {[original, base, distilled].map((arm) => (
          <GridItem key={arm.id}>
            <DistillationArmCard
              arm={arm}
              baseline={original}
              highlighted={arm.id === distilled.id}
            />
          </GridItem>
        ))}
      </Grid>
      <Text kind="body/regular/xs" className="text-secondary">
        Cost is estimated from tokens per task at ${ORIGINAL_PRICE.inputPerMillion.toFixed(2)} /{' '}
        ${ORIGINAL_PRICE.outputPerMillion.toFixed(2)} per million input / output tokens for the
        original model and ${STUDENT_PRICE.inputPerMillion.toFixed(2)} /{' '}
        ${STUDENT_PRICE.outputPerMillion.toFixed(2)} for the small model. The fine-tuned adapter is
        priced as its base model. All three evaluations use the same {HELD_OUT_TASKS} held-out tasks
        and judge, and none of those tasks were used for training.
      </Text>
    </Stack>
  );
};
