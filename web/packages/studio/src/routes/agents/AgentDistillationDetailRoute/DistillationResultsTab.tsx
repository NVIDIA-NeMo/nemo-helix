// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  Banner,
  Flex,
  RadioGroupRoot,
  SegmentedControl,
  Spinner,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import type { WorkflowStatus } from '@studio/routes/agents/AgentDetailRoute/distillation/distillationApiTypes';
import {
  BASE_ARM,
  type MockArmResult,
  ORIGINAL_ARM,
} from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationResults';
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
  /** The Intake figures are still being read. */
  loading?: boolean;
  /** The candidate Promote would use. */
  selectedArm?: string;
  onSelectArm: (arm: string) => void;
}

/** The workflow's evaluations side by side. Cost is the platform's estimate from token usage. */
export const DistillationResultsTab: FC<DistillationResultsTabProps> = ({
  status,
  arms,
  loading = false,
  selectedArm,
  onSelectArm,
}) => {
  const [view, setView] = useState<ComparisonView>('cards');

  if (status !== 'completed') {
    return (
      <Text kind="body/regular/md" className="text-secondary" role="status">
        Results appear once every evaluation has finished.
      </Text>
    );
  }
  if (loading) {
    return <Spinner size="medium" aria-label="Loading results" />;
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
  const candidates = arms.filter((arm) => arm.id !== ORIGINAL_ARM && arm.id !== BASE_ARM);
  const distilled = candidates.reduce((best, arm) => (arm.score > best.score ? arm : best));
  const untrained = arms.find((arm) => arm.id === BASE_ARM);
  const references = arms.filter((arm) => arm.id === ORIGINAL_ARM || arm.id === BASE_ARM);
  const selected = candidates.find((arm) => arm.id === selectedArm) ?? distilled;
  const modelSwap = arms.find((arm) => arm.id === BASE_ARM);
  const scoreKept = Math.round((distilled.score / baseline.score) * 100);
  const costShare =
    distilled.estimatedCostPer1kTasks === null || baseline.estimatedCostPer1kTasks === null
      ? null
      : Math.round((distilled.estimatedCostPer1kTasks / baseline.estimatedCostPer1kTasks) * 100);

  const canSelect = candidates.length > 1;

  return (
    <RadioGroupRoot
      name="promote-candidate"
      value={selected.id}
      onValueChange={onSelectArm}
      className="w-full"
    >
    <Stack gap="density-2xl">
      <Banner kind="inline" status="success">
        {baseline.id === ORIGINAL_ARM
          ? `The distilled model keeps ${scoreKept}% of the baseline score${
              costShare === null ? '' : ` at ${costShare}% of the cost`
            }${
              modelSwap && modelSwap.id !== baseline.id
                ? `, and beats a plain model swap by ${(distilled.score - modelSwap.score).toFixed(2)} points`
                : ''
            }.`
          : `The distilled model scores ${distilled.score.toFixed(2)}, against ${baseline.score.toFixed(2)} for the same model untrained.`}
      </Banner>
      <SegmentedControl
        className="w-fit"
        aria-label="Comparison view"
        value={view}
        onValueChange={(next) => setView(next === 'cards' ? 'cards' : 'table')}
        items={VIEW_ITEMS}
      />
      {view === 'table' ? (
        <DistillationScoreboard
          arms={arms}
          selectedArm={selected.id}
          bestArm={distilled.id}
          selectable={canSelect}
        />
      ) : (
        <Stack gap="density-2xl">
          <Stack gap="density-lg">
            <Text kind="title/xs">{canSelect ? 'Candidates: pick one to promote' : 'Candidate'}</Text>
            <Flex gap="density-xl" className="w-full items-stretch overflow-x-auto pb-2">
              {candidates.map((arm) => (
                <div key={arm.id} className="w-80 shrink-0">
                  <DistillationArmCard
                    arm={arm}
                    baseline={baseline}
                    best={canSelect && arm.id === distilled.id}
                    untrained={untrained}
                    selectable={canSelect}
                    selected={canSelect && arm.id === selected.id}
                    showUntrainedDelta
                  />
                </div>
              ))}
            </Flex>
          </Stack>
          <Stack gap="density-lg">
            <Text kind="title/xs">Reference</Text>
            <Flex gap="density-xl" className="w-full items-stretch overflow-x-auto pb-2">
              {references.map((arm) => (
                <div key={arm.id} className="w-80 shrink-0">
                  <DistillationArmCard arm={arm} baseline={baseline} />
                </div>
              ))}
            </Flex>
          </Stack>
        </Stack>
      )}
      <DistillationRelativeBars arms={arms} />
      <Text kind="body/regular/xs" className="text-secondary">
        Cost is estimated from tokens per task at each model's list price. A model with no listed
        price shows no cost.
      </Text>
    </Stack>
    </RadioGroupRoot>
  );
};
