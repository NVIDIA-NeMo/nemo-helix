// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { Badge, Banner, Card, Divider, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { OptimizationTarget } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/optimizationTargets';
import {
  formatRange,
  type OptimizationBudget,
  type OptimizationIntent,
  type SearchParameter,
} from '@studio/routes/agents/AgentDetailRoute/optimizations/optimizationCatalog';
import { type FC } from 'react';

const Stat: FC<{ value: string; label: string }> = ({ value, label }) => (
  <Stack gap="density-xxs">
    <Text kind="body/bold/lg">{value}</Text>
    <Text kind="body/regular/xs" color="secondary">
      {label}
    </Text>
  </Stack>
);

const SAFETY_NOTES = [
  'Runs against the deployed agent — read-only, no config is written',
  'Lands in the Optimizations list straight away',
  'Promotion is a separate, explicit step after the run',
];

export interface RunSummaryPanelProps {
  intent: OptimizationIntent;
  budget: OptimizationBudget;
  searchSpace: SearchParameter[];
  target?: OptimizationTarget;
  blockingReason?: string;
  isSubmitting: boolean;
  onRun: () => void;
}

/**
 * What this run will actually do, and the button that starts it.
 *
 * Restates the form's answers as consequences rather than settings — how many agent runs, how
 * long, what stays untouched — because that is the question a user has right before committing.
 * The estimate is trials × test cases, which is exactly the number of agent invocations the study
 * will make; it is not a cost figure and does not pretend to be one.
 */
export const RunSummaryPanel: FC<RunSummaryPanelProps> = ({
  intent,
  budget,
  searchSpace,
  target,
  blockingReason,
  isSubmitting,
  onRun,
}) => {
  const rows = target?.evaluation.test_case_count;
  const agentRuns = rows === undefined ? undefined : budget.trials * rows;

  return (
    <Card className="h-fit self-start">
      <Stack gap="density-md" className="w-full">
        <Text kind="label/bold/xs" color="secondary">
          THIS RUN WILL SWEEP
        </Text>
        <Flex gap="density-sm" wrap="wrap">
          {searchSpace.map((parameter) => (
            <Badge key={parameter.path} color="gray" kind="outline">
              {parameter.label} {formatRange(parameter)}
            </Badge>
          ))}
        </Flex>
        <Text kind="body/regular/sm" color="secondary">
          {intent.objective} Derived from the {intent.title} intent.
        </Text>

        <Divider />

        <Flex gap="density-xl" wrap="wrap">
          <Stat value={agentRuns === undefined ? '—' : String(agentRuns)} label="agent runs" />
          <Stat value={`~${budget.estimatedMinutes} min`} label="runtime" />
          <Stat value={String(budget.trials)} label="trials" />
        </Flex>
        <Text kind="body/regular/xs" color="secondary">
          {rows === undefined
            ? 'Agent runs are unknown until the evaluation reports its test-case count.'
            : `${budget.trials} trials × ${rows} rows.`}{' '}
          Trials run against the deployed agent — nothing is overwritten until you promote one.
        </Text>

        <Divider />

        {blockingReason ? (
          <Banner kind="inline" status="warning">
            {blockingReason}
          </Banner>
        ) : (
          <Stack gap="density-sm">
            {SAFETY_NOTES.map((note) => (
              <Text key={note} kind="body/regular/sm" color="secondary">
                {note}
              </Text>
            ))}
          </Stack>
        )}

        <LoadingButton
          color="brand"
          className="w-full"
          loading={isSubmitting}
          disabled={!!blockingReason || isSubmitting}
          onClick={onRun}
        >
          Run optimization
        </LoadingButton>
      </Stack>
    </Card>
  );
};
