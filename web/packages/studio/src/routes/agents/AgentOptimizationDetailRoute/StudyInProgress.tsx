// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KeyValueGrid } from '@nemo/common/src/components/KeyValueGrid';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { useAgentOptimizationGetRunStrategyJobStatus } from '@nemo/sdk/generated/agent-optimization/agent-optimization';
import type { RunStrategyJob } from '@nemo/sdk/generated/agent-optimization/schema';
import { Flex, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import { ListCheckmark, Sliders } from '@nvidia/react-gui-icons/icons';
import { routingStrategyTitle } from '@studio/components/dataViews/SwitchyardRoutesDataView/useRouteColumns';
import { SWITCHYARD_STRATEGY } from '@studio/routes/agents/AgentDetailRoute/optimizations/OptimizationStrategySelect/constants';
import type { FC } from 'react';

/** A spec list field — strategy-specific, so untyped on `RunStrategySpec` — as one line. */
const listText = (value: unknown, format: (item: string) => string = (item) => item) =>
  Array.isArray(value) && value.length
    ? value.map((item) => format(String(item))).join(', ')
    : undefined;

export interface StudyInProgressProps {
  workspace: string;
  job: RunStrategyJob;
  /** The study has been submitted but its job has not started running yet. */
  isQueued: boolean;
}

/**
 * What a study shows before its trials exist: the submitted spec, the job's step-by-step
 * progress — polled until the study reaches a terminal status.
 */
export const StudyInProgress: FC<StudyInProgressProps> = ({ workspace, job, isQueued }) => {
  const { name: jobName, spec } = job;

  const { data: jobStatus } = useAgentOptimizationGetRunStrategyJobStatus(workspace, jobName, {
    query: { enabled: !!workspace && !!jobName, refetchInterval: JOB_POLLING_INTERVAL_MS },
  });
  const steps = jobStatus?.steps ?? [];
  const isSwitchyard = spec.strategy === SWITCHYARD_STRATEGY;
  const resultNoun = isSwitchyard ? 'Virtual models' : 'Trials';

  const strategyItems = isSwitchyard
    ? [
        { key: 'models', label: 'Models', value: listText(spec.models), wrapValue: true },
        {
          key: 'routing-strategies',
          label: 'Routing strategies',
          value: listText(spec.routing_strategies, routingStrategyTitle),
          wrapValue: true,
        },
      ]
    : [
        { key: 'config', label: 'Config', value: spec.optimize_config, wrapValue: true },
        {
          key: 'config-fileset',
          label: 'Config fileset',
          value: spec.optimize_config_fileset,
          wrapValue: true,
        },
        { key: 'output', label: 'Output', value: spec.output, wrapValue: true },
      ];

  return (
    <Stack gap="density-2xl" className="min-h-0 w-full" data-testid="study-in-progress">
      <Panel slotHeading="Study details" slotIcon={<Sliders />} elevation="high" density="compact">
        <Stack gap="density-xl">
          <Text kind="body/regular/md" className="text-secondary" role="status">
            {isQueued
              ? `Waiting for the study to start. ${resultNoun} appear once it finishes.`
              : `${resultNoun} appear once the study finishes.`}
          </Text>
          <KeyValueGrid
            items={[
              { key: 'strategy', label: 'Strategy', value: spec.strategy },
              { key: 'agent', label: 'Agent', value: spec.agent },
              ...strategyItems,
              {
                key: 'submitted',
                label: 'Submitted',
                value: job.created_at ? <RelativeTime datetime={job.created_at} /> : undefined,
              },
              {
                key: 'updated',
                label: 'Last updated',
                value: job.updated_at ? <RelativeTime datetime={job.updated_at} /> : undefined,
              },
            ]}
          />
        </Stack>
      </Panel>

      {steps.length > 0 && (
        <Panel
          slotHeading="Progress"
          slotIcon={<ListCheckmark />}
          elevation="high"
          density="compact"
        >
          <Stack gap="density-md" role="list" aria-label="Study steps">
            {steps.map((step) => (
              <Flex key={step.id} align="center" justify="between" gap="density-md" role="listitem">
                <Text kind="body/regular/md">{step.name}</Text>
                <Flex align="center" gap="density-md">
                  {step.updated_at && (
                    <Text kind="body/regular/sm" className="text-secondary">
                      <RelativeTime datetime={step.updated_at} />
                    </Text>
                  )}
                  <StatusBadge status={step.status} />
                </Flex>
              </Flex>
            ))}
          </Stack>
        </Panel>
      )}
    </Stack>
  );
};
