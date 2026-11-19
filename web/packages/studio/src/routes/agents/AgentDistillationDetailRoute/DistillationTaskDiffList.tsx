// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Checkbox, Flex, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import type { ArmId } from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationResults';
import {
  MOCK_TASKS,
  type TaskOutcome,
} from '@studio/routes/agents/AgentDetailRoute/distillation/mockDistillationTasks';
import { type FC, useState } from 'react';

const OUTCOME_BADGES: Record<TaskOutcome, { label: string; color: 'green' | 'yellow' | 'red' }> = {
  1: { label: 'Passed', color: 'green' },
  0.5: { label: 'Partial', color: 'yellow' },
  0: { label: 'Failed', color: 'red' },
};

const COLUMNS: { id: ArmId; title: string }[] = [
  { id: 'original', title: 'Baseline' },
  { id: 'base', title: 'Model swap' },
  { id: 'distilled', title: 'After distillation' },
];

/** The held-out tasks where the distilled model's result differs from the original's. */
export const DistillationTaskDiffList: FC = () => {
  const [showAll, setShowAll] = useState(false);
  const tasks = MOCK_TASKS.filter(
    (task) => showAll || task.outcomes.distilled !== task.outcomes.original
  );

  return (
    <Panel
      slotHeading={showAll ? 'All held-out tasks' : 'Where the distilled model differs from the baseline'}
      elevation="high"
      density="compact"
    >
      <Stack gap="density-lg">
        <Checkbox
          checked={showAll}
          onChange={(event) => setShowAll(event.target.checked)}
          slotLabel="Show all tasks"
        />
        {tasks.length === 0 && (
          <Text kind="body/regular/sm" className="text-secondary">
            The distilled model matches the baseline on every held-out task.
          </Text>
        )}
        {tasks.map((task) => {
          const change = task.outcomes.distilled - task.outcomes.original;
          return (
            <Flex
              key={task.number}
              align="center"
              justify="between"
              gap="density-lg"
              wrap="wrap"
              className="rounded-md border border-base p-3"
            >
              <Flex align="center" gap="density-md">
                <Text kind="body/semibold/sm">Task {task.number}</Text>
              </Flex>
              <Flex align="center" gap="density-xl" wrap="wrap">
                {COLUMNS.map((column) => (
                  <Stack key={column.id} gap="density-xs">
                    <Text kind="body/regular/xs" className="text-secondary">
                      {column.title}
                    </Text>
                    <Badge kind="solid" color={OUTCOME_BADGES[task.outcomes[column.id]].color}>
                      {OUTCOME_BADGES[task.outcomes[column.id]].label}
                    </Badge>
                  </Stack>
                ))}
                {change !== 0 && (
                  <Text
                    kind="body/semibold/sm"
                    className={change > 0 ? 'text-feedback-success' : 'text-feedback-danger'}
                  >
                    {change > 0 ? '▲ Improved' : '▼ Regressed'}
                  </Text>
                )}
              </Flex>
            </Flex>
          );
        })}
      </Stack>
    </Panel>
  );
};
