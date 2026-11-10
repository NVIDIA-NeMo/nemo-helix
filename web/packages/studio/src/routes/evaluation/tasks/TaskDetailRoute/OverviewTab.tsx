// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Task } from '@nemo/sdk/generated/evals/schema';
import { Flex, Stack } from '@nvidia/foundations-react-core';
import { TaskDefinition } from '@studio/routes/evaluation/tasks/TaskDefinition';
import { TaskMetricsPanel } from '@studio/routes/evaluation/tasks/TaskDefinition/TaskMetricsPanel';
import { TaskDetailsPanel } from '@studio/routes/evaluation/tasks/TaskDetailRoute/TaskDetailsPanel';
import { TaskMetadataPanel } from '@studio/routes/evaluation/tasks/TaskDetailRoute/TaskMetadataPanel';
import { TaskUsagePanel } from '@studio/routes/evaluation/tasks/TaskDetailRoute/TaskUsagePanel';
import type { FC } from 'react';

interface OverviewTabProps {
  task: Task;
  onViewUsage: () => void;
}

/** Landing view for a task: what it asks and how it is graded, beside what it is and where it
 *  is used. */
export const OverviewTab: FC<OverviewTabProps> = ({ task, onViewUsage }) => (
  <Flex gap="density-2xl" align="start" wrap="wrap" className="w-full pb-6">
    <Stack className="min-w-0 flex-1 basis-[32rem]">
      <TaskDefinition task={task} />
    </Stack>
    <Stack gap="density-2xl" className="w-full shrink-0 lg:w-90">
      <TaskDetailsPanel task={task} />
      <TaskMetricsPanel task={task} />
      <TaskMetadataPanel task={task} />
      <TaskUsagePanel task={task} onViewAll={onViewUsage} />
    </Stack>
  </Flex>
);
