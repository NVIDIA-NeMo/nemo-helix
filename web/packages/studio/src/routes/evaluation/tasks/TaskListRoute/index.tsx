// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { Flex, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { FeatureFlagBadge } from '@studio/components/FeatureFlagBadge';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { TasksDataView } from '@studio/routes/evaluation/tasks/TaskListRoute/TasksDataView';
import { getTasksRoute } from '@studio/routes/utils';
import type { FC } from 'react';

export const TaskListRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  useBreadcrumbs({ items: [{ href: getTasksRoute(workspace), slotLabel: 'Tasks' }] });

  return (
    <AccessibleTitle title="Tasks">
      <Stack className="h-full" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading={
            <Flex align="center" gap="density-md">
              Tasks
              <FeatureFlagBadge flag="tasksetsEnabled" />
            </Flex>
          }
          slotDescription="A task is one unit of agent work and how it is graded: a Harbor task, or an inline instruction scored by metrics. Tasks are versioned and grouped into tasksets; a taskset pins each task to the revision it was published with."
        />
        <TasksDataView workspace={workspace} />
      </Stack>
    </AccessibleTitle>
  );
};
