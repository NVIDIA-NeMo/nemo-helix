// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { Flex, PageHeader, Stack } from '@nvidia/foundations-react-core';
import { FeatureFlagBadge } from '@studio/components/FeatureFlagBadge';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { TasksetsDataView } from '@studio/routes/evaluation/tasksets/TasksetListRoute/TasksetsDataView';
import { getTasksetsRoute } from '@studio/routes/utils';
import type { FC } from 'react';

export const TasksetListRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  useBreadcrumbs({ items: [{ href: getTasksetsRoute(workspace), slotLabel: 'Tasksets' }] });

  return (
    <AccessibleTitle title="Tasksets">
      <Stack className="h-full" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading={
            <Flex align="center" gap="density-md">
              Tasksets
              <FeatureFlagBadge flag="tasksetsEnabled" />
            </Flex>
          }
          slotDescription="A taskset is a versioned group of stored tasks that an agent evaluation runs against. Tasksets are created with the CLI or SDK; each change publishes a new revision, and tags such as latest point at one."
        />
        <TasksetsDataView workspace={workspace} />
      </Stack>
    </AccessibleTitle>
  );
};
