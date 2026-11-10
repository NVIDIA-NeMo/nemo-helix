// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { useEvalsGetTask } from '@nemo/sdk/generated/evals/evals-plugin-tasks-routes';
import { Block, Flex, PageHeader, Stack, Tabs, Tag } from '@nvidia/foundations-react-core';
import { FeatureFlagBadge } from '@studio/components/FeatureFlagBadge';
import { Loading } from '@studio/components/Layouts/Loading';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { OverviewTab } from '@studio/routes/evaluation/tasks/TaskDetailRoute/OverviewTab';
import { RevisionHistory } from '@studio/routes/evaluation/tasks/TaskDetailRoute/RevisionHistory';
import {
  isTaskDetailTab,
  TASK_DETAIL_TABS,
} from '@studio/routes/evaluation/tasks/TaskDetailRoute/tabs';
import { UsedByTable } from '@studio/routes/evaluation/tasks/TaskDetailRoute/UsedByTable';
import { RUNNER_LABELS } from '@studio/routes/evaluation/tasksets/helpers';
import { getTaskDetailRoute, getTasksRoute } from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import type { FC } from 'react';
import { Link, Navigate, useNavigate, useParams } from 'react-router';

export const TaskDetailRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { taskName } = useRequiredPathParams([ROUTE_PARAMS.taskName]);
  const { [ROUTE_PARAMS.taskTab]: tabParam } = useParams();
  const navigate = useNavigate();
  const tab = tabParam ?? TASK_DETAIL_TABS[0].value;

  useBreadcrumbs({
    items: [{ href: getTasksRoute(workspace), slotLabel: 'Tasks' }, { slotLabel: taskName }],
  });

  const { data: task, isPending, error } = useEvalsGetTask(workspace, taskName);

  if (!isTaskDetailTab(tab))
    return <Navigate to={getTaskDetailRoute(workspace, taskName)} replace />;
  if (isPending) return <Loading description="Loading task..." />;
  if (error || !task) {
    return (
      <AccessibleTitle title={`Task ${taskName}`}>
        <Stack className="h-full" padding="density-2xl">
          <ErrorPanel errorMessage={getErrorMessage(error ?? new Error('Task not found'))} />
        </Stack>
      </AccessibleTitle>
    );
  }

  return (
    <AccessibleTitle title={`Task ${taskName}`}>
      <Stack className="h-full" gap="density-xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading={
            <Flex align="center" gap="density-md">
              {task.name}
              <Tag kind="outline" color="gray" readOnly>
                {RUNNER_LABELS[task.spec.kind]}
              </Tag>
              <FeatureFlagBadge flag="tasksetsEnabled" />
            </Flex>
          }
        />
        <Tabs
          // Tabs used purely for navigation (renderLink); override KUI's default overflow:hidden.
          className="overflow-visible"
          value={tab}
          items={TASK_DETAIL_TABS.map(({ value, label }) => ({
            value,
            children: label,
            href: getTaskDetailRoute(workspace, taskName, value),
          }))}
          renderLink={(item) => <Link to={item.href ?? ''}>{item.children}</Link>}
        />
        {tab === 'overview' && (
          <Block className="min-h-0 flex-1 overflow-auto">
            <OverviewTab
              task={task}
              onViewUsage={() => navigate(getTaskDetailRoute(workspace, taskName, 'used-by'))}
            />
          </Block>
        )}
        {tab === 'revisions' && (
          <Block className="flex min-h-0 flex-1 flex-col">
            <RevisionHistory task={task} />
          </Block>
        )}
        {tab === 'used-by' && (
          <Block className="flex min-h-0 flex-1 flex-col">
            <UsedByTable task={task} />
          </Block>
        )}
      </Stack>
    </AccessibleTitle>
  );
};
