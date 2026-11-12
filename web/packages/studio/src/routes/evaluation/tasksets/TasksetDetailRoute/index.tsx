// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { KeyValueGrid } from '@nemo/common/src/components/KeyValueGrid';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { useQueryParams } from '@nemo/common/src/hooks/useQueryParams';
import { useEvalsGetTaskset } from '@nemo/sdk/generated/evals/evals-plugin-tasksets-routes';
import { Anchor, Block, Flex, PageHeader, Stack, Tabs } from '@nvidia/foundations-react-core';
import { FeatureFlagBadge } from '@studio/components/FeatureFlagBadge';
import { Loading } from '@studio/components/Layouts/Loading';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { QUERY_PARAMETERS } from '@studio/routes/constants';
import {
  getFilesetRoute,
  parseFilesRef,
  tagNames,
} from '@studio/routes/evaluation/tasksets/helpers';
import { TaskDetailDrawer } from '@studio/routes/evaluation/tasksets/TaskDetailDrawer';
import { RevisionHistory } from '@studio/routes/evaluation/tasksets/TasksetDetailRoute/RevisionHistory';
import {
  isTasksetDetailTab,
  TASKSET_DETAIL_TABS,
} from '@studio/routes/evaluation/tasksets/TasksetDetailRoute/tabs';
import { TaskTable } from '@studio/routes/evaluation/tasksets/TasksetDetailRoute/TaskTable';
import { getTasksetDetailRoute, getTasksetsRoute } from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import type { FC } from 'react';
import { Link, Navigate, useParams } from 'react-router';

export const TasksetDetailRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { tasksetName } = useRequiredPathParams([ROUTE_PARAMS.tasksetName]);
  const { [ROUTE_PARAMS.tasksetTab]: tabParam } = useParams();
  const tab = tabParam ?? TASKSET_DETAIL_TABS[0].value;
  const { getQueryParam, setQueryParam } = useQueryParams();

  useBreadcrumbs({
    items: [
      { href: getTasksetsRoute(workspace), slotLabel: 'Tasksets' },
      { slotLabel: tasksetName },
    ],
  });

  const { data: taskset, isPending, error } = useEvalsGetTaskset(workspace, tasksetName);

  if (!isTasksetDetailTab(tab)) {
    return <Navigate to={getTasksetDetailRoute(workspace, tasksetName)} replace />;
  }
  if (isPending) return <Loading description="Loading taskset..." />;
  if (error || !taskset) {
    return (
      <AccessibleTitle title={`Taskset ${tasksetName}`}>
        <Stack className="h-full" padding="density-2xl">
          <ErrorPanel errorMessage={getErrorMessage(error ?? new Error('Taskset not found'))} />
        </Stack>
      </AccessibleTitle>
    );
  }

  const files = taskset.files_ref ? parseFilesRef(taskset.files_ref) : undefined;

  return (
    <AccessibleTitle title={`Taskset ${tasksetName}`}>
      <Stack className="h-full" gap="density-xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading={
            <Flex align="center" gap="density-md">
              {taskset.name}
              <FeatureFlagBadge flag="tasksetsEnabled" />
            </Flex>
          }
          slotDescription={taskset.description}
        />
        <KeyValueGrid
          items={[
            { key: 'tasks', label: 'Tasks', value: String(taskset.tasks?.length ?? 0) },
            { key: 'revision', label: 'Latest Revision', value: `r${taskset.revision}` },
            { key: 'tags', label: 'Tags', value: tagNames(taskset.tags).join(', ') },
            {
              key: 'files',
              label: 'Shared Files',
              wrapValue: true,
              value: files && (
                <Anchor asChild>
                  <Link to={getFilesetRoute(files)}>{taskset.files_ref}</Link>
                </Anchor>
              ),
            },
            {
              key: 'updated',
              label: 'Updated',
              value: <RelativeTime datetime={taskset.updated_at} />,
            },
          ]}
        />
        <Tabs
          // Tabs used purely for navigation (renderLink); override KUI's default overflow:hidden.
          className="overflow-visible"
          value={tab}
          items={TASKSET_DETAIL_TABS.map(({ value, label }) => ({
            value,
            children: label,
            href: getTasksetDetailRoute(workspace, tasksetName, { tab: value }),
          }))}
          renderLink={(item) => <Link to={item.href ?? ''}>{item.children}</Link>}
        />
        {tab === 'tasks' && (
          <Block className="flex min-h-0 flex-1 flex-col">
            <TaskTable
              taskset={taskset}
              onTaskClick={(row) => setQueryParam(QUERY_PARAMETERS.task, row.ref)}
            />
          </Block>
        )}
        {tab === 'revisions' && (
          <Block className="flex min-h-0 flex-1 flex-col">
            <RevisionHistory taskset={taskset} />
          </Block>
        )}
      </Stack>
      <TaskDetailDrawer
        taskRef={getQueryParam(QUERY_PARAMETERS.task) || undefined}
        workspace={taskset.workspace}
        onClose={() => setQueryParam(QUERY_PARAMETERS.task, '')}
      />
    </AccessibleTitle>
  );
};
