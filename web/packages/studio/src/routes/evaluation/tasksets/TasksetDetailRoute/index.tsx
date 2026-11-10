// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { KeyValueGrid } from '@nemo/common/src/components/KeyValueGrid';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { useQueryParams } from '@nemo/common/src/hooks/useQueryParams';
import { FILTERS_SEARCH_PARAM } from '@nemo/common/src/hooks/useStudioDataViewState/columnFiltersParam';
import { useEvalsGetTaskset } from '@nemo/sdk/generated/evals/evals-plugin-tasksets-routes';
import {
  Anchor,
  Flex,
  PageHeader,
  Stack,
  TabsContent,
  TabsList,
  TabsRoot,
  TabsTrigger,
} from '@nvidia/foundations-react-core';
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
import { TaskTable } from '@studio/routes/evaluation/tasksets/TasksetDetailRoute/TaskTable';
import { getTasksetsRoute } from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import type { FC } from 'react';
import { Link } from 'react-router';

/** Both tabs' tables keep their state in the same URL params; clear it when switching tabs. */
const TABLE_STATE_RESET = {
  page: undefined,
  page_size: undefined,
  sort: undefined,
  s: undefined,
  [FILTERS_SEARCH_PARAM]: undefined,
};

export const TasksetDetailRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { tasksetName } = useRequiredPathParams([ROUTE_PARAMS.tasksetName]);
  const { getQueryParam, setQueryParam, setQueryParams } = useQueryParams();

  useBreadcrumbs({
    items: [
      { href: getTasksetsRoute(workspace), slotLabel: 'Tasksets' },
      { slotLabel: tasksetName },
    ],
  });

  const { data: taskset, isPending, error } = useEvalsGetTaskset(workspace, tasksetName);

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
        <TabsRoot
          value={getQueryParam(QUERY_PARAMETERS.tab) || 'tasks'}
          onValueChange={(tab) =>
            setQueryParams({ ...TABLE_STATE_RESET, [QUERY_PARAMETERS.tab]: tab })
          }
          className="flex min-h-0 flex-1 flex-col"
        >
          <TabsList>
            <TabsTrigger value="tasks">Tasks</TabsTrigger>
            <TabsTrigger value="revisions">Revisions</TabsTrigger>
          </TabsList>
          <TabsContent value="tasks" className="flex min-h-0 flex-1 flex-col p-0 pt-density-lg">
            <TaskTable
              taskset={taskset}
              onTaskClick={(row) => setQueryParam(QUERY_PARAMETERS.task, row.ref)}
            />
          </TabsContent>
          <TabsContent value="revisions" className="flex min-h-0 flex-1 flex-col p-0 pt-density-lg">
            <RevisionHistory taskset={taskset} />
          </TabsContent>
        </TabsRoot>
      </Stack>
      <TaskDetailDrawer
        taskRef={getQueryParam(QUERY_PARAMETERS.task) || undefined}
        workspace={taskset.workspace}
        onClose={() => setQueryParam(QUERY_PARAMETERS.task, '')}
      />
    </AccessibleTitle>
  );
};
