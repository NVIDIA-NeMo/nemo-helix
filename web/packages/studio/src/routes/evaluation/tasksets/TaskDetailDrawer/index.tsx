// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { KeyValueGrid } from '@nemo/common/src/components/KeyValueGrid';
import { Anchor, SidePanel, Stack, Text } from '@nvidia/foundations-react-core';
import { TaskDefinition } from '@studio/routes/evaluation/tasks/TaskDefinition';
import { TaskMetricsPanel } from '@studio/routes/evaluation/tasks/TaskDefinition/TaskMetricsPanel';
import { parseTaskRef, RUNNER_LABELS } from '@studio/routes/evaluation/tasksets/helpers';
import type { TasksetTaskRow } from '@studio/routes/evaluation/tasksets/types';
import { useResolvedTasks } from '@studio/routes/evaluation/tasksets/useResolvedTasks';
import { getTaskDetailRoute } from '@studio/routes/utils';
import { type FC, useMemo } from 'react';
import { Link } from 'react-router';

interface TaskDetailDrawerProps {
  /** The member ref to show; the drawer is closed when unset. */
  taskRef?: string;
  /** Workspace a bare ref resolves against: the taskset's. */
  workspace: string;
  onClose: () => void;
}

/** A member task's definition, rendered by the runner that executes it. Resolves its own task so
 *  a `?task=` deep link opens whatever page of the table is showing. */
export const TaskDetailDrawer: FC<TaskDetailDrawerProps> = ({ taskRef, workspace, onClose }) => {
  const requested = useMemo<TasksetTaskRow[]>(
    () => (taskRef ? [{ id: taskRef, ref: taskRef, ...parseTaskRef(taskRef, workspace) }] : []),
    [taskRef, workspace]
  );
  const { rows, isFetching, error } = useResolvedTasks(requested);
  const row = rows[0];
  const task = row?.task;

  const renderBody = () => {
    if (!row) return null;
    if (isFetching && !task) return <Text className="text-subtle">Loading task...</Text>;
    if (error) return <Text className="text-feedback-danger">{getErrorMessage(error)}</Text>;
    if (!task) {
      return (
        <Text className="text-subtle">
          {row.ref} could not be loaded. The task may have been deleted.
        </Text>
      );
    }
    return (
      <Stack gap="density-xl">
        <KeyValueGrid
          items={[
            { key: 'runner', label: 'Runner', value: RUNNER_LABELS[task.spec.kind] },
            { key: 'revision', label: 'Revision', value: `r${task.revision}` },
            { key: 'ref', label: 'Task Ref', value: row.ref, wrapValue: true },
          ]}
        />
        <Anchor asChild>
          <Link to={getTaskDetailRoute(task.workspace, task.name)}>
            View task revisions and usage
          </Link>
        </Anchor>
        <TaskDefinition task={task} />
        <TaskMetricsPanel task={task} />
      </Stack>
    );
  };

  return (
    <SidePanel
      open={Boolean(row)}
      onOpenChange={(open) => !open && onClose()}
      side="right"
      bordered
      modal
      className="w-[520px]"
      slotHeading={row?.name}
    >
      {renderBody()}
    </SidePanel>
  );
};
