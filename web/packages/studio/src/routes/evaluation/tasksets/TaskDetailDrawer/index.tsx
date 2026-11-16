// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { KeyValueGrid } from '@nemo/common/src/components/KeyValueGrid';
import {
  getEvalsGetTaskQueryOptions,
  getEvalsGetTaskRevisionQueryOptions,
} from '@nemo/sdk/generated/evals/evals-plugin-tasks-routes';
import { SidePanel, Stack, Text } from '@nvidia/foundations-react-core';
import { parseTaskRef, RUNNER_LABELS } from '@studio/routes/evaluation/tasksets/helpers';
import { HarborDetails } from '@studio/routes/evaluation/tasksets/TaskDetailDrawer/HarborDetails';
import { InlineDetails } from '@studio/routes/evaluation/tasksets/TaskDetailDrawer/InlineDetails';
import { useQuery } from '@tanstack/react-query';
import type { FC } from 'react';

interface TaskDetailDrawerProps {
  /** The member ref to show; the drawer is closed when unset. */
  taskRef?: string;
  /** Workspace a bare ref resolves against: the taskset's. */
  workspace: string;
  onClose: () => void;
}

/** A member task's definition, rendered by the runner that executes it. A pinned ref shows the
 *  pinned revision; an unpinned one shows the latest. Fetches its own task so a `?task=` deep link
 *  opens whatever page of the table is showing. */
export const TaskDetailDrawer: FC<TaskDetailDrawerProps> = ({ taskRef, workspace, onClose }) => {
  const { name, version, ...parts } = parseTaskRef(taskRef ?? '', workspace);
  const query = { query: { enabled: Boolean(taskRef) } };
  const {
    data: task,
    isPending,
    error,
  } = useQuery(
    version
      ? getEvalsGetTaskRevisionQueryOptions(parts.workspace, name, version, query)
      : getEvalsGetTaskQueryOptions(parts.workspace, name, query)
  );

  const renderBody = () => {
    if (isPending) {
      return <Text className="text-subtle">Loading task...</Text>;
    }
    if (error || !task) {
      return (
        <Text className="text-feedback-danger">
          {getErrorMessage(error ?? new Error(`${taskRef} could not be loaded.`))}
        </Text>
      );
    }
    return (
      <Stack gap="density-xl">
        <KeyValueGrid
          items={[
            { key: 'runner', label: 'Runner', value: RUNNER_LABELS[task.spec.kind] },
            { key: 'revision', label: 'Revision', value: `r${task.revision}` },
            { key: 'ref', label: 'Task Ref', value: taskRef, wrapValue: true },
          ]}
        />
        {task.spec.kind === 'harbor' ? (
          <HarborDetails spec={task.spec} />
        ) : (
          <InlineDetails spec={task.spec} />
        )}
      </Stack>
    );
  };

  return (
    <SidePanel
      open={Boolean(taskRef)}
      onOpenChange={(open) => !open && onClose()}
      side="right"
      bordered
      modal
      className="w-[600px]"
      slotHeading={taskRef && name}
    >
      {taskRef && renderBody()}
    </SidePanel>
  );
};
