// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Task } from '@nemo/sdk/generated/evals/schema';
import { Block, Button, Flex, Skeleton, Stack, Text } from '@nvidia/foundations-react-core';
import { DetailPanel } from '@studio/components/DetailPanel';
import { PinnedRevision } from '@studio/routes/evaluation/tasks/TaskDetailRoute/PinnedRevision';
import { useTaskUsage } from '@studio/routes/evaluation/tasks/TaskDetailRoute/useTaskUsage';
import { getTasksetDetailRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { useNavigate } from 'react-router';

/** Tasksets previewed before "View all" switches to the Used By tab. */
const PREVIEW_SIZE = 5;

const Message: FC<{ children: string }> = ({ children }) => (
  <Flex justify="center" align="center" padding="density-xl">
    <Text kind="body/regular/md" className="max-w-72 text-center text-secondary">
      {children}
    </Text>
  </Flex>
);

interface TaskUsagePanelProps {
  task: Task;
  onViewAll: () => void;
}

/** The first tasksets that use the task, with the revision each pins. */
export const TaskUsagePanel: FC<TaskUsagePanelProps> = ({ task, onViewAll }) => {
  const navigate = useNavigate();
  const { ref, rows, totalCount, isFetching, error } = useTaskUsage(task, {
    page_size: PREVIEW_SIZE,
  });
  const isPending = isFetching && rows.length === 0;

  const renderBody = () => {
    if (error) return <Message>Tasksets couldn&apos;t be loaded right now.</Message>;
    if (isPending) {
      return (
        <Stack gap="3" className="p-4">
          <Skeleton className="h-5 w-2/3" />
          <Skeleton className="h-5 w-1/2" />
        </Stack>
      );
    }
    if (rows.length === 0) {
      return (
        <Message>No taskset uses this task yet. Add it to one to run it in an evaluation.</Message>
      );
    }
    return rows.map((row, index) => (
      <Block key={row.id} className={index > 0 ? 'border-t border-base' : undefined}>
        <Button
          kind="tertiary"
          size="small"
          type="button"
          onClick={() => navigate(getTasksetDetailRoute(row.workspace, row.name, { task: ref }))}
          className="w-full justify-start px-4 py-3.5 text-left"
        >
          <Flex align="center" justify="between" gap="density-md" className="w-full">
            <Text kind="body/semibold/md" className="truncate">
              {row.name}
            </Text>
            <PinnedRevision {...row} />
          </Flex>
        </Button>
      </Block>
    ));
  };

  return (
    <DetailPanel
      title="Used by"
      flush
      slotAction={
        !error &&
        !isPending && (
          <Flex gap="3" align="center">
            <Text kind="body/regular/sm" className="text-secondary">
              {`${totalCount ?? rows.length} total`}
            </Text>
            {rows.length > 0 && (
              <Button kind="tertiary" size="small" onClick={onViewAll}>
                View all
              </Button>
            )}
          </Flex>
        )
      }
    >
      {renderBody()}
    </DetailPanel>
  );
};
