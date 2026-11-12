// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KVPair } from '@nemo/common/src/components/KVPair';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import type { Task } from '@nemo/sdk/generated/evals/schema';
import { Flex, Stack, Tag } from '@nvidia/foundations-react-core';
import { DetailPanel } from '@studio/components/DetailPanel';
import { RUNNER_LABELS, tagNames } from '@studio/routes/evaluation/tasksets/helpers';
import type { FC } from 'react';

/** At-a-glance identity for the task, beside its definition. */
export const TaskDetailsPanel: FC<{ task: Task }> = ({ task }) => (
  <DetailPanel title="Details">
    <Stack gap="3">
      <KVPair
        label="Runner"
        value={RUNNER_LABELS[task.spec.kind]}
        size="narrow"
        orientation="horizontal"
      />
      <KVPair label="Revision" value={`r${task.revision}`} size="narrow" orientation="horizontal" />
      <KVPair
        label="Tags"
        size="narrow"
        orientation="horizontal"
        value={
          <Flex gap="density-xs" wrap="wrap">
            {tagNames(task.tags).map((tag) => (
              <Tag key={tag} kind="outline" color="gray" readOnly>
                {tag}
              </Tag>
            ))}
          </Flex>
        }
      />
      {task.spec.kind === 'harbor' && (
        <>
          <KVPair
            label="Harbor ID"
            value={task.spec.native_task_id}
            size="narrow"
            orientation="horizontal"
            truncate
          />
          <KVPair
            label="Harbor"
            value={`v${task.spec.harbor_hash.harbor_version}`}
            size="narrow"
            orientation="horizontal"
          />
        </>
      )}
      {task.project && (
        <KVPair label="Project" value={task.project} size="narrow" orientation="horizontal" />
      )}
      <KVPair label="Task ID" value={task.id} size="narrow" orientation="horizontal" truncate />
      <KVPair
        label="Created"
        value={<RelativeTime datetime={task.created_at} />}
        size="narrow"
        orientation="horizontal"
      />
      <KVPair
        label="Updated"
        value={<RelativeTime datetime={task.updated_at} />}
        size="narrow"
        orientation="horizontal"
      />
    </Stack>
  </DetailPanel>
);
