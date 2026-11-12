// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Task } from '@nemo/sdk/generated/evals/schema';
import { Stack, Text } from '@nvidia/foundations-react-core';
import { DetailPanel } from '@studio/components/DetailPanel';
import type { FC } from 'react';

/** What scores the task: stored metric refs by name, an inline bundle by its type, and the
 *  reward every Harbor task reports. */
export const TaskMetricsPanel: FC<{ task: Task }> = ({ task: { spec } }) => {
  const metrics = [
    ...(spec.kind === 'harbor' ? [{ name: 'Harbor reward', detail: 'From the task’s tests' }] : []),
    ...(spec.metrics ?? []).map((metric) =>
      typeof metric === 'string'
        ? { name: metric, detail: 'Stored metric' }
        : { name: metric.metric_type, detail: metric.metadata?.description ?? 'Inline metric' }
    ),
  ];

  return (
    <DetailPanel
      title="Metrics"
      flush
      slotAction={
        <Text kind="body/regular/sm" className="text-secondary">
          {metrics.length}
        </Text>
      }
    >
      {metrics.length === 0 ? (
        <Text className="text-subtle block p-4">No metrics score this task.</Text>
      ) : (
        metrics.map(({ name, detail }, index) => (
          <Stack
            key={`${name}-${index}`}
            gap="1"
            className={`px-4 py-3 ${index > 0 ? 'border-t border-base' : ''}`}
          >
            <Text kind="body/semibold/md" className="break-all">
              {name}
            </Text>
            <Text kind="body/regular/sm" className="text-secondary">
              {detail}
            </Text>
          </Stack>
        ))
      )}
    </DetailPanel>
  );
};
