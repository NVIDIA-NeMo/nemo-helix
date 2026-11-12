// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Tag, Text } from '@nvidia/foundations-react-core';
import type { TaskUsageRow } from '@studio/routes/evaluation/tasks/types';
import type { FC } from 'react';

type PinnedRevisionProps = Pick<TaskUsageRow, 'pinnedDigest' | 'pinnedRevision' | 'behindLatest'>;

/** The task revision a taskset pins: its ordinal (or short digest), flagged when behind latest. */
export const PinnedRevision: FC<PinnedRevisionProps> = ({
  pinnedDigest,
  pinnedRevision,
  behindLatest,
}) => {
  if (!pinnedDigest) return <Text className="text-subtle">latest</Text>;
  return (
    <Flex align="center" gap="density-sm">
      {pinnedRevision ? (
        <Text>r{pinnedRevision}</Text>
      ) : (
        <Text kind="mono/sm" title={pinnedDigest}>
          {pinnedDigest.slice(0, 12)}
        </Text>
      )}
      {behindLatest && (
        <Tag kind="outline" color="yellow" readOnly>
          Behind latest
        </Tag>
      )}
    </Flex>
  );
};
