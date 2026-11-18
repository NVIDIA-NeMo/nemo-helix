// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Tag, Text } from '@nvidia/foundations-react-core';
import type { TaskUsageRow } from '@studio/routes/evaluation/tasks/types';
import type { FC } from 'react';

type PinnedRevisionProps = Pick<
  TaskUsageRow,
  'pinnedDigest' | 'pinnedRevision' | 'latestRevision' | 'hasNewerRevision'
>;

/** The task revision a taskset pins: its ordinal (or short digest), noting when a newer one exists.
 *  An older pin is intentional, so the note is informational, not a warning. */
export const PinnedRevision: FC<PinnedRevisionProps> = ({
  pinnedDigest,
  pinnedRevision,
  latestRevision,
  hasNewerRevision,
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
      {hasNewerRevision && (
        <Tag
          kind="outline"
          color="gray"
          readOnly
          title={`The task's latest revision is r${latestRevision}. This taskset keeps its pin so its evaluations stay reproducible.`}
        >
          Newer revision available
        </Tag>
      )}
      {hasNewerRevision === undefined && (
        <Tag
          kind="outline"
          color="gray"
          readOnly
          title="The task's revisions couldn't be loaded to compare this pin with latest."
        >
          Latest unknown
        </Tag>
      )}
    </Flex>
  );
};
