// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import type { InsightListItem } from '@nemo/sdk/generated/insights/schema';
import { Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface OpenInsightRowProps {
  readonly insight: InsightListItem;
  readonly onOpen: (insight: InsightListItem) => void;
}

/**
 * One insight in the overview panel: evidence volume in a fixed gutter, then what the
 * problem is and when it was last seen.
 */
export const OpenInsightRow: FC<OpenInsightRowProps> = ({ insight, onOpen }) => {
  const traceCount = insight.trace_refs?.length ?? 0;

  return (
    <Button
      kind="tertiary"
      size="small"
      type="button"
      onClick={() => onOpen(insight)}
      className="w-full px-4 py-3.5 text-left justify-start"
    >
      <Flex gap="4" align="start">
        <Stack align="center" className="w-14 shrink-0">
          <Text kind="label/bold/2xl">{traceCount}</Text>
          <Text kind="body/regular/sm" className="text-placeholder">
            {traceCount === 1 ? 'Trace' : 'Traces'}
          </Text>
        </Stack>
        <Stack gap="1" className="min-w-0 flex-1">
          <Text kind="body/semibold/md">{insight.title}</Text>
          {insight.last_seen_at ? (
            <Text kind="body/regular/sm" className="text-placeholder">
              <RelativeTime datetime={insight.last_seen_at} />
            </Text>
          ) : null}
        </Stack>
      </Flex>
    </Button>
  );
};
