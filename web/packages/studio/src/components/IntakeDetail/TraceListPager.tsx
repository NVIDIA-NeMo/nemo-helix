// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Flex } from '@nvidia/foundations-react-core';
import {
  encodeTraceListQuery,
  type TraceListQuery,
} from '@studio/components/IntakeDetail/traceListQuery';
import {
  type TraceListNeighbor,
  useTraceListNeighbors,
} from '@studio/components/IntakeDetail/useTraceListNeighbors';
import { getIntakeSessionTraceRoute } from '@studio/routes/utils';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import type { FC, ReactNode } from 'react';
import { useNavigate } from 'react-router';

const ICON_SIZE = 16;

interface TraceListPagerProps {
  workspace: string;
  sessionId: string;
  traceId?: string;
  /** `null` when the page was not opened from the list, which hides the pager. */
  listQuery: TraceListQuery | null;
}

/**
 * Steps to the trace run either side of this one in the list the page was
 * opened from, so a filtered set can be read through without the table.
 */
export const TraceListPager: FC<TraceListPagerProps> = ({
  workspace,
  sessionId,
  traceId,
  listQuery,
}) => {
  const navigate = useNavigate();
  const { previous, next, inList } = useTraceListNeighbors(workspace, listQuery, {
    traceId,
    sessionId,
  });

  if (!listQuery || !inList) {
    return null;
  }

  const go = (neighbor: TraceListNeighbor) =>
    navigate(
      getIntakeSessionTraceRoute(workspace, neighbor.sessionId, neighbor.traceId, {
        // Carried forward, or the pager would work exactly once.
        traceList: encodeTraceListQuery({ ...listQuery, page: neighbor.page }),
      })
    );

  const step = (label: string, neighbor: TraceListNeighbor | undefined, children: ReactNode) => (
    <Button
      type="button"
      kind="tertiary"
      aria-label={label}
      disabled={!neighbor}
      onClick={neighbor ? () => go(neighbor) : undefined}
    >
      {children}
    </Button>
  );

  return (
    <Flex align="center" gap="density-sm" data-testid="trace-list-pager">
      {step(
        'Previous trace run',
        previous,
        <>
          <ArrowLeft size={ICON_SIZE} />
          Previous
        </>
      )}
      {step(
        'Next trace run',
        next,
        <>
          Next
          <ArrowRight size={ICON_SIZE} />
        </>
      )}
    </Flex>
  );
};
