// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Trace } from '@nemo/sdk/generated/platform/schema';
import { useListTraces } from '@nemo/sdk/generated/platform/traces';
import type { TraceListQuery } from '@studio/components/IntakeDetail/traceListQuery';

export interface TraceListNeighbor {
  sessionId: string;
  traceId: string;
  /** Moves with the row, or the next replay looks on the page it just left. */
  page: number;
}

export interface TraceListNeighbors {
  previous?: TraceListNeighbor;
  next?: TraceListNeighbor;
  inList: boolean;
}

const EMPTY: TraceListNeighbors = { inList: false };

const toNeighbor = (trace: Trace | undefined, page: number): TraceListNeighbor | undefined =>
  trace ? { sessionId: trace.session_id, traceId: trace.id, page } : undefined;

/**
 * The rows either side of the current one in the traces list a detail page was
 * opened from. A row at a page edge costs one extra fetch for its neighbour;
 * its own page is normally already cached by the table.
 */
export const useTraceListNeighbors = (
  workspace: string,
  query: TraceListQuery | null,
  current: { traceId?: string; sessionId: string }
): TraceListNeighbors => {
  const enabled = !!workspace && query !== null;

  const { data: page } = useListTraces(workspace, query ?? undefined, {
    query: { enabled },
  });

  const rows = page?.data ?? [];
  // Without a trace id, the session's first trace is the row the list showed.
  const index = rows.findIndex((trace) =>
    current.traceId ? trace.id === current.traceId : trace.session_id === current.sessionId
  );
  const totalPages = page?.pagination?.total_pages ?? 1;
  const pageNumber = query?.page ?? 1;

  const wantsPreviousPage = index === 0 && pageNumber > 1;
  const wantsNextPage = index >= 0 && index === rows.length - 1 && pageNumber < totalPages;

  const { data: previousPage } = useListTraces(
    workspace,
    query ? { ...query, page: pageNumber - 1 } : undefined,
    { query: { enabled: enabled && wantsPreviousPage } }
  );
  const { data: nextPage } = useListTraces(
    workspace,
    query ? { ...query, page: pageNumber + 1 } : undefined,
    { query: { enabled: enabled && wantsNextPage } }
  );

  // Row gone from the list: no pager beats one that steps somewhere unrelated.
  if (!query || index < 0) {
    return EMPTY;
  }

  const previousRows = previousPage?.data ?? [];
  const nextRows = nextPage?.data ?? [];

  return {
    previous:
      index > 0
        ? toNeighbor(rows[index - 1], pageNumber)
        : toNeighbor(previousRows[previousRows.length - 1], pageNumber - 1),
    next:
      index < rows.length - 1
        ? toNeighbor(rows[index + 1], pageNumber)
        : toNeighbor(nextRows[0], pageNumber + 1),
    inList: true,
  };
};
