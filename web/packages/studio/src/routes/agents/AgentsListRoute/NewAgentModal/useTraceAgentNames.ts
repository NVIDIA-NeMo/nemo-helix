// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isNotFoundError } from '@nemo/common/src/api/common/utils';
import { agentsGetAgent, getAgentsGetAgentQueryKey } from '@nemo/sdk/generated/agents/agents';
import { useListTraces } from '@nemo/sdk/generated/platform/traces';
import { useQueries } from '@tanstack/react-query';
import { useMemo } from 'react';

/** The API's ceiling. Intake exposes no distinct-agent facet, so the names are deduped from a
 *  page of traces; anything older than this many traces is not offered. */
const TRACE_SCAN_PAGE_SIZE = 1000;

export interface TraceAgentNames {
  /** Agent names seen on traces that have no agent entity yet, oldest naming preserved. */
  readonly names: readonly string[];
  readonly isLoading: boolean;
}

/**
 * Agent names observed in ingested traces that are not registered agents yet.
 *
 * `summary` mode is the cheapest response that still carries `agent_name`: it omits payloads and
 * rollups, which this never reads.
 */
export const useTraceAgentNames = (workspace: string, enabled: boolean): TraceAgentNames => {
  const { data: tracesResponse, isLoading: isTracesLoading } = useListTraces(
    workspace,
    { page_size: TRACE_SCAN_PAGE_SIZE, mode: 'summary', sort: '-started_at' },
    { query: { enabled: enabled && !!workspace } }
  );

  const tracedNames = useMemo(() => {
    const seen = new Set<string>();
    for (const trace of tracesResponse?.data ?? []) {
      const name = trace.agent_name?.trim();
      if (name) seen.add(name);
    }
    return [...seen].sort((a, b) => a.localeCompare(b));
  }, [tracesResponse]);

  // One lookup per name rather than a list: agents paginate with no name filter, so a registered
  // agent past the first page would be offered as new. Here a 404 is the answer, not a failure,
  // which is why these must not retry.
  const lookups = useQueries({
    queries: tracedNames.map((name) => ({
      queryKey: getAgentsGetAgentQueryKey(workspace, name),
      queryFn: () => agentsGetAgent(workspace, name),
      enabled: enabled && !!workspace,
      retry: false,
    })),
  });

  return useMemo(
    () => ({
      // Only a confirmed 404 means unregistered; any other failure leaves the name out rather
      // than offering one whose create would collide.
      names: tracedNames.filter((_, index) => isNotFoundError(lookups[index]?.error)),
      isLoading: isTracesLoading || lookups.some((lookup) => lookup.isLoading),
    }),
    [tracedNames, lookups, isTracesLoading]
  );
};
