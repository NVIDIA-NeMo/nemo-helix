// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useInsightsListInsights } from '@nemo/sdk/generated/insights/insights-insights';
import type { InsightListItem } from '@nemo/sdk/generated/insights/schema';
import { useMemo } from 'react';

/**
 * Enough rows to rank a meaningful "top N" client-side without paging. The panel shows
 * {@link OVERVIEW_INSIGHT_LIMIT}; the rest of the page only feeds the ordering.
 */
const INSIGHT_PAGE_SIZE = 20;

export const OVERVIEW_INSIGHT_LIMIT = 5;

interface UseOpenInsightsParams {
  readonly workspace: string;
  /** Registered agent name. Insights carry agent attribution, so this filters server-side. */
  readonly agent?: string;
  readonly enabled: boolean;
}

export interface UseOpenInsightsResult {
  readonly insights: InsightListItem[];
  /** Every open insight for the agent, not just the ranked slice rendered. */
  readonly totalCount: number;
  /** Also pending before the agent resolves, so the panel never flashes its empty state. */
  readonly isPending: boolean;
  readonly error: unknown;
}

/**
 * Rank by evidence volume, then recency.
 *
 * The analyst prioritizes by impact when it files an Insight, but that ranking is not persisted —
 * `Insight` has no severity field — so the panel falls back to the two impact signals the list
 * response does carry. Replace this with a server-side sort once severity lands.
 */
const byImpact = (a: InsightListItem, b: InsightListItem): number => {
  const volume = (b.trace_refs?.length ?? 0) - (a.trace_refs?.length ?? 0);
  if (volume !== 0) return volume;
  return (
    Date.parse(b.last_seen_at ?? b.created_at ?? '') -
    Date.parse(a.last_seen_at ?? a.created_at ?? '')
  );
};

/** Open insights the analyst has filed against this agent, ranked for the overview panel. */
export const useOpenInsights = ({
  workspace,
  agent,
  enabled,
}: UseOpenInsightsParams): UseOpenInsightsResult => {
  const { data, isPending, error } = useInsightsListInsights(
    workspace,
    {
      agent,
      status: 'open',
      page: 1,
      page_size: INSIGHT_PAGE_SIZE,
      sort: '-created_at',
    },
    { query: { enabled: enabled && !!workspace && !!agent } }
  );

  const insights = useMemo(
    () => [...(data?.data ?? [])].sort(byImpact).slice(0, OVERVIEW_INSIGHT_LIMIT),
    [data]
  );

  return {
    insights,
    totalCount: data?.pagination?.total_results ?? insights.length,
    isPending: enabled && (!agent || isPending),
    error,
  };
};
