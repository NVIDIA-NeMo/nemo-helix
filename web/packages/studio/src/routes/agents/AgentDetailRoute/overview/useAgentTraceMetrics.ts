// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { TraceFilter, TraceMetricPointResponse } from '@nemo/sdk/generated/platform/schema';
import { useGetTraceMetrics } from '@nemo/sdk/generated/platform/traces';
import type {
  TraceStatisticsBucket,
  TraceStatisticsRange,
  TraceStatisticsSummary,
} from '@studio/components/AgentTraceStatistics/types';
import { bucketParamForRange, fillBucketGaps } from '@studio/components/AgentTraceStatistics/utils';
import { keepPreviousData } from '@tanstack/react-query';
import { useMemo } from 'react';

const RANGE_MS: Record<TraceStatisticsRange, number> = {
  day: 24 * 60 * 60 * 1000,
  week: 7 * 24 * 60 * 60 * 1000,
  month: 30 * 24 * 60 * 60 * 1000,
  max: 90 * 24 * 60 * 60 * 1000,
};

/** Buckets align to the viewer's day, not UTC, so "yesterday" on the chart is their yesterday. */
const browserTimezone = (): string => Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';

/**
 * Intake returns naive UTC timestamps on some payloads (`2026-08-17T17:42:40.999490`). `new Date`
 * reads a string with no zone designator as *local* time, which slides every point by the browser's
 * offset — so pin it to UTC unless the server already said otherwise.
 */
const parseTimestamp = (value: string): number =>
  new Date(value.endsWith('Z') || /[+-]\d{2}:\d{2}$/.test(value) ? value : `${value}Z`).getTime();

const finite = (value: number | null | undefined): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

const toSummary = (point: TraceMetricPointResponse | undefined): TraceStatisticsSummary | null => {
  if (!point || point.run_count === 0) return null;
  return {
    totalTraces: point.run_count,
    avgLatencyMs: finite(point.latency_ms.mean) ?? 0,
    avgTokensPerRun: finite(point.total_tokens.mean) ?? 0,
    avgCostUsd: finite(point.cost_usd.mean) ?? 0,
  };
};

const toBucket = (point: TraceMetricPointResponse): TraceStatisticsBucket | null => {
  if (!point.bucket_start) return null;
  const timestamp = parseTimestamp(point.bucket_start);
  if (!Number.isFinite(timestamp)) return null;
  return {
    timestamp,
    costUsd: finite(point.cost_usd.mean),
    tokens: finite(point.total_tokens.mean),
    latencyMs: finite(point.latency_ms.mean),
  };
};

interface UseAgentTraceMetricsParams {
  workspace: string;
  /** Scopes the rollup to one agent via the root span's agent name. */
  agentName?: string;
  range: TraceStatisticsRange;
  enabled: boolean;
}

interface UseAgentTraceMetricsResult {
  summary: TraceStatisticsSummary | null;
  buckets: TraceStatisticsBucket[];
  isPending: boolean;
  /** `summary` is still the previous range's data, kept on screen while the new range loads. */
  isPlaceholderData: boolean;
  error: unknown;
}

/**
 * Trace rollups backing the overview statistics, aggregated by Intake rather than in the browser.
 *
 * Two calls rather than one: `bucket=total` gives exact headline numbers for the whole range, which
 * averaging the per-bucket means would only approximate, and the bucketed call feeds the trend.
 */
export const useAgentTraceMetrics = ({
  workspace,
  agentName,
  range,
  enabled,
}: UseAgentTraceMetricsParams): UseAgentTraceMetricsResult => {
  const filter = useMemo<TraceFilter>(() => {
    const now = Date.now();
    const minute = 60 * 1000;
    const since = new Date(Math.floor(now / minute) * minute - RANGE_MS[range]).toISOString();
    return { started_at: { $gte: since }, ...(agentName ? { agent_name: agentName } : {}) };
  }, [range, agentName]);

  const isEnabled = enabled && !!workspace && !!agentName;
  const timezone = browserTimezone();
  const queryOptions = { query: { enabled: isEnabled, placeholderData: keepPreviousData } };

  const {
    data: totals,
    isPending: isTotalsPending,
    isPlaceholderData: isTotalsPlaceholder,
    error: totalsError,
  } = useGetTraceMetrics(workspace, { bucket: 'total', timezone, filter }, queryOptions);

  const {
    data: series,
    isPending: isSeriesPending,
    error: seriesError,
  } = useGetTraceMetrics(
    workspace,
    { bucket: bucketParamForRange(range), timezone, filter },
    queryOptions
  );

  const summary = useMemo(() => toSummary(totals?.data[0]), [totals]);

  const buckets = useMemo(
    () =>
      fillBucketGaps(
        (series?.data ?? []).map(toBucket).filter((b) => b !== null),
        range
      ),
    [series, range]
  );

  return {
    summary,
    buckets,
    isPending: isEnabled ? isTotalsPending || isSeriesPending : enabled,
    isPlaceholderData: isEnabled && isTotalsPlaceholder,
    error: isEnabled ? (totalsError ?? seriesError) : null,
  };
};
