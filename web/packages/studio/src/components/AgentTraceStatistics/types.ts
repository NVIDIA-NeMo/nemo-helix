// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Window the statistics cover. Drives both the header label and the bucket size. `max` reaches
 * Intake's full retention window (90 days of ClickHouse TTL), for finding traces older than a month.
 */
export type TraceStatisticsRange = 'day' | 'week' | 'month' | 'max';

/**
 * Headline numbers for the range, as returned by Intake's `bucket=total` rollup. The caller owns
 * the aggregation, so stories and tests can build a summary without a trace payload.
 */
export interface TraceStatisticsSummary {
  totalTraces: number;
  /** Mean wall-clock duration of a run. */
  avgLatencyMs: number;
  avgTokensPerRun: number;
  avgCostUsd: number;
}

/** One point on the trend chart. `null` renders a gap rather than a drop to zero. */
export interface TraceStatisticsBucket {
  /** Bucket start, epoch milliseconds. */
  timestamp: number;
  costUsd: number | null;
  tokens: number | null;
  latencyMs: number | null;
}
