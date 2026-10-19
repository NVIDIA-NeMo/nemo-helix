// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import {
  Button,
  Card,
  Flex,
  SelectContent,
  SelectItem,
  SelectListbox,
  SelectRoot,
  SelectTrigger,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { TraceStatisticsChart } from '@studio/components/AgentTraceStatistics/TraceStatisticsChart';
import { TraceStatisticsEmptyState } from '@studio/components/AgentTraceStatistics/TraceStatisticsEmptyState';
import { TraceStatisticsTiles } from '@studio/components/AgentTraceStatistics/TraceStatisticsTiles';
import type {
  TraceStatisticsBucket,
  TraceStatisticsRange,
  TraceStatisticsSummary,
} from '@studio/components/AgentTraceStatistics/types';
import { RANGE_LABELS } from '@studio/components/AgentTraceStatistics/utils';
import { AxiosError } from 'axios';
import { ListTree } from 'lucide-react';
import { type FC } from 'react';

const RANGE_OPTIONS: TraceStatisticsRange[] = ['day', 'week', 'month', 'max'];

const isTraceStatisticsRange = (value: unknown): value is TraceStatisticsRange =>
  typeof value === 'string' && (RANGE_OPTIONS as string[]).includes(value);

const TRACE_STORE_UNREACHABLE_MESSAGE =
  "The trace store couldn't be reached, so trace statistics can't be loaded right now. This " +
  'usually means ClickHouse is down — check that it is running and reachable, then refresh.';

const traceStatisticsErrorMessage = (error: unknown): string => {
  if (error instanceof AxiosError || error instanceof Error) {
    return getErrorMessage(error, TRACE_STORE_UNREACHABLE_MESSAGE);
  }
  return TRACE_STORE_UNREACHABLE_MESSAGE;
};

export interface AgentTraceStatisticsProps {
  /**
   * Headline numbers for the selected range, or `null` when the range saw no runs. The caller owns
   * the query, so changing `range` refetches rather than re-slicing a cached superset.
   */
  summary: TraceStatisticsSummary | null;
  /** Trend points for the same range, gaps already filled. */
  buckets: TraceStatisticsBucket[];
  range: TraceStatisticsRange;
  onRangeChange: (range: TraceStatisticsRange) => void;
  /** Omit to hide the "View traces" action. */
  onViewTraces?: () => void;
  /** Empty-state action: invoke the agent so it emits its first traces. */
  onRunAgent?: () => void;
  /** Empty-state action: open the tracing setup docs. */
  onLearnMore?: () => void;
  /** Muted note after the heading, e.g. how the traces are scoped. */
  caption?: string;
  isPending?: boolean;
  chartHeight?: number;
  /** When set, replaces the tiles/chart with a warning — e.g. the trace store is unreachable. */
  error?: unknown;
}

export const AgentTraceStatistics: FC<AgentTraceStatisticsProps> = ({
  summary,
  buckets,
  range,
  onRangeChange,
  onViewTraces,
  onRunAgent,
  onLearnMore,
  caption,
  isPending,
  chartHeight,
  error,
}) => {
  const isEmpty = !error && !isPending && (summary === null || summary.totalTraces === 0);

  return (
    <Stack gap="4">
      <Flex justify="between" align="center" gap="2" wrap="wrap">
        <Flex align="center" gap="3">
          <Text kind="title/md">Trace statistics</Text>
          {caption ? (
            <Text kind="body/regular/md" className="text-secondary">
              {caption}
            </Text>
          ) : null}
        </Flex>
        {error ? null : (
          <Flex gap="2" align="center">
            {onViewTraces && !isEmpty ? (
              <Button kind="tertiary" onClick={onViewTraces}>
                <ListTree size={16} aria-hidden />
                View traces
              </Button>
            ) : null}
            <SelectRoot
              value={range}
              onValueChange={(value: string) => onRangeChange(value as TraceStatisticsRange)}
            >
              <SelectTrigger
                aria-label="Statistics range"
                className="w-32"
                renderValue={(value) =>
                  isTraceStatisticsRange(value) ? RANGE_LABELS[value] : undefined
                }
              />
              <SelectContent className="min-w-40">
                <SelectListbox>
                  {RANGE_OPTIONS.map((option) => (
                    <SelectItem key={option} value={option}>
                      {RANGE_LABELS[option]}
                    </SelectItem>
                  ))}
                </SelectListbox>
              </SelectContent>
            </SelectRoot>
          </Flex>
        )}
      </Flex>

      {error ? (
        <Card>
          <Flex justify="center" padding="density-2xl">
            <ErrorMessage
              header="Trace statistics are unavailable"
              message={traceStatisticsErrorMessage(error)}
              height="auto"
            />
          </Flex>
        </Card>
      ) : isEmpty ? (
        <TraceStatisticsEmptyState
          onRunAgent={onRunAgent}
          onLearnMore={onLearnMore}
          onExpandRange={range === 'max' ? undefined : () => onRangeChange('max')}
        />
      ) : (
        <>
          <TraceStatisticsTiles summary={summary} isPending={isPending} />
          <TraceStatisticsChart
            buckets={buckets}
            range={range}
            isPending={isPending}
            height={chartHeight}
          />
        </>
      )}
    </Stack>
  );
};
