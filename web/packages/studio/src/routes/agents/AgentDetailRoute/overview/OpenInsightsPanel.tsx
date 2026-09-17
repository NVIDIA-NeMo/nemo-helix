// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { InsightListItem } from '@nemo/sdk/generated/insights/schema';
import { Button, Flex, Skeleton, Stack, Text } from '@nvidia/foundations-react-core';
import { DetailPanel } from '@studio/routes/agents/AgentDetailRoute/overview/DetailPanel';
import { OpenInsightRow } from '@studio/routes/agents/AgentDetailRoute/overview/OpenInsightRow';
import type { FC } from 'react';

interface OpenInsightsPanelProps {
  readonly insights: InsightListItem[];
  /** Open insights for the agent overall, which can exceed the rendered slice. */
  readonly totalCount: number;
  readonly isPending?: boolean;
  readonly hasError?: boolean;
  readonly onOpenInsight: (insight: InsightListItem) => void;
  readonly awaitingTelemetry?: boolean;
  /** Omit to hide the "View all" action. */
  readonly onViewAll?: () => void;
}

/**
 * Open insights the analyst has filed against this agent — the agent's outstanding work list.
 *
 * Insights are produced by the NeMo Insights analyst, which reads Intake spans, annotations
 * (user feedback), and evaluator scores on a schedule. This panel only reads them.
 */
export const OpenInsightsPanel: FC<OpenInsightsPanelProps> = ({
  insights,
  totalCount,
  isPending,
  hasError,
  awaitingTelemetry,
  onOpenInsight,
  onViewAll,
}) => {
  const isEmpty = !isPending && !hasError && insights.length === 0;

  return (
    <DetailPanel
      title="Open insights"
      flush
      slotAction={
        hasError || isPending ? null : (
          <Flex gap="3" align="center">
            <Text kind="body/regular/sm" className="text-secondary">
              {`${totalCount} total`}
            </Text>
            {!isEmpty && onViewAll ? (
              <Button kind="tertiary" size="small" onClick={onViewAll}>
                View all
              </Button>
            ) : null}
          </Flex>
        )
      }
    >
      {hasError ? (
        <Flex justify="center" align="center" padding="density-xl" className="min-h-60">
          <Text kind="body/regular/md" className="max-w-72 text-center text-secondary">
            Insights couldn&apos;t be loaded right now.
          </Text>
        </Flex>
      ) : isPending ? (
        <Stack gap="3" className="p-4">
          <Skeleton className="h-5 w-2/3" />
          <Skeleton className="h-5 w-1/2" />
          <Skeleton className="h-5 w-3/5" />
        </Stack>
      ) : isEmpty ? (
        <Flex justify="center" align="center" padding="density-xl" className="min-h-60">
          <Text kind="body/regular/md" className="max-w-72 text-center text-secondary">
            {awaitingTelemetry
              ? 'Generating insights requires importing traces or integrating your agent with NeMo Platform.'
              : "Insights are filed by the analyst from this agent's traces, feedback, and evaluation scores. Once analysis has run, recurring problems show up here with the traces that prove them."}
          </Text>
        </Flex>
      ) : (
        <div>
          {insights.map((insight, index) => (
            <div key={insight.id} className={index > 0 ? 'border-t border-base' : undefined}>
              <OpenInsightRow insight={insight} onOpen={onOpenInsight} />
            </div>
          ))}
        </div>
      )}
    </DetailPanel>
  );
};
