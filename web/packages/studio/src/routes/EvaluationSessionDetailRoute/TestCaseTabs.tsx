// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatDurationMs } from '@nemo/common/src/utils/date';
import { type Span } from '@nemo/sdk/generated/platform/schema';
import { useGetSpan } from '@nemo/sdk/generated/platform/spans';
import {
  Button,
  Flex,
  Stack,
  TabsContent,
  TabsList,
  TabsRoot,
  TabsTrigger,
  Text,
} from '@nvidia/foundations-react-core';
import { PayloadPending } from '@studio/components/IntakeDetail/IntakeComponents/PayloadPending';
import { SpanPayloadFormatToggle } from '@studio/components/IntakeDetail/IntakeComponents/SpanPayloadFormatToggle';
import { SpanPayloadView } from '@studio/components/IntakeDetail/IntakeComponents/SpanPayloadView';
import { useSpanPayloadFormat } from '@studio/components/IntakeDetail/IntakeComponents/useSpanPayloadFormat';
import { QUERY_PARAMETERS } from '@studio/routes/constants';
import {
  findReplySpan,
  findToolCallSpans,
} from '@studio/routes/EvaluationSessionDetailRoute/transcriptSpans';
import type { SessionTrajectory } from '@studio/util/intakeTelemetry';
import { Wrench } from 'lucide-react';
import { type FC, type ReactNode, useCallback, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router';

const spanDuration = (span: Span): string | undefined => {
  if (!span.ended_at) {
    return undefined;
  }
  return formatDurationMs(Date.parse(span.ended_at) - Date.parse(span.started_at));
};

const AgentReply: FC<{ workspace: string; replySpanId: string | undefined }> = ({
  workspace,
  replySpanId,
}) => {
  const { data: replySpan, isLoading } = useGetSpan(workspace, replySpanId ?? '', {
    query: { enabled: Boolean(replySpanId) },
  });
  const outputFormat = useSpanPayloadFormat(replySpan?.output);

  return (
    <Stack gap="density-sm" className="min-w-0">
      <Flex justify="end">
        <SpanPayloadFormatToggle state={outputFormat} payloadLabel="reply" />
      </Flex>
      {isLoading ? (
        <PayloadPending />
      ) : (
        <SpanPayloadView
          value={replySpan?.output}
          format={outputFormat.format}
          emptyMessage="No reply was recorded for this test."
        />
      )}
    </Stack>
  );
};

const ToolCalls: FC<{
  toolSpans: readonly Span[];
  openSpan: (span: Span) => void;
  selectedSpanId: string | null;
}> = ({ toolSpans, openSpan, selectedSpanId }) => {
  return (
    <Stack
      gap="density-xs"
      className="rounded-sm border border-base bg-surface-raised p-density-lg"
      data-testid="test-case-tool-calls"
    >
      <Text kind="label/regular/md" className="text-secondary">
        Tool calls ({toolSpans.length})
      </Text>
      {toolSpans.length === 0 ? (
        <Text kind="body/regular/sm" className="text-secondary">
          No tool calls were recorded for this test.
        </Text>
      ) : (
        <Flex align="center" gap="density-sm" className="flex-wrap">
          {toolSpans.map((span) => (
            <Button
              key={span.span_id}
              kind={span.span_id === selectedSpanId ? 'primary' : 'secondary'}
              size="small"
              onClick={() => openSpan(span)}
            >
              <Wrench size={14} aria-hidden />
              {span.tool_name ?? span.name ?? 'tool'}
              {spanDuration(span) ? (
                <span className="text-secondary">{spanDuration(span)}</span>
              ) : null}
            </Button>
          ))}
        </Flex>
      )}
    </Stack>
  );
};

type TestCaseTab = 'reply' | 'trace';

/** The agent's reply, and the full trace with shortcuts to each tool call. */
export const TestCaseTabs: FC<{
  workspace: string;
  trajectories: readonly SessionTrajectory[];
  traceView: ReactNode;
}> = ({ workspace, trajectories, traceView }) => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [tab, setTab] = useState<TestCaseTab>(() =>
    searchParams.has(QUERY_PARAMETERS.spanId) ? 'trace' : 'reply'
  );
  const spans = useMemo(
    () => trajectories.flatMap((trajectory) => trajectory.spans),
    [trajectories]
  );
  const replySpanId = useMemo(() => findReplySpan(spans)?.span_id, [spans]);
  const toolSpans = useMemo(() => findToolCallSpans(spans), [spans]);

  const openSpan = useCallback(
    (span: Span) => {
      setSearchParams((previous) => {
        const next = new URLSearchParams(previous);
        if (span.trace_id) {
          next.set(QUERY_PARAMETERS.traceId, span.trace_id);
        }
        next.set(QUERY_PARAMETERS.spanId, span.span_id);
        return next;
      });
    },
    [setSearchParams]
  );

  return (
    <TabsRoot value={tab} onValueChange={(value) => setTab(value as TestCaseTab)}>
      <TabsList aria-label="Test case details">
        <TabsTrigger value="reply">Agent reply</TabsTrigger>
        <TabsTrigger value="trace">Trace</TabsTrigger>
      </TabsList>
      <TabsContent value="reply" className="items-stretch p-0 pt-density-lg">
        <AgentReply workspace={workspace} replySpanId={replySpanId} />
      </TabsContent>
      <TabsContent value="trace" className="items-stretch p-0 pt-density-lg">
        <Stack gap="density-2xl" className="min-w-0">
          <ToolCalls
            toolSpans={toolSpans}
            openSpan={openSpan}
            selectedSpanId={searchParams.get(QUERY_PARAMETERS.spanId)}
          />
          {traceView}
        </Stack>
      </TabsContent>
    </TabsRoot>
  );
};
