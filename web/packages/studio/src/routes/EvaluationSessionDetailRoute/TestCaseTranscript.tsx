// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatDurationMs } from '@nemo/common/src/utils/date';
import { type Span } from '@nemo/sdk/generated/platform/schema';
import { useGetSpan } from '@nemo/sdk/generated/platform/spans';
import { Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
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
import { type FC, useCallback, useMemo } from 'react';
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
      <Flex align="center" justify="between" gap="density-sm">
        <Text kind="label/semibold/md">Agent reply</Text>
        <SpanPayloadFormatToggle state={outputFormat} payloadLabel="reply" />
      </Flex>
      {isLoading ? (
        <Text kind="body/regular/sm" className="text-secondary">
          Loading the reply…
        </Text>
      ) : (
        <div className="max-h-[360px] overflow-auto">
          <SpanPayloadView
            value={replySpan?.output}
            format={outputFormat.format}
            emptyMessage="No reply was recorded for this test."
          />
        </div>
      )}
    </Stack>
  );
};

const ToolCalls: FC<{ toolSpans: readonly Span[] }> = ({ toolSpans }) => {
  const [, setSearchParams] = useSearchParams();
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
    <Stack gap="density-sm" className="min-w-0">
      <Text kind="label/semibold/md">Tool calls ({toolSpans.length})</Text>
      {toolSpans.length === 0 ? (
        <Text kind="body/regular/sm" className="text-secondary">
          No tool calls were recorded for this test.
        </Text>
      ) : (
        <Flex gap="density-sm" className="flex-wrap">
          {toolSpans.map((span) => (
            <Button key={span.span_id} kind="secondary" size="small" onClick={() => openSpan(span)}>
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

/** The agent's final reply and the tools it called, pulled out of the span tree. */
export const TestCaseTranscript: FC<{
  workspace: string;
  trajectories: readonly SessionTrajectory[];
}> = ({ workspace, trajectories }) => {
  const spans = useMemo(
    () => trajectories.flatMap((trajectory) => trajectory.spans),
    [trajectories]
  );
  const replySpanId = useMemo(() => findReplySpan(spans)?.span_id, [spans]);
  const toolSpans = useMemo(() => findToolCallSpans(spans), [spans]);

  return (
    <Stack
      gap="density-lg"
      className="rounded-sm border border-base bg-surface-raised p-density-lg"
      data-testid="test-case-transcript"
    >
      <AgentReply workspace={workspace} replySpanId={replySpanId} />
      <ToolCalls toolSpans={toolSpans} />
    </Stack>
  );
};
