// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type Span, SpanStatus, type Trace } from '@nemo/sdk/generated/platform/schema';
import { useGetSession } from '@nemo/sdk/generated/platform/sessions';
import { useListSpans } from '@nemo/sdk/generated/platform/spans';
import { useListTraces } from '@nemo/sdk/generated/platform/traces';
import { type SessionExplorerData } from '@studio/components/IntakeDetail/TraceSpanAccordions';
import {
  buildSpanTree,
  compareSpansByStartedAt,
  type SessionTrajectory,
} from '@studio/util/intakeTelemetry';
import { useMemo } from 'react';

const SESSION_TRACES_PAGE_SIZE = 1000;

/**
 * Loads a session's summary, its traces, and their span trajectories — plus the
 * explorer bundle the span views need and the session's producer test case name.
 *
 * Shared by the single session detail view and the test-case comparison columns;
 * React Query dedupes the fetches when the same session renders in more than one
 * place (e.g. the primary column and the surrounding route both read it).
 */
export function useSessionTrajectories(workspace: string, sessionId: string) {
  const {
    data: session,
    error: sessionError,
    isLoading: isSessionLoading,
  } = useGetSession(workspace, sessionId);

  const {
    data: tracesResponse,
    error: tracesError,
    isLoading: isTracesLoading,
  } = useListTraces(workspace, {
    filter: { session_id: sessionId },
    mode: 'summary',
    page: 1,
    page_size: SESSION_TRACES_PAGE_SIZE,
    sort: 'started_at',
  });
  const traces = useMemo(() => tracesResponse?.data ?? [], [tracesResponse?.data]);

  const {
    data: sessionSpansResponse,
    error: sessionSpansError,
    isFetching: isSessionSpansFetching,
  } = useListSpans(workspace, {
    filter: { session_id: sessionId },
    mode: 'summary',
    page: 1,
    page_size: SESSION_TRACES_PAGE_SIZE,
    sort: 'started_at',
  });

  const trajectories = useMemo<SessionTrajectory[]>(() => {
    const groupedSpans = new Map<string, Span[]>();
    for (const span of sessionSpansResponse?.data ?? []) {
      if (!span.trace_id) continue;
      const traceSpans = groupedSpans.get(span.trace_id) ?? [];
      traceSpans.push(span);
      groupedSpans.set(span.trace_id, traceSpans);
    }

    const traceSummaries: Trace[] = [...traces];
    const knownTraceIds = new Set(traces.map((trace) => trace.id));
    for (const [traceId, spans] of groupedSpans) {
      if (knownTraceIds.has(traceId)) continue;
      const earliestSpan = spans.reduce((earliest, span) =>
        compareSpansByStartedAt(span, earliest) < 0 ? span : earliest
      );
      traceSummaries.push({
        id: traceId,
        session_id: sessionId,
        workspace,
        started_at: earliestSpan.started_at,
        status: SpanStatus.unknown,
        span_count: spans.length,
      });
    }

    traceSummaries.sort((a, b) => {
      const startedAtDifference = Date.parse(a.started_at) - Date.parse(b.started_at);
      return startedAtDifference || a.id.localeCompare(b.id);
    });
    return traceSummaries.map((trace) => {
      const spans = groupedSpans.get(trace.id) ?? [];
      return { trace, spans, spanTree: buildSpanTree(spans) };
    });
  }, [sessionId, sessionSpansResponse?.data, traces, workspace]);

  const explorer = useMemo<SessionExplorerData>(
    () => ({
      trajectories,
      spansLoaded: sessionSpansResponse !== undefined,
      spansError: sessionSpansError,
      isSpansFetching: isSessionSpansFetching,
    }),
    [isSessionSpansFetching, sessionSpansError, sessionSpansResponse, trajectories]
  );

  const testCaseName = traces.find((trace) => trace.evaluation_context?.test_case_name)
    ?.evaluation_context?.test_case_name;

  return {
    session,
    sessionError,
    isSessionLoading,
    traces,
    tracesError,
    isTracesLoading,
    trajectories,
    explorer,
    testCaseName,
  };
}
