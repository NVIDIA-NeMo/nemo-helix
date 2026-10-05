// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { KVPair } from '@nemo/common/src/components/KVPair';
import { useListAnnotations } from '@nemo/sdk/generated/platform/annotations';
import {
  AnnotationSortField,
  type EvaluationSessionResponse,
  type FeedbackAnnotationInputValue,
} from '@nemo/sdk/generated/platform/schema';
import { Button, Flex, Spinner, Stack, Text } from '@nvidia/foundations-react-core';
import { SpanListView } from '@studio/components/IntakeDetail/TraceSpanListView';
import { type NoteRequest } from '@studio/components/IntakeDetail/traceSpanShared';
import { useSessionTrajectories } from '@studio/components/IntakeDetail/useSessionTrajectories';
import { Loading } from '@studio/components/Layouts/Loading';
import { runLabel } from '@studio/routes/EvaluationSessionDetailRoute/runLabel';
import { buildSpanHierarchyRows } from '@studio/util/intakeTelemetry';
import { ChevronsDownUp, ChevronsUpDown, CircleAlert } from 'lucide-react';
import { type FC, useCallback, useMemo, useState } from 'react';

const ANNOTATIONS_PAGE_SIZE = 1000;

// ── Metrics helpers ───────────────────────────────────────────────────────────

const fmtNum = (n?: number | null) => {
  if (n == null) return '—';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${Math.round(n / 1_000)}k`;
  return String(Math.round(n));
};

const fmtMs = (ms?: number | null) => {
  if (ms == null) return '—';
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1_000);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
};

const fmtCost = (usd?: number | null) => {
  if (usd == null) return '—';
  if (usd === 0) return '$0';
  return usd < 0.01 ? `$${usd.toFixed(4)}` : `$${usd.toFixed(2)}`;
};

/** Evaluator means in the 0–1 range read as percentages; anything else is a raw scale. */
const fmtScore = (v: number) => (v >= 0 && v <= 1 ? `${Math.round(v * 100)}%` : v.toFixed(2));

/** The per-session metrics shown in the column card, in display order. */
const sessionMetrics = (run: EvaluationSessionResponse | undefined) => [
  { label: 'Cost', value: fmtCost(run?.cost_total_usd) },
  { label: 'Latency', value: fmtMs(run?.latency_ms) },
  { label: 'Tkns In', value: fmtNum(run?.input_tokens) },
  { label: 'Tkns Out', value: fmtNum(run?.output_tokens) },
  { label: 'Cached Tkns', value: fmtNum(run?.cached_tokens) },
  ...Object.entries(run?.evaluator_scores ?? {}).map(([name, score]) => ({
    label: name,
    value: fmtScore(score),
  })),
  ...(run?.failed_evaluators ?? [])
    .filter((name) => run?.evaluator_scores?.[name] == null)
    .map((name) => ({ label: name, value: 'failed' })),
];

// ── Column ────────────────────────────────────────────────────────────────────

interface SessionCompareColumnProps {
  workspace: string;
  sessionId: string;
  /** The run's session-summary row (metrics + label); undefined while runs load. */
  run: EvaluationSessionResponse | undefined;
}

/**
 * One column of the test-case comparison: this run's label + expand/collapse
 * controls and per-session metrics, then every span of the session as a
 * collapsible accordion row (Attributes / Evaluation Context per span).
 *
 * Self-contained on purpose: the accordion open-state is local so the two columns
 * never fight over shared URL params the way the interactive single-session view
 * (which drives selection through ?spanId) would.
 */
export const SessionCompareColumn: FC<SessionCompareColumnProps> = ({
  workspace,
  sessionId,
  run,
}) => {
  const { session, sessionError, isSessionLoading, isTracesLoading, trajectories, explorer } =
    useSessionTrajectories(workspace, sessionId);

  const spanRows = useMemo(
    () => buildSpanHierarchyRows(trajectories.flatMap((trajectory) => trajectory.spans)),
    [trajectories]
  );

  const [openSpanIds, setOpenSpanIds] = useState<string[]>([]);
  const [noteRequest, setNoteRequest] = useState<NoteRequest>(null);

  const expandAll = useCallback(
    () => setOpenSpanIds(spanRows.map((span) => span.span_id)),
    [spanRows]
  );
  const collapseAll = useCallback(() => setOpenSpanIds([]), []);

  const handleAddNote = useCallback((spanId: string) => {
    setOpenSpanIds((open) => (open.includes(spanId) ? open : [...open, spanId]));
    setNoteRequest((prev) => ({ spanId, nonce: (prev?.nonce ?? 0) + 1 }));
  }, []);

  // One query for the session's annotations so each row can show its feedback
  // sentiment and note/annotation count. Newest-first; keep the latest feedback.
  const { data: annotationsResponse } = useListAnnotations(workspace, {
    page: 1,
    page_size: ANNOTATIONS_PAGE_SIZE,
    sort: AnnotationSortField['-created_at'],
    filter: { session_id: sessionId },
  });
  const { feedbackBySpan, annotationCountBySpan, notesBySpan } = useMemo(() => {
    const feedback = new Map<string, FeedbackAnnotationInputValue>();
    const counts = new Map<string, number>();
    const notes = new Set<string>();
    for (const annotation of annotationsResponse?.data ?? []) {
      if (!annotation.span_id) continue;
      counts.set(annotation.span_id, (counts.get(annotation.span_id) ?? 0) + 1);
      if (annotation.kind === 'note') notes.add(annotation.span_id);
      if (annotation.kind === 'feedback' && !feedback.has(annotation.span_id)) {
        feedback.set(annotation.span_id, annotation.value);
      }
    }
    return { feedbackBySpan: feedback, annotationCountBySpan: counts, notesBySpan: notes };
  }, [annotationsResponse]);

  const header = (
    <Stack gap="density-md">
      <Flex align="center" justify="between" gap="density-sm">
        <Text kind="title/sm">{run ? runLabel(run) : `Trial ${sessionId.slice(-5)}`}</Text>
        <Flex align="center" gap="density-xs">
          <Button
            kind="tertiary"
            size="tiny"
            type="button"
            aria-label="Collapse all"
            title="Collapse all"
            onClick={collapseAll}
            disabled={spanRows.length === 0}
          >
            <ChevronsDownUp size={14} aria-hidden />
          </Button>
          <Button
            kind="tertiary"
            size="tiny"
            type="button"
            aria-label="Expand all"
            title="Expand all"
            onClick={expandAll}
            disabled={spanRows.length === 0}
          >
            <ChevronsUpDown size={14} aria-hidden />
          </Button>
        </Flex>
      </Flex>
      <div className="rounded-sm border border-base bg-surface-raised p-density-lg">
        <Flex align="stretch" gap="density-2xl" className="flex-wrap">
          {sessionMetrics(run).map((m) => (
            <KVPair key={m.label} label={m.label} value={m.value} orientation="vertical" />
          ))}
        </Flex>
      </div>
    </Stack>
  );

  let body: React.ReactNode;
  if (isSessionLoading || isTracesLoading) {
    body = <Loading description="Loading run…" />;
  } else if (sessionError?.response?.status === 404 || !session) {
    body = (
      <div className="flex flex-col items-center gap-2 p-density-2xl text-center">
        <CircleAlert className="h-8 w-8 text-feedback-warning" aria-hidden />
        <p className="text-sm text-secondary">This run’s session could not be loaded.</p>
      </div>
    );
  } else if (explorer.spansError) {
    body = (
      <div className="min-h-[200px] p-density-lg">
        <ErrorMessage message={getErrorMessage(explorer.spansError)} />
      </div>
    );
  } else if (explorer.isSpansFetching && spanRows.length === 0) {
    body = (
      <Flex align="center" justify="center" className="min-h-[200px]">
        <Spinner size="medium" aria-label="Loading spans" />
      </Flex>
    );
  } else if (spanRows.length === 0) {
    body = (
      <Text kind="body/regular/sm" className="text-secondary p-density-lg">
        No spans were found for this run.
      </Text>
    );
  } else {
    body = (
      <SpanListView
        spanRows={spanRows}
        workspace={workspace}
        openSpanIds={openSpanIds}
        onValueChange={setOpenSpanIds}
        banner={null}
        feedbackBySpan={feedbackBySpan}
        annotationCountBySpan={annotationCountBySpan}
        notesBySpan={notesBySpan}
        noteRequest={noteRequest}
        onAddNote={handleAddNote}
      />
    );
  }

  return (
    <Stack gap="density-lg" padding="density-lg" className="min-w-0">
      {header}
      {body}
    </Stack>
  );
};
