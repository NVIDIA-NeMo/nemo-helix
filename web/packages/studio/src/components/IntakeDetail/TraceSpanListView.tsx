// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { IntakeAccordion } from '@nemo/common/src/components/IntakeAccordion';
import type { FeedbackAnnotationInputValue } from '@nemo/sdk/generated/platform/schema';
import { Stack } from '@nvidia/foundations-react-core';
import { SpanFeedbackControls } from '@studio/components/IntakeDetail/IntakeComponents/SpanFeedbackControls';
import { SpanTriggerLabel } from '@studio/components/IntakeDetail/IntakeComponents/SpanTriggerLabel';
import { SpanTriggerMeta } from '@studio/components/IntakeDetail/IntakeComponents/SpanTriggerMeta';
import { TraceSpanAccordionContent } from '@studio/components/IntakeDetail/TraceSpanAccordionContent';
import {
  type NoteRequest,
  noteFocusNonce,
  spanAccordionId,
} from '@studio/components/IntakeDetail/traceSpanShared';
import type { SpanTableRow } from '@studio/util/intakeTelemetry';
import type { FC, ReactNode } from 'react';

interface SpanListViewProps {
  spanRows: SpanTableRow[];
  workspace: string;
  openSpanIds: string[];
  onValueChange: (next: string[]) => void;
  banner: ReactNode;
  feedbackBySpan: Map<string, FeedbackAnnotationInputValue>;
  annotationCountBySpan: Map<string, number>;
  notesBySpan: ReadonlySet<string>;
  noteRequest: NoteRequest;
  onAddNote: (spanId: string) => void;
}

/** List view: every span as a collapsible accordion row (no tree). */
export const SpanListView: FC<SpanListViewProps> = ({
  spanRows,
  workspace,
  openSpanIds,
  onValueChange,
  banner,
  feedbackBySpan,
  annotationCountBySpan,
  notesBySpan,
  noteRequest,
  onAddNote,
}) => (
  <Stack gap="density-lg" className="min-w-0">
    {banner}
    <div className="min-w-0 overflow-hidden rounded-lg bg-surface-raised">
      <IntakeAccordion
        variant="row"
        value={openSpanIds}
        onValueChange={onValueChange}
        items={spanRows.map((span) => ({
          value: span.span_id,
          id: spanAccordionId(span.span_id),
          slotLabel: <SpanTriggerLabel span={span} />,
          slotEnd: (
            <>
              <SpanTriggerMeta span={span} />
              <SpanFeedbackControls
                workspace={workspace}
                spanId={span.span_id}
                sessionId={span.session_id}
                activeFeedback={feedbackBySpan.get(span.span_id)}
                hasNotes={notesBySpan.has(span.span_id)}
                onAddNote={() => onAddNote(span.span_id)}
              />
            </>
          ),
          slotContent: openSpanIds.includes(span.span_id) ? (
            <TraceSpanAccordionContent
              workspace={workspace}
              spanId={span.span_id}
              sessionId={span.session_id}
              summarySpan={span}
              annotationCount={annotationCountBySpan.get(span.span_id)}
              focusNoteNonce={noteFocusNonce(noteRequest, span.span_id)}
            />
          ) : null,
        }))}
      />
    </div>
  </Stack>
);
