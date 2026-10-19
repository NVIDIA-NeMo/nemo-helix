// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FeedbackAnnotationInputValue } from '@nemo/sdk/generated/platform/schema';
import { TraceDetailLayout } from '@studio/components/IntakeDetail/TraceDetailLayout';
import { TraceSpanTree } from '@studio/components/IntakeDetail/TraceDetailSpanTree';
import { TraceSelectedSpanPanel } from '@studio/components/IntakeDetail/TraceSelectedSpanPanel';
import type { SessionTrajectory, SpanTableRow } from '@studio/util/intakeTelemetry';
import type { FC, ReactNode } from 'react';

interface SpanTreeViewProps {
  trajectories: SessionTrajectory[];
  activeTraceId: string;
  selectedSpan: SpanTableRow | undefined;
  workspace: string;
  sessionDurationMs?: number;
  sessionErrored: boolean;
  activeSpanId: string | null;
  onSelectSpan: (spanId: string, traceId: string) => void;
  onSelectTrace: (traceId: string) => void;
  onSelectSession: () => void;
  banner: ReactNode;
  expandToken: number;
  collapseToken: number;
  activeFeedback?: FeedbackAnnotationInputValue;
  annotationCount?: number;
  hasNotes?: boolean;
  focusNoteNonce?: number;
  onAddNote: () => void;
  emptyContent?: ReactNode;
}

/** Tree view: trajectory tree on the left, the selected span on the right. */
export const SpanTreeView: FC<SpanTreeViewProps> = ({
  trajectories,
  activeTraceId,
  selectedSpan,
  workspace,
  sessionDurationMs,
  sessionErrored,
  activeSpanId,
  onSelectSpan,
  onSelectTrace,
  onSelectSession,
  banner,
  expandToken,
  collapseToken,
  activeFeedback,
  annotationCount,
  hasNotes,
  focusNoteNonce,
  onAddNote,
  emptyContent,
}) => (
  <TraceDetailLayout
    navigation={
      <TraceSpanTree
        trajectories={trajectories}
        activeTraceId={activeTraceId}
        sessionDurationMs={sessionDurationMs}
        sessionErrored={sessionErrored}
        activeSpanId={activeSpanId}
        onSelectSpan={onSelectSpan}
        onSelectTrace={onSelectTrace}
        onSelectSession={onSelectSession}
      />
    }
  >
    <TraceSelectedSpanPanel
      workspace={workspace}
      selectedSpan={selectedSpan}
      banner={banner}
      expandToken={expandToken}
      collapseToken={collapseToken}
      activeFeedback={activeFeedback}
      annotationCount={annotationCount}
      hasNotes={hasNotes}
      focusNoteNonce={focusNoteNonce}
      onAddNote={onAddNote}
      emptyContent={emptyContent}
    />
  </TraceDetailLayout>
);
