// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FeedbackAnnotationInputValue } from '@nemo/sdk/generated/platform/schema';
import { Flex, Text } from '@nvidia/foundations-react-core';
import { SpanFeedbackControls } from '@studio/components/IntakeDetail/IntakeComponents/SpanFeedbackControls';
import { SpanTriggerLabel } from '@studio/components/IntakeDetail/IntakeComponents/SpanTriggerLabel';
import { SpanTriggerMeta } from '@studio/components/IntakeDetail/IntakeComponents/SpanTriggerMeta';
import { TraceSpanAccordionContent } from '@studio/components/IntakeDetail/TraceSpanAccordionContent';
import type { SpanTableRow } from '@studio/util/intakeTelemetry';
import type { FC, ReactNode } from 'react';

interface TraceSelectedSpanPanelProps {
  workspace: string;
  selectedSpan: SpanTableRow | undefined;
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

export const TraceSelectedSpanPanel: FC<TraceSelectedSpanPanelProps> = ({
  workspace,
  selectedSpan,
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
  <>
    {banner}
    <div className="min-w-0 overflow-hidden rounded-lg bg-surface-raised">
      {selectedSpan ? (
        <>
          <Flex
            align="center"
            gap="density-lg"
            className="border-b border-base px-density-lg py-density-md min-w-0"
          >
            <Flex align="center" gap="density-sm" className="min-w-0 flex-1">
              <SpanTriggerLabel span={selectedSpan} showHierarchy={false} />
            </Flex>
            <Flex align="center" gap="density-lg" className="shrink-0">
              <SpanTriggerMeta span={selectedSpan} />
              <SpanFeedbackControls
                workspace={workspace}
                spanId={selectedSpan.span_id}
                sessionId={selectedSpan.session_id}
                activeFeedback={activeFeedback}
                hasNotes={hasNotes}
                onAddNote={onAddNote}
              />
            </Flex>
          </Flex>
          <div className="p-density-lg">
            <TraceSpanAccordionContent
              workspace={workspace}
              spanId={selectedSpan.span_id}
              summarySpan={selectedSpan}
              expandToken={expandToken}
              collapseToken={collapseToken}
              annotationCount={annotationCount}
              focusNoteNonce={focusNoteNonce}
            />
          </div>
        </>
      ) : (
        (emptyContent ?? (
          <Text kind="body/regular/sm" className="text-secondary p-density-lg">
            Select a span to view its details.
          </Text>
        ))
      )}
    </div>
  </>
);
