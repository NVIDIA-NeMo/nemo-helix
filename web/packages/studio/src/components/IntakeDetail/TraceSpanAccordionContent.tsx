// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Span } from '@nemo/sdk/generated/platform/schema';
import { useGetSpan } from '@nemo/sdk/generated/platform/spans';
import { Spinner, Stack, StatusMessage } from '@nvidia/foundations-react-core';
import { mergeSpanDetails } from '@studio/components/IntakeDetail/IntakeComponents/spanKeyValues';
import { SpanMetadataAccordions } from '@studio/components/IntakeDetail/SpanMetadataAccordions';
import { CircleAlert } from 'lucide-react';
import { type FC, useMemo } from 'react';

interface TraceSpanAccordionContentProps {
  workspace: string;
  spanId: string;
  summarySpan?: Span;
  /** Forwarded to the span's sections so a toolbar can expand/collapse them all. */
  expandToken?: number;
  collapseToken?: number;
  /** Annotation count for the span, shown on the Annotations section trigger. */
  annotationCount?: number;
  /** Bumped to open the annotations section and focus its note field. */
  focusNoteNonce?: number;
}

/** Loads full span detail when an accordion section is expanded. */
export const TraceSpanAccordionContent: FC<TraceSpanAccordionContentProps> = ({
  workspace,
  spanId,
  summarySpan,
  expandToken,
  collapseToken,
  annotationCount,
  focusNoteNonce,
}) => {
  const { data: detailSpan, error, isLoading } = useGetSpan(workspace, spanId);
  const span = useMemo(
    () => (detailSpan ? mergeSpanDetails(summarySpan, detailSpan) : summarySpan),
    [detailSpan, summarySpan]
  );

  if (isLoading && !detailSpan) {
    return (
      <Stack
        gap="density-md"
        padding="density-xl"
        className="items-center justify-center min-h-[200px]"
      >
        <Spinner size="medium" aria-label="Loading span details" />
      </Stack>
    );
  }

  if (error && !span) {
    return (
      <StatusMessage
        size="small"
        slotMedia={<CircleAlert width={40} height={40} />}
        slotHeading="Error loading span"
        slotSubheading={error.message}
      />
    );
  }

  if (!span) {
    return null;
  }

  return (
    <Stack className="min-w-0">
      <SpanMetadataAccordions
        span={span}
        workspace={workspace}
        expandToken={expandToken}
        collapseToken={collapseToken}
        annotationCount={annotationCount}
        focusNoteNonce={focusNoteNonce}
      />
    </Stack>
  );
};
