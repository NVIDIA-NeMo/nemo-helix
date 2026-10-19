// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@nvidia/foundations-react-core';
import { getSpanTemplate } from '@studio/components/IntakeDetail/SpanTemplates/registry';
import { SpanKindBadge } from '@studio/components/SpanKindBadge';
import {
  getSpanDisplayName,
  getSpanSubject,
  type SpanTableRow,
} from '@studio/util/intakeTelemetry';
import type { FC } from 'react';

const HIERARCHY_SPACER_LIMIT = 12;

/** Span name/kind/subject label shared by the tree and list span views. */
export const SpanTriggerLabel: FC<{ span: SpanTableRow; showHierarchy?: boolean }> = ({
  span,
  showHierarchy = true,
}) => {
  const depth = span.hierarchyDepth;
  const hierarchyLabel =
    !showHierarchy || span.hierarchyStatus === undefined
      ? undefined
      : span.hierarchyStatus === 'parent_outside_page'
        ? 'Parent outside page'
        : 'Unresolved hierarchy';

  return (
    <>
      {showHierarchy &&
        Array.from({ length: Math.min(depth, HIERARCHY_SPACER_LIMIT) }).map((_, index) => (
          <span
            key={`${span.span_id}-hierarchy-spacer-${index}`}
            aria-hidden
            className="w-[18px] shrink-0"
          />
        ))}
      {showHierarchy && depth > 0 && (
        <span aria-hidden className="relative h-5 w-5 shrink-0">
          <span className="absolute left-0 top-1/2 w-full border-t border-base" />
          <span className="absolute left-0 top-0 h-1/2 border-l border-base" />
        </span>
      )}
      <Text
        kind="body/semibold/sm"
        className="min-w-0 shrink truncate font-mono"
        title={getSpanTemplate(span.kind).headerTitle?.(span) ?? getSpanDisplayName(span)}
      >
        {getSpanTemplate(span.kind).headerTitle?.(span) ?? getSpanDisplayName(span)}
      </Text>
      <SpanKindBadge kind={span.kind} />
      <Text
        kind="body/regular/sm"
        className="min-w-8 flex-1 truncate text-secondary"
        title={getSpanSubject(span)}
      >
        {getSpanSubject(span)}
      </Text>
      {hierarchyLabel && (
        <Text kind="body/regular/xs" className="shrink-0 text-secondary">
          {hierarchyLabel}
        </Text>
      )}
    </>
  );
};
