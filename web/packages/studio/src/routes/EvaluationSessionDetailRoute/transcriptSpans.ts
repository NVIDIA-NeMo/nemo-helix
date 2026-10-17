// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type Span, SpanKind } from '@nemo/sdk/generated/platform/schema';

const spanDepth = (span: Span, spansById: ReadonlyMap<string, Span>): number => {
  let depth = 0;
  let parentId = span.parent_span_id;
  while (parentId && spansById.has(parentId)) {
    depth += 1;
    parentId = spansById.get(parentId)?.parent_span_id;
  }
  return depth;
};

// Wrappers such as Fabric's DeepAgent nest the real agent inside an outer agent whose output is
// only the hand-off, so the reply is the deepest agent that finished.
export const findReplySpan = (spans: readonly Span[]): Span | undefined => {
  const spansById = new Map(spans.map((span) => [span.span_id, span]));
  const finishedAgents = spans.filter((span) => span.kind === SpanKind.AGENT && span.ended_at);
  if (finishedAgents.length === 0) {
    return spans.find((span) => !span.parent_span_id);
  }
  return finishedAgents.reduce((best, span) => {
    const depthDifference = spanDepth(span, spansById) - spanDepth(best, spansById);
    if (depthDifference !== 0) {
      return depthDifference > 0 ? span : best;
    }
    return Date.parse(span.ended_at ?? '') > Date.parse(best.ended_at ?? '') ? span : best;
  });
};

// A tool call made alongside a sub-agent span (same parent) is the hand-off to that agent, not a
// tool the agent used.
export const findToolCallSpans = (spans: readonly Span[]): Span[] => {
  const agentParentIds = new Set(
    spans.filter((span) => span.kind === SpanKind.AGENT).map((span) => span.parent_span_id)
  );
  return spans.filter(
    (span) => span.kind === SpanKind.TOOL && !agentParentIds.has(span.parent_span_id)
  );
};
