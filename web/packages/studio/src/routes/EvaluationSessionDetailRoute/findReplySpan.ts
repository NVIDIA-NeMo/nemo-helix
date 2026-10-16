// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type Span, SpanKind } from '@nemo/sdk/generated/platform/schema';

// The agent's final answer is the output of its outermost span; prefer an AGENT span over a wrapper.
export const findReplySpan = (spans: readonly Span[]): Span | undefined => {
  const roots = spans.filter((span) => !span.parent_span_id);
  return roots.find((span) => span.kind === SpanKind.AGENT) ?? roots[0];
};
