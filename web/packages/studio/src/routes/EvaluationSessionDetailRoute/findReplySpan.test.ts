// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type Span, SpanKind, SpanStatus } from '@nemo/sdk/generated/platform/schema';
import { findReplySpan } from '@studio/routes/EvaluationSessionDetailRoute/findReplySpan';

const span = (span_id: string, kind: SpanKind, parent_span_id?: string): Span => ({
  span_id,
  kind,
  parent_span_id,
  session_id: 'session',
  workspace: 'default',
  source: 'test',
  started_at: '2026-10-16T00:00:00Z',
  ingested_at: '2026-10-16T00:00:00Z',
  status: SpanStatus.success,
});

describe('findReplySpan', () => {
  it('prefers the outermost agent span', () => {
    const spans = [
      span('wrapper', SpanKind.CHAIN),
      span('agent', SpanKind.AGENT),
      span('llm', SpanKind.LLM, 'agent'),
    ];
    expect(findReplySpan(spans)?.span_id).toBe('agent');
  });

  it('falls back to the first root span', () => {
    const spans = [span('wrapper', SpanKind.CHAIN), span('tool', SpanKind.TOOL, 'wrapper')];
    expect(findReplySpan(spans)?.span_id).toBe('wrapper');
  });

  it('ignores nested agent spans', () => {
    const spans = [span('root', SpanKind.CHAIN), span('agent', SpanKind.AGENT, 'root')];
    expect(findReplySpan(spans)?.span_id).toBe('root');
  });

  it('returns nothing without spans', () => {
    expect(findReplySpan([])).toBeUndefined();
  });
});
