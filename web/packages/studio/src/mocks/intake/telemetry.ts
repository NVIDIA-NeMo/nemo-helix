// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  Annotation,
  AnnotationInput,
  AnnotationsPage,
  Session,
  Span,
  SpansPage,
  Trace,
  TraceMetricBucketParam,
  TraceMetricPointResponse,
  TraceMetrics,
  TracesPage,
} from '@nemo/sdk/generated/platform/schema';
import { FeedbackAnnotationInputKind } from '@nemo/sdk/generated/platform/schema/FeedbackAnnotationInputKind';
import { LabelAnnotationInputKind } from '@nemo/sdk/generated/platform/schema/LabelAnnotationInputKind';
import { MetadataAnnotationInputKind } from '@nemo/sdk/generated/platform/schema/MetadataAnnotationInputKind';
import { NoteAnnotationInputKind } from '@nemo/sdk/generated/platform/schema/NoteAnnotationInputKind';

const trace1: Trace = {
  id: 'trace-agent-run-001',
  root_span_id: 'span-root-001',
  session_id: 'session-agent-run-001',
  workspace: 'default',
  name: 'Answer customer policy question',
  input: 'Can I deploy this model in a private workspace?',
  output: 'Yes. Use a private workspace and restrict access through workspace membership.',
  started_at: '2026-05-20T16:42:00Z',
  ended_at: '2026-05-20T16:42:12Z',
  duration_ms: 12_230,
  status: 'success',
  input_tokens: 1240,
  output_tokens: 386,
  cached_tokens: 128,
  total_tokens: 1754,
  cost_usd: 0.0032,
  span_count: 4,
  error_count: 0,
  evaluation_context: {
    evaluation_name: 'support-policy-smoke',
    test_case_name: 'case-0042',
  },
};

const trace2: Trace = {
  id: 'trace-agent-run-002',
  root_span_id: 'span-root-002',
  session_id: 'session-agent-run-002',
  workspace: 'default',
  name: 'Retrieve deployment troubleshooting steps',
  input: 'Why did my deployment fail?',
  output: 'The knowledge base lookup timed out before troubleshooting steps were retrieved.',
  started_at: '2026-05-20T15:11:00Z',
  ended_at: '2026-05-20T15:11:07Z',
  duration_ms: 7240,
  status: 'error',
  input_tokens: 780,
  output_tokens: 96,
  total_tokens: 876,
  cost_usd: 0.0011,
  span_count: 3,
  error_count: 1,
};

const trace3: Trace = {
  id: 'trace-agent-run-003',
  root_span_id: 'span-root-003',
  session_id: 'session-agent-run-001',
  workspace: 'default',
  name: 'Explain private workspace access controls',
  input: 'How do I restrict access to that private workspace?',
  output:
    'Add only the required users or groups as workspace members and assign least-privilege roles.',
  started_at: '2026-05-20T16:43:00Z',
  ended_at: '2026-05-20T16:43:09Z',
  duration_ms: 9130,
  status: 'success',
  input_tokens: 980,
  output_tokens: 214,
  total_tokens: 1194,
  cost_usd: 0.0021,
  span_count: 2,
  error_count: 0,
};

const span1: Span = {
  span_id: 'span-root-001',
  session_id: 'session-agent-run-001',
  workspace: 'default',
  kind: 'AGENT',
  name: 'Answer customer policy question',
  source: 'otel',
  trace_id: 'trace-agent-run-001',
  started_at: '2026-05-20T16:42:00Z',
  ended_at: '2026-05-20T16:42:12Z',
  status: 'success',
  agent_name: 'support-agent',
  total_tokens: 1754,
  cost_total_usd: 0.0032,
  input: 'Can I deploy this model in a private workspace?',
  output: 'Yes. Use a private workspace and restrict access through workspace membership.',
  ingested_at: '2026-05-20T16:42:15Z',
};

const span2: Span = {
  span_id: 'span-llm-001',
  session_id: 'session-agent-run-001',
  workspace: 'default',
  parent_span_id: 'span-root-001',
  kind: 'LLM',
  name: 'Generate final response',
  source: 'otel',
  trace_id: 'trace-agent-run-001',
  started_at: '2026-05-20T16:42:08Z',
  ended_at: '2026-05-20T16:42:12Z',
  status: 'success',
  provider: 'nim',
  model: 'meta/llama-3.1-70b-instruct',
  input_tokens: 1240,
  output_tokens: 386,
  cached_tokens: 128,
  total_tokens: 1754,
  cost_total_usd: 0.0032,
  input: 'Workspace policy context and user question',
  output: 'Yes. Use a private workspace and restrict access through workspace membership.',
  ingested_at: '2026-05-20T16:42:15Z',
};

const span3: Span = {
  span_id: 'span-root-002',
  session_id: 'session-agent-run-002',
  workspace: 'default',
  kind: 'AGENT',
  name: 'Retrieve deployment troubleshooting steps',
  source: 'otel',
  trace_id: 'trace-agent-run-002',
  started_at: '2026-05-20T15:11:00Z',
  ended_at: '2026-05-20T15:11:07Z',
  status: 'error',
  error_type: 'ToolTimeout',
  error_message: 'Knowledge base lookup timed out after 5s.',
  agent_name: 'support-agent',
  total_tokens: 876,
  cost_total_usd: 0.0011,
  ingested_at: '2026-05-20T15:11:09Z',
};

const span4: Span = {
  span_id: 'span-root-003',
  session_id: 'session-agent-run-001',
  workspace: 'default',
  kind: 'AGENT',
  name: 'Explain private workspace access controls',
  source: 'otel',
  trace_id: 'trace-agent-run-003',
  started_at: '2026-05-20T16:43:00Z',
  ended_at: '2026-05-20T16:43:09Z',
  status: 'success',
  agent_name: 'support-agent',
  total_tokens: 1194,
  cost_total_usd: 0.0021,
  input: 'How do I restrict access to that private workspace?',
  output:
    'Add only the required users or groups as workspace members and assign least-privilege roles.',
  ingested_at: '2026-05-20T16:43:12Z',
};

const span5: Span = {
  span_id: 'span-llm-003',
  session_id: 'session-agent-run-001',
  workspace: 'default',
  parent_span_id: 'span-root-003',
  kind: 'LLM',
  name: 'Generate access-control guidance',
  source: 'otel',
  trace_id: 'trace-agent-run-003',
  started_at: '2026-05-20T16:43:04Z',
  ended_at: '2026-05-20T16:43:09Z',
  status: 'success',
  provider: 'nim',
  model: 'meta/llama-3.1-70b-instruct',
  input_tokens: 980,
  output_tokens: 214,
  total_tokens: 1194,
  cost_total_usd: 0.0021,
  input: 'Conversation history, workspace policy, and the follow-up question',
  output:
    'Add only the required users or groups as workspace members and assign least-privilege roles.',
  ingested_at: '2026-05-20T16:43:12Z',
};

export const mockTracesPage: TracesPage = {
  data: [trace3, trace1, trace2],
  pagination: {
    page: 1,
    page_size: 50,
    current_page_size: 3,
    total_pages: 1,
    total_results: 3,
  },
  sort: '-started_at',
};

const session1: Session = {
  id: 'session-agent-run-001',
  workspace: 'default',
  started_at: '2026-05-20T16:42:00Z',
  ended_at: '2026-05-20T16:43:09Z',
  duration_ms: 69_000,
  status: 'success',
  input_tokens: 2220,
  output_tokens: 600,
  cached_tokens: 128,
  total_tokens: 2948,
  cost_usd: 0.0053,
  trace_count: 2,
  span_count: 6,
};

const session2: Session = {
  id: 'session-agent-run-002',
  workspace: 'default',
  started_at: '2026-05-20T15:11:00Z',
  ended_at: '2026-05-20T15:11:07Z',
  duration_ms: 7240,
  status: 'error',
  input_tokens: 780,
  output_tokens: 96,
  total_tokens: 876,
  cost_usd: 0.0011,
  trace_count: 1,
  span_count: 3,
};

export const mockSpansPage: SpansPage = {
  data: [span1, span2, span3, span4, span5],
  pagination: {
    page: 1,
    page_size: 50,
    current_page_size: 5,
    total_pages: 1,
    total_results: 5,
  },
  sort: '-started_at',
};

export const mockTraceById = (id: string): Trace | undefined =>
  mockTracesPage.data.find((trace) => trace.id === id);

export const mockSessionById = (id: string): Session | undefined =>
  [session1, session2].find((session) => session.id === id);

export const mockSpanById = (id: string): Span | undefined =>
  mockSpansPage.data.find((span) => span.span_id === id);

const baseAnnotations: Annotation[] = [
  {
    annotation_id: 'annotation-feedback-001',
    workspace: 'default',
    span_id: span1.span_id,
    session_id: span1.session_id,
    created_by: 'test-user',
    created_at: '2026-05-20T16:43:00Z',
    ingested_at: '2026-05-20T16:43:01Z',
    kind: 'feedback',
    value: 'positive',
  },
  {
    annotation_id: 'annotation-note-001',
    workspace: 'default',
    span_id: span1.span_id,
    session_id: span1.session_id,
    created_by: 'test-user',
    created_at: '2026-05-20T16:43:30Z',
    ingested_at: '2026-05-20T16:43:31Z',
    kind: 'note',
    text: 'Good final response, but verify policy citations.',
  },
  {
    annotation_id: 'annotation-metadata-001',
    workspace: 'default',
    span_id: span2.span_id,
    session_id: span2.session_id,
    created_by: 'test-user',
    created_at: '2026-05-20T16:44:00Z',
    ingested_at: '2026-05-20T16:44:01Z',
    kind: 'metadata',
    metadata: {
      review_queue: 'support-policy',
      priority: 'normal',
    },
  },
];

let mockAnnotations = [...baseAnnotations];
let nextAnnotationSequence = baseAnnotations.length + 1;

const pageAnnotations = (annotations: Annotation[], page: number, pageSize: number): Annotation[] =>
  annotations.slice((page - 1) * pageSize, page * pageSize);

export const resetMockAnnotations = (): void => {
  mockAnnotations = [...baseAnnotations];
  nextAnnotationSequence = baseAnnotations.length + 1;
};

const nextAnnotationId = (): string => `annotation-${nextAnnotationSequence++}`;

export const mockAnnotationsPage = ({
  spanId,
  sessionId,
  page = 1,
  pageSize = 100,
}: {
  spanId?: string;
  sessionId?: string;
  page?: number;
  pageSize?: number;
}): AnnotationsPage => {
  const filtered = mockAnnotations.filter(
    (annotation) =>
      (!spanId || annotation.span_id === spanId) &&
      (!sessionId || annotation.session_id === sessionId)
  );
  const sorted = [...filtered].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
  const data = pageAnnotations(sorted, page, pageSize);

  return {
    data,
    pagination: {
      page,
      page_size: pageSize,
      current_page_size: data.length,
      total_pages: Math.ceil(sorted.length / pageSize),
      total_results: sorted.length,
    },
    sort: '-created_at',
    filter: spanId || sessionId ? { span_id: spanId, session_id: sessionId } : undefined,
  };
};

export const createMockAnnotation = ({
  workspace,
  data,
}: {
  workspace: string;
  data: AnnotationInput;
}): Annotation => {
  const now = new Date().toISOString();
  const base = {
    annotation_id: nextAnnotationId(),
    workspace,
    span_id: data.span_id,
    session_id: data.session_id,
    created_by: 'test-user',
    created_at: now,
    ingested_at: now,
  };

  let annotation: Annotation;
  switch (data.kind) {
    case FeedbackAnnotationInputKind.feedback:
      annotation = {
        ...base,
        kind: 'feedback',
        value: data.value,
      };
      break;
    case NoteAnnotationInputKind.note:
      annotation = {
        ...base,
        kind: 'note',
        text: data.text,
      };
      break;
    case LabelAnnotationInputKind.label:
      annotation = {
        ...base,
        kind: 'label',
        value: data.value,
        value_type: data.value_type,
        name: data.name,
      };
      break;
    case MetadataAnnotationInputKind.metadata:
      annotation = {
        ...base,
        kind: 'metadata',
        metadata: data.metadata,
      };
      break;
  }

  mockAnnotations = [annotation, ...mockAnnotations];
  return annotation;
};

export const deleteMockAnnotation = (annotationId: string): boolean => {
  const nextAnnotations = mockAnnotations.filter(
    (annotation) => annotation.annotation_id !== annotationId
  );
  const deleted = nextAnnotations.length !== mockAnnotations.length;
  mockAnnotations = nextAnnotations;
  return deleted;
};

const rollup = (mean: number) => ({ sum: mean * 4, mean });

/**
 * Trace rollups for the agent overview statistics. Bucket starts are relative to now so the
 * points always land inside the range the overview asks for.
 */
export const mockTraceMetrics = (
  bucket: TraceMetricBucketParam,
  timezone: string
): TraceMetrics => {
  const point = (bucketStart?: string): TraceMetricPointResponse => ({
    ...(bucketStart ? { bucket_start: bucketStart } : {}),
    run_count: 4,
    failed_run_count: 1,
    input_tokens: rollup(820),
    output_tokens: rollup(240),
    cached_tokens: rollup(0),
    total_tokens: rollup(1060),
    cost_usd: rollup(0.0142),
    latency_ms: { mean: 2410 },
  });

  if (bucket === 'total') return { bucket, timezone, data: [point()] };

  const day = 24 * 60 * 60 * 1000;
  const data = [3, 2, 1, 0].map((daysAgo) =>
    point(new Date(Date.now() - daysAgo * day).toISOString())
  );
  return { bucket, timezone, data };
};
