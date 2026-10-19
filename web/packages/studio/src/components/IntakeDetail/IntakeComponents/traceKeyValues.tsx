// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Data-builder module: exports functions that assemble key/value entries into various segements (header, summary, etc.) Some
// produce JSX such as status badges and links. Not a component module.
/* eslint-disable react-refresh/only-export-components */

import type { EvaluationContext, Session, Trace } from '@nemo/sdk/generated/platform/schema';
import {
  formatUnknownKeyValue,
  isMeaningfulValue,
} from '@studio/components/IntakeDetail/IntakeComponents/keyValueFormatting';
import type {
  HighlightMetric,
  KeyValueEntry,
} from '@studio/components/IntakeDetail/IntakeComponents/keyValueTypes';
import { getIntakeSessionTraceRoute } from '@studio/routes/utils';
import {
  EMPTY_VALUE,
  formatCost,
  formatDurationMs,
  formatInteger,
  formatMaybe,
} from '@studio/util/intakeTelemetry';
import type { ReactNode } from 'react';
import { Link } from 'react-router';

interface TraceKeyValueContext {
  workspace: string;
}

type TraceFieldResolver = (trace: Trace, ctx: TraceKeyValueContext) => ReactNode | null | undefined;

interface TraceFieldDescriptor {
  readonly key: keyof Trace | string;
  readonly label: string;
  readonly resolve: TraceFieldResolver;
  readonly include?: (trace: Trace) => boolean;
}

const WRAPPED_TRACE_KEYS = new Set<keyof Trace | string>([
  'id',
  'root_span_id',
  'session_id',
  'workspace',
]);

const humanizeFieldLabel = (key: string): string =>
  key
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');

type TraceKeyValueEntry = KeyValueEntry;

type TraceHighlightMetric = HighlightMetric;

/**
 * Trace fields surfaced in the header (the metrics card + its leading status/
 * timing items) instead of the summary accordion. Used to exclude them from the
 * metadata so nothing is duplicated.
 */
export const TRACE_HIGHLIGHT_METRIC_KEYS = new Set<keyof Trace | string>([
  'status',
  'started_at',
  'ended_at',
  'agent_name',
  'span_count',
  'error_count',
  'duration_ms',
  'input_tokens',
  'output_tokens',
  'cached_tokens',
  'total_tokens',
  'cost_input_usd',
  'cost_output_usd',
  'cost_usd',
]);

const formatTokens = (value: number | null | undefined): string =>
  value != null ? formatInteger(value) : EMPTY_VALUE;

const formatCostValue = (value: number | null | undefined): string =>
  value != null ? formatCost(value) : EMPTY_VALUE;

const TRACE_SUMMARY_DESCRIPTORS: readonly TraceFieldDescriptor[] = [
  {
    key: 'name',
    label: 'Name',
    resolve: (trace) => formatMaybe(trace.name),
    include: (trace) => isMeaningfulValue(trace.name),
  },
  {
    key: 'id',
    label: 'Trace ID',
    resolve: (trace) => trace.id,
  },
  {
    key: 'root_span_id',
    label: 'Root Span',
    resolve: (trace, { workspace }) =>
      trace.root_span_id ? (
        <Link
          to={getIntakeSessionTraceRoute(workspace, trace.session_id, trace.id, {
            spanId: trace.root_span_id,
          })}
          className="break-all"
        >
          {trace.root_span_id}
        </Link>
      ) : (
        EMPTY_VALUE
      ),
    include: (trace) => isMeaningfulValue(trace.root_span_id),
  },
  {
    key: 'session_id',
    label: 'Session ID',
    resolve: (trace) => trace.session_id,
  },
  {
    key: 'workspace',
    label: 'Workspace',
    resolve: (trace) => trace.workspace,
  },
];

const EVALUATION_CONTEXT_DESCRIPTORS: readonly {
  readonly key: keyof EvaluationContext | string;
  readonly label: string;
}[] = [
  { key: 'evaluation_name', label: 'Evaluation Name' },
  { key: 'test_case_name', label: 'Test Case Name' },
];

const collectDescriptorEntries = (
  descriptors: readonly TraceFieldDescriptor[],
  trace: Trace,
  ctx: TraceKeyValueContext
): TraceKeyValueEntry[] =>
  descriptors.flatMap((descriptor) => {
    if (descriptor.include && !descriptor.include(trace)) {
      return [];
    }

    const value = descriptor.resolve(trace, ctx);
    if (value == null || value === '') {
      return [];
    }

    return [
      {
        id: String(descriptor.key),
        label: descriptor.label,
        value,
        wrapValue: WRAPPED_TRACE_KEYS.has(descriptor.key),
      },
    ];
  });

const collectUnmappedTraceEntries = (trace: Trace): TraceKeyValueEntry[] => {
  const mappedKeys = new Set([
    ...TRACE_SUMMARY_DESCRIPTORS.map((descriptor) => descriptor.key),
    ...TRACE_HIGHLIGHT_METRIC_KEYS,
    'evaluation_context',
    'experiment_context',
  ]);

  return Object.entries(trace)
    .filter(([key, value]) => !mappedKeys.has(key) && isMeaningfulValue(value))
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value]) => ({
      id: key,
      label: humanizeFieldLabel(key),
      value: formatUnknownKeyValue(value),
      wrapValue: true,
    }));
};

export const buildTraceSummaryEntries = (
  trace: Trace,
  ctx: TraceKeyValueContext
): TraceKeyValueEntry[] => [
  ...collectDescriptorEntries(TRACE_SUMMARY_DESCRIPTORS, trace, ctx),
  ...collectUnmappedTraceEntries(trace),
];

/**
 * Headline metrics for the trace summary header. Token and cost are collapsed to
 * their totals; the per-direction breakdown rides along as `details`, surfaced
 * in a hover popover.
 */
type HighlightTelemetry = Pick<
  Trace,
  | 'span_count'
  | 'duration_ms'
  | 'input_tokens'
  | 'output_tokens'
  | 'cached_tokens'
  | 'total_tokens'
  | 'cost_usd'
  | 'cost_input_usd'
  | 'cost_output_usd'
>;

const buildHighlightMetrics = (telemetry: HighlightTelemetry): TraceHighlightMetric[] => [
  {
    id: 'span_count',
    label: 'Spans',
    value: telemetry.span_count != null ? formatInteger(telemetry.span_count) : EMPTY_VALUE,
  },
  {
    id: 'duration_ms',
    label: 'Duration',
    value: telemetry.duration_ms != null ? formatDurationMs(telemetry.duration_ms) : EMPTY_VALUE,
  },
  {
    id: 'total_tokens',
    label: 'Total Tokens',
    value: formatTokens(telemetry.total_tokens),
    details: [
      {
        id: 'input_tokens',
        label: 'Input Tokens',
        value: formatTokens(telemetry.input_tokens),
      },
      {
        id: 'output_tokens',
        label: 'Output Tokens',
        value: formatTokens(telemetry.output_tokens),
      },
      {
        id: 'cached_tokens',
        label: 'Cached Tokens',
        value: formatTokens(telemetry.cached_tokens),
      },
    ],
  },
  {
    id: 'cost_usd',
    label: 'Total Cost',
    value: formatCostValue(telemetry.cost_usd),
    details: [
      {
        id: 'cost_input_usd',
        label: 'Input Cost',
        value: formatCostValue(telemetry.cost_input_usd),
      },
      {
        id: 'cost_output_usd',
        label: 'Output Cost',
        value: formatCostValue(telemetry.cost_output_usd),
      },
    ],
  },
];

export const buildSessionHighlightMetrics = (session: Session): TraceHighlightMetric[] =>
  buildHighlightMetrics(session);

export const buildTraceHighlightMetrics = (trace: Trace): TraceHighlightMetric[] =>
  buildHighlightMetrics(trace);

export const buildEvaluationContextEntries = (
  evaluationContext: EvaluationContext | null | undefined
): TraceKeyValueEntry[] => {
  if (!evaluationContext) {
    return [];
  }

  const mappedKeys = new Set<string>();
  const knownEntries = EVALUATION_CONTEXT_DESCRIPTORS.flatMap(({ key, label }) => {
    mappedKeys.add(String(key));
    const value = evaluationContext[key as keyof EvaluationContext];
    if (!isMeaningfulValue(value)) {
      return [];
    }

    return [
      {
        id: String(key),
        label,
        value: String(value),
        wrapValue: true,
      },
    ];
  });

  const extraEntries = Object.entries(evaluationContext)
    .filter(([key, value]) => !mappedKeys.has(key) && isMeaningfulValue(value))
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value]) => ({
      id: key,
      label: humanizeFieldLabel(key),
      value: formatUnknownKeyValue(value),
      wrapValue: true,
    }));

  return [...knownEntries, ...extraEntries];
};
