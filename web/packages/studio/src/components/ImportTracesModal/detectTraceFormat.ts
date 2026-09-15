// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** The Intake ingest endpoints a picked file can be sent to. */
export type TraceFormat = 'atif' | 'spans' | 'chat-completions' | 'otlp-protobuf';

export interface DetectedFormat {
  format: TraceFormat;
  /** The parsed body. Absent for `otlp-protobuf`, which is forwarded as raw bytes. */
  document?: unknown;
}

export interface UndetectedFormat {
  format: null;
  message: string;
}

export type Detection = DetectedFormat | UndetectedFormat;

const PROTOBUF_EXTENSIONS = ['.pb', '.binpb', '.protobuf'];

export const FORMAT_LABELS: Record<TraceFormat, string> = {
  atif: 'ATIF',
  spans: 'Spans',
  'chat-completions': 'Chat completions',
  'otlp-protobuf': 'OTLP',
};

/** Whether a name looks like a serialized OTLP protobuf, which is never parsed as text. */
export const isProtobufFileName = (fileName: string): boolean =>
  PROTOBUF_EXTENSIONS.some((extension) => fileName.toLowerCase().endsWith(extension));

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const looksAtif = (value: Record<string, unknown>): boolean =>
  typeof value.schema_version === 'string' ||
  (isRecord(value.agent) && typeof value.agent.name === 'string');

const looksSpan = (value: unknown): boolean =>
  isRecord(value) &&
  typeof value.span_id === 'string' &&
  typeof value.trace_id === 'string' &&
  typeof value.started_at === 'string';

/** Reads a JSONL body as one array, or null when any line is not a JSON value of its own. */
const parseJsonLines = (text: string): unknown[] | null => {
  const lines = text.split('\n').filter((line) => line.trim().length > 0);
  if (lines.length < 2) return null;

  const records: unknown[] = [];
  for (const line of lines) {
    try {
      records.push(JSON.parse(line));
    } catch {
      return null;
    }
  }
  return records;
};

const looksChatCompletion = (value: Record<string, unknown>): boolean =>
  isRecord(value.request) && isRecord(value.response);

/**
 * Names the ingest endpoint a picked file belongs to.
 *
 * Trace sets arrive a directory at a time and a directory is rarely one format, so the
 * format is read off each file rather than asked for once up front. The four bodies are
 * structurally disjoint, and every JSON endpoint forbids unknown top-level fields, so a
 * wrong guess is rejected by Intake rather than landing malformed telemetry.
 *
 * `text` is null for a file that was never read as text, which is how a serialized OTLP
 * protobuf reaches here.
 */
export const detectTraceFormat = (fileName: string, text: string | null): Detection => {
  if (text === null || isProtobufFileName(fileName)) return { format: 'otlp-protobuf' };

  if (text.trim().length === 0) return { format: null, message: 'The file is empty.' };

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    // Span and trajectory exports are as often line-delimited as they are one JSON array.
    const lines = parseJsonLines(text);
    if (lines === null) {
      return {
        format: null,
        message: error instanceof Error ? error.message : 'The file is not valid JSON.',
      };
    }
    parsed = lines;
  }

  if (Array.isArray(parsed) && parsed.length === 0) {
    return { format: null, message: 'The file holds an empty array.' };
  }

  const sample = Array.isArray(parsed) ? parsed[0] : parsed;

  if (looksSpan(sample)) return { format: 'spans', document: parsed };

  if (!isRecord(sample)) {
    return { format: null, message: 'Expected a JSON object, or an array of them.' };
  }

  // Intake's OTLP endpoint only accepts application/x-protobuf, so an OTLP JSON export
  // cannot be forwarded from here however well it is recognized.
  if ('resourceSpans' in sample || 'resource_spans' in sample) {
    return {
      format: null,
      message:
        'OTLP JSON cannot be uploaded — Intake accepts OTLP as protobuf only. Re-export as protobuf, or use the skill, which can convert it.',
    };
  }

  if (looksAtif(sample)) return { format: 'atif', document: parsed };
  if (Array.isArray(sample.spans)) return { format: 'spans', document: parsed };
  if (looksChatCompletion(sample)) return { format: 'chat-completions', document: parsed };

  return {
    format: null,
    message:
      'Unrecognized trace format. Expected ATIF, direct spans, captured chat completions, or an OTLP protobuf.',
  };
};
