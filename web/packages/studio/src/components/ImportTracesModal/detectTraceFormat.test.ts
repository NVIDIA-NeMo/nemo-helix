// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  detectTraceFormat,
  isProtobufFileName,
} from '@studio/components/ImportTracesModal/detectTraceFormat';

const atif = {
  schema_version: 'ATIF-v1.5',
  agent: { name: 'email-security-triage' },
  steps: [],
};

const span = {
  span_id: 'span-1',
  trace_id: 'trace-1',
  started_at: '2026-01-01T00:00:00Z',
  name: 'agent.run',
};

const chatCompletion = {
  request: { model: 'gpt-4o', messages: [{ role: 'user', content: 'hi' }] },
  response: { id: 'chatcmpl-1', model: 'gpt-4o', choices: [] },
};

const detect = (value: unknown, fileName = 'traces.json') =>
  detectTraceFormat(fileName, JSON.stringify(value));

describe('detectTraceFormat', () => {
  it('detects ATIF from its schema version, alone or batched', () => {
    expect(detect(atif)).toEqual({ format: 'atif', document: atif });
    expect(detect([atif, atif]).format).toBe('atif');
  });

  it('detects ATIF that carries only an agent block', () => {
    expect(detect({ agent: { name: 'a' }, steps: [] }).format).toBe('atif');
  });

  it('detects direct spans as a bare array or a wrapped batch', () => {
    expect(detect([span]).format).toBe('spans');
    expect(detect({ source: 'langsmith', spans: [span] }).format).toBe('spans');
  });

  it('reads a line-delimited export as one batch', () => {
    const jsonl = [span, { ...span, span_id: 'span-2' }].map((r) => JSON.stringify(r)).join('\n');
    const detection = detectTraceFormat('spans.jsonl', jsonl);

    expect(detection.format).toBe('spans');
    expect(detection.format !== null && detection.document).toHaveLength(2);
  });

  it('does not mistake a broken JSON file for JSONL', () => {
    expect(detectTraceFormat('traces.json', '{"a": 1,\n"b": 2').format).toBeNull();
  });

  it('detects captured chat completions', () => {
    expect(detect(chatCompletion).format).toBe('chat-completions');
    expect(detect([chatCompletion]).format).toBe('chat-completions');
  });

  it('treats a protobuf extension as OTLP without reading it', () => {
    expect(detectTraceFormat('export.binpb', null)).toEqual({ format: 'otlp-protobuf' });
    expect(detectTraceFormat('export.pb', 'not json at all')).toEqual({ format: 'otlp-protobuf' });
  });

  it('rejects OTLP JSON, which the protobuf-only endpoint cannot take', () => {
    const detection = detect({ resourceSpans: [] });
    expect(detection.format).toBeNull();
    expect(detection.format === null && detection.message).toMatch(/OTLP JSON cannot be uploaded/);
  });

  it('reports unusable files instead of guessing', () => {
    expect(detect([]).format).toBeNull();
    expect(detectTraceFormat('traces.json', '   ').format).toBeNull();
    expect(detectTraceFormat('traces.json', '{not json').format).toBeNull();
    expect(detect({ hello: 'world' }).format).toBeNull();
    expect(detect(['just a string']).format).toBeNull();
  });
});

describe('isProtobufFileName', () => {
  it('matches the serialized OTLP extensions, case-insensitively', () => {
    expect(isProtobufFileName('Export.PB')).toBe(true);
    expect(isProtobufFileName('export.binpb')).toBe(true);
    expect(isProtobufFileName('export.protobuf')).toBe(true);
    expect(isProtobufFileName('export.json')).toBe(false);
  });
});
