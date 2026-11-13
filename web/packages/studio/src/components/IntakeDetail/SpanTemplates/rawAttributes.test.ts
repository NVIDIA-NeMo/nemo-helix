// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Span } from '@nemo/sdk/generated/platform/schema';
import {
  asBoolean,
  asList,
  asNumber,
  collectIndexedEntries,
  extractRerankedDocuments,
  extractRetrievedDocuments,
  parseRawAttributes,
  readRawAttribute,
} from '@studio/components/IntakeDetail/SpanTemplates/rawAttributes';

const makeSpan = (rawAttributes: Record<string, unknown> | null, input?: string): Span =>
  ({
    span_id: 'span-1',
    session_id: 'session-1',
    workspace: 'default',
    kind: 'RETRIEVER',
    source: 'otel',
    status: 'success',
    started_at: '2026-06-23T00:00:00Z',
    ingested_at: '2026-06-23T00:00:00Z',
    raw_attributes: rawAttributes === null ? undefined : JSON.stringify(rawAttributes),
    input,
  }) as Span;

describe('parseRawAttributes', () => {
  it('parses a JSON object string', () => {
    expect(parseRawAttributes('{"a":1,"b":"x"}')).toEqual({ a: 1, b: 'x' });
  });

  it('returns {} for null, empty, or non-object JSON', () => {
    expect(parseRawAttributes(undefined)).toEqual({});
    expect(parseRawAttributes(null)).toEqual({});
    expect(parseRawAttributes('')).toEqual({});
    expect(parseRawAttributes('[1,2]')).toEqual({});
    expect(parseRawAttributes('not json')).toEqual({});
  });
});

describe('collectIndexedEntries', () => {
  it('groups dotted indexed keys into ordered records', () => {
    const attrs = {
      'retrieval.documents.1.document.id': 'b',
      'retrieval.documents.0.document.id': 'a',
      'retrieval.documents.0.document.score': 0.9,
      unrelated: 'ignored',
    };
    expect(collectIndexedEntries(attrs, 'retrieval.documents')).toEqual([
      { 'document.id': 'a', 'document.score': 0.9 },
      { 'document.id': 'b' },
    ]);
  });
});

describe('extractRetrievedDocuments', () => {
  it('elevates retrieval.documents.* into ranked, typed documents', () => {
    const span = makeSpan({
      'retrieval.documents.0.document.id': 'doc-7f2a',
      'retrieval.documents.0.document.score': 0.88,
      'retrieval.documents.0.document.content': 'Sulfide electrolytes reached 25 Ah cells.',
      'retrieval.documents.1.document.id': 'doc-3c91',
      'retrieval.documents.1.document.score': 0.81,
      'retrieval.top_k': 8,
    });

    expect(extractRetrievedDocuments(span)).toEqual([
      {
        rank: 1,
        id: 'doc-7f2a',
        score: 0.88,
        content: 'Sulfide electrolytes reached 25 Ah cells.',
        metadata: {},
      },
      { rank: 2, id: 'doc-3c91', score: 0.81, metadata: {} },
    ]);
  });

  it('returns [] when there are no retrieval documents', () => {
    expect(extractRetrievedDocuments(makeSpan({ 'retrieval.top_k': 4 }))).toEqual([]);
    expect(extractRetrievedDocuments(makeSpan(null))).toEqual([]);
  });
});

describe('extractRerankedDocuments', () => {
  it('elevates reranker.documents.* (bare fields) into ranked documents', () => {
    const span = makeSpan({
      'reranker.documents.0.score': 0.94,
      'reranker.documents.1.score': 0.71,
      'reranker.top_n': 3,
    });

    expect(extractRerankedDocuments(span)).toEqual([
      { rank: 1, score: 0.94, metadata: {} },
      { rank: 2, score: 0.71, metadata: {} },
    ]);
  });

  it('returns [] when there are no reranked documents', () => {
    expect(extractRerankedDocuments(makeSpan({ 'reranker.top_n': 3 }))).toEqual([]);
  });
});

describe('readRawAttribute', () => {
  it('reads a single raw attribute value', () => {
    expect(readRawAttribute(makeSpan({ 'retrieval.top_k': 8 }), 'retrieval.top_k')).toBe(8);
    expect(readRawAttribute(makeSpan({}), 'missing')).toBeUndefined();
  });
});

describe('typed raw readers', () => {
  it('asNumber returns finite numbers, else undefined', () => {
    expect(asNumber(0)).toBe(0);
    expect(asNumber(0.2)).toBe(0.2);
    expect(asNumber('3')).toBeUndefined();
    expect(asNumber(Number.NaN)).toBeUndefined();
  });

  it('asBoolean returns booleans, else undefined', () => {
    expect(asBoolean(true)).toBe(true);
    expect(asBoolean(false)).toBe(false);
    expect(asBoolean('true')).toBeUndefined();
  });

  it('asList coerces arrays and JSON-array strings into string lists', () => {
    expect(asList(['safe', 'pii'])).toEqual(['safe', 'pii']);
    expect(asList('["safe","pii"]')).toEqual(['safe', 'pii']);
    expect(asList('safe')).toEqual(['safe']);
    expect(asList(42)).toEqual([]);
  });
});
