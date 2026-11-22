// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Span } from '@nemo/sdk/generated/platform/schema';

/**
 * Helpers for reading the per-kind data that Intake leaves in the
 * `raw_attributes` JSON bucket. Intake promotes a fixed catalog of semantic
 * keys to typed `Span` fields; everything else (retriever documents, guardrail
 * decisions, reranker scores, ...) lands here as flat dotted keys. Per-kind
 * templates use these helpers to elevate that data into first-class views.
 */

/** Parse `span.raw_attributes` (a JSON string) into a flat key/value record. */
export const parseRawAttributes = (raw: string | null | undefined): Record<string, unknown> => {
  if (!raw) {
    return {};
  }
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return {};
  } catch {
    return {};
  }
};

/** Read a single raw attribute value, regardless of where it sits in the bucket. */
export const readRawAttribute = (span: Span, key: string): unknown =>
  parseRawAttributes(span.raw_attributes)[key];

/** Narrow an unknown raw value to a number, else undefined. */
export const asNumber = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined;

/** Narrow an unknown raw value to a boolean, else undefined. */
export const asBoolean = (value: unknown): boolean | undefined =>
  typeof value === 'boolean' ? value : undefined;

/** Coerce a raw value (array, or JSON-array string) into a list of strings. */
export const asList = (value: unknown): string[] => {
  let parsed = value;
  if (typeof value === 'string') {
    try {
      parsed = JSON.parse(value);
    } catch {
      return [value];
    }
  }
  return Array.isArray(parsed) ? parsed.map((item) => String(item)) : [];
};

/**
 * Collect indexed attributes of the form `${prefix}.${index}.${field...}` into
 * an array ordered by index. Each element maps the remaining field path to its
 * value, e.g. for `retrieval.documents.0.document.id` with
 * prefix `retrieval.documents` the element is `{ 'document.id': ... }`.
 */
export const collectIndexedEntries = (
  attributes: Record<string, unknown>,
  prefix: string
): Array<Record<string, unknown>> => {
  const byIndex = new Map<number, Record<string, unknown>>();
  const dottedPrefix = `${prefix}.`;

  for (const [key, value] of Object.entries(attributes)) {
    if (!key.startsWith(dottedPrefix)) {
      continue;
    }
    const rest = key.slice(dottedPrefix.length);
    const separator = rest.indexOf('.');
    if (separator < 0) {
      continue;
    }
    const index = Number(rest.slice(0, separator));
    if (!Number.isInteger(index)) {
      continue;
    }
    const field = rest.slice(separator + 1);
    const record = byIndex.get(index) ?? {};
    record[field] = value;
    byIndex.set(index, record);
  }

  return [...byIndex.entries()].sort(([left], [right]) => left - right).map(([, record]) => record);
};

export interface RankedDocument {
  rank: number;
  id?: string;
  score?: number;
  content?: string;
  /** Any other per-document fields that weren't recognized. */
  metadata: Record<string, unknown>;
}

/** Map one collected indexed entry into a typed ranked document. */
const toRankedDocument = (entry: Record<string, unknown>, index: number): RankedDocument => {
  const document: RankedDocument = { rank: index + 1, metadata: {} };
  for (const [field, value] of Object.entries(entry)) {
    // Retriever docs nest under `document.<field>`; reranker docs use bare fields.
    const normalized = field.replace(/^document\./, '');
    if (normalized === 'id' && typeof value === 'string') {
      document.id = value;
    } else if (normalized === 'score' && typeof value === 'number') {
      document.score = value;
    } else if (normalized === 'content' && typeof value === 'string') {
      document.content = value;
    } else {
      document.metadata[normalized] = value;
    }
  }
  return document;
};

/**
 * Extract retriever documents from a RETRIEVER span's raw attributes. Handles
 * the OpenInference convention `retrieval.documents.<i>.document.<field>`.
 */
export const extractRetrievedDocuments = (span: Span): RankedDocument[] =>
  collectIndexedEntries(parseRawAttributes(span.raw_attributes), 'retrieval.documents').map(
    toRankedDocument
  );

/**
 * Extract reranked documents from a RERANKER span's raw attributes
 * (`reranker.documents.<i>.<field>`), ordered by index (rank).
 */
export const extractRerankedDocuments = (span: Span): RankedDocument[] =>
  collectIndexedEntries(parseRawAttributes(span.raw_attributes), 'reranker.documents').map(
    toRankedDocument
  );
