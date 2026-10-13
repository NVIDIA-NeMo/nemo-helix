// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROW_ID_KEY, type DataFileRow } from '@studio/components/FileRowEditor/types';

/** Data file formats recognized by {@link formatFromFileName}. */
export type DataFileFormat = 'json' | 'jsonl' | 'csv' | 'parquet' | 'unknown';

/** Formats that {@link parseDataFile} can parse from in-browser text content. */
export const TEXT_PARSEABLE_FORMATS: readonly DataFileFormat[] = ['json', 'jsonl', 'csv'];

/** Derives a data file format from a file name's extension. */
export const formatFromFileName = (fileName: string): DataFileFormat => {
  const extension = fileName.split('.').pop()?.toLowerCase();
  switch (extension) {
    case 'json':
      return 'json';
    case 'jsonl':
    case 'ndjson':
      return 'jsonl';
    case 'csv':
      return 'csv';
    case 'parquet':
    case 'pq':
      return 'parquet';
    default:
      return 'unknown';
  }
};

/**
 * Normalizes an arbitrary parsed value into a row object. Plain objects pass through;
 * arrays and primitives are wrapped as `{ value }` so any JSON-array file still renders.
 */
const normalizeRow = (value: unknown): DataFileRow =>
  value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as DataFileRow)
    : { value };

const parseJson = (content: string): DataFileRow[] => {
  const data: unknown = JSON.parse(content);
  const list = Array.isArray(data) ? data : [data];
  return list.map(normalizeRow);
};

const parseJsonl = (content: string): DataFileRow[] =>
  content
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => normalizeRow(JSON.parse(line)));

/**
 * Parses CSV text into a grid of string cells, honoring RFC-4180 quoting: quoted
 * fields may contain commas and newlines, and `""` is an escaped quote.
 */
export const parseCsvGrid = (content: string): string[][] => {
  const rows: string[][] = [];
  let field = '';
  let row: string[] = [];
  let inQuotes = false;

  for (let i = 0; i < content.length; i++) {
    const char = content[i];
    if (inQuotes) {
      if (char === '"') {
        if (content[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += char;
      }
    } else if (char === '"') {
      inQuotes = true;
    } else if (char === ',') {
      row.push(field);
      field = '';
    } else if (char === '\n' || char === '\r') {
      if (char === '\r' && content[i + 1] === '\n') {
        i++;
      }
      row.push(field);
      field = '';
      rows.push(row);
      row = [];
    } else {
      field += char;
    }
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  // Drop only blank trailing lines (a final newline yields a single empty cell). Interior
  // empty rows are preserved — for a single-column file they are legitimate empty values.
  while (rows.length > 0) {
    const last = rows[rows.length - 1];
    if (last.length <= 1 && (last[0] ?? '') === '') {
      rows.pop();
    } else {
      break;
    }
  }
  return rows;
};

/** Coerces a raw CSV string cell into a boolean/number when it unambiguously is one. */
const coerceScalar = (raw: string): unknown => {
  if (raw === '') {
    return '';
  }
  if (raw === 'true') {
    return true;
  }
  if (raw === 'false') {
    return false;
  }
  // Reject leading-zero runs (e.g. "007", "00501") so zero-padded identifiers keep their
  // exact string form rather than silently losing digits to numeric coercion.
  if (/^-?(0|[1-9]\d*)$/.test(raw)) {
    const value = Number(raw);
    if (Number.isSafeInteger(value)) {
      return value;
    }
  }
  if (/^-?\d*\.\d+$/.test(raw)) {
    return Number(raw);
  }
  return raw;
};

const parseCsv = (content: string): DataFileRow[] => {
  const grid = parseCsvGrid(content);
  if (grid.length === 0) {
    return [];
  }
  const [header, ...body] = grid;
  return body.map((cells) => {
    const row: DataFileRow = {};
    header.forEach((key, index) => {
      row[key] = coerceScalar(cells[index] ?? '');
    });
    return row;
  });
};

/**
 * Parses the text content of a data file into row-like records. Supports JSON (array
 * or single object), JSONL/NDJSON, and CSV. Binary formats (e.g. Parquet) are not
 * parseable in-browser and must be loaded through the Files API. Throws on malformed
 * input or an unsupported format.
 */
export const parseDataFile = (content: string, format: DataFileFormat): DataFileRow[] => {
  switch (format) {
    case 'json':
      return parseJson(content);
    case 'jsonl':
      return parseJsonl(content);
    case 'csv':
      return parseCsv(content);
    default:
      throw new Error(`Unsupported data file format: ${format}`);
  }
};

/** Drops the synthetic {@link ROW_ID_KEY} so it never leaks into serialized output. */
const stripRowId = (row: DataFileRow): DataFileRow => {
  const rest: DataFileRow = {};
  for (const [key, value] of Object.entries(row)) {
    if (key !== ROW_ID_KEY) {
      rest[key] = value;
    }
  }
  return rest;
};

/** RFC-4180-escapes a single CSV cell, quoting when it contains a comma, quote, or newline. */
const serializeCsvCell = (value: unknown): string => {
  const str =
    value === null || value === undefined
      ? ''
      : typeof value === 'object'
        ? JSON.stringify(value)
        : String(value);
  return /[",\n\r]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
};

/**
 * Serializes rows back into data-file text content — the counterpart to {@link parseDataFile}
 * used by the in-browser Download action. The synthetic {@link ROW_ID_KEY} is stripped. JSONL
 * emits one compact object per line; CSV derives its header from the union of row keys (in
 * first-seen order) and escapes every cell; all other formats fall back to a pretty JSON array.
 */
export const serializeDataFile = (rows: DataFileRow[], format: DataFileFormat): string => {
  const clean = rows.map(stripRowId);

  switch (format) {
    case 'jsonl':
      return clean.map((row) => JSON.stringify(row)).join('\n');
    case 'csv': {
      const keys: string[] = [];
      for (const row of clean) {
        for (const key of Object.keys(row)) {
          if (!keys.includes(key)) {
            keys.push(key);
          }
        }
      }
      const header = keys.map(serializeCsvCell).join(',');
      const body = clean.map((row) => keys.map((key) => serializeCsvCell(row[key])).join(','));
      return [header, ...body].join('\n');
    }
    default:
      return JSON.stringify(clean, null, 2);
  }
};
