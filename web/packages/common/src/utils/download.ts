// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { triggerDownload } from '@nemo/common/src/utils/file';

export type DownloadFormat = 'json' | 'jsonl' | 'csv';

export type CsvRow = Record<string, unknown>;

/** What a page hands the download menu: the full record for JSON, the unflattened per-row records
 *  for JSONL, and those rows flattened into a table for CSV. */
export interface DownloadPayload {
  json: unknown;
  jsonl: unknown[];
  csv: CsvRow[];
}

export const MIME_TYPES: Record<DownloadFormat, string> = {
  json: 'application/json',
  jsonl: 'application/jsonl',
  csv: 'text/csv',
};

/** Parquet INT64 columns decode as BigInt, which JSON.stringify rejects by default. */
const jsonReplacer = (_key: string, value: unknown): unknown =>
  typeof value === 'bigint' ? value.toString() : value;

export interface CsvOptions {
  /** Column order; defaults to the union of every row's keys, in first-seen order. */
  columns?: readonly string[];
  /** Prefixes strings a spreadsheet would run as a formula with `'`. Disable for data files that
   *  must round-trip unchanged. @defaultValue true */
  formulaSafe?: boolean;
}

/** Formats one CSV cell: nested values as JSON, quoted when it holds a comma, quote, or newline. */
export const toCsvCell = (value: unknown, { formulaSafe = true }: CsvOptions = {}): string => {
  if (value == null) return '';
  const text = typeof value === 'object' ? JSON.stringify(value, jsonReplacer) : String(value);
  // Only strings: a negative number is not a formula.
  const safe =
    formulaSafe && typeof value === 'string' && /^[=+\-@]/.test(text.trimStart())
      ? `'${text}`
      : text;
  return /[,"\n\r]/.test(safe) ? `"${safe.replace(/"/g, '""')}"` : safe;
};

/** Serializes rows to CSV, writing nested values as JSON so no data is lost. */
export const toCsv = <T extends object>(rows: readonly T[], options: CsvOptions = {}): string => {
  const columns = options.columns ?? [...new Set(rows.flatMap(Object.keys))];
  return [
    columns.map((column) => toCsvCell(column, options)),
    ...rows.map((row) => columns.map((column) => toCsvCell((row as CsvRow)[column], options))),
  ]
    .map((cells) => cells.join(','))
    .join('\n');
};

/** Strips characters that are unsafe in file names (e.g. a workspace `/`). */
export const toFileName = (name: string): string => name.replace(/[^\w.-]+/g, '_');

export const toJsonl = (rows: readonly unknown[]): string =>
  rows.map((row) => JSON.stringify(row, jsonReplacer)).join('\n');

const SERIALIZERS: Record<DownloadFormat, (payload: DownloadPayload) => string> = {
  json: (payload) => JSON.stringify(payload.json, jsonReplacer, 2),
  jsonl: (payload) => toJsonl(payload.jsonl),
  csv: (payload) => toCsv(payload.csv),
};

/** A payload with no per-row records has no results, even though its JSON record is not empty. */
export const hasResults = (payload: DownloadPayload): boolean =>
  payload.jsonl.length > 0 || payload.csv.length > 0;

/** Downloads the payload in the given format. Returns false, without downloading, when there are no
 *  results, so a run with nothing in it never produces an empty file. */
export const downloadPayload = (
  payload: DownloadPayload,
  name: string,
  format: DownloadFormat
): boolean => {
  const content = SERIALIZERS[format](payload);
  if (!hasResults(payload) || !content) return false;
  triggerDownload(
    new Blob([content], { type: MIME_TYPES[format] }),
    `${toFileName(name)}.${format}`
  );
  return true;
};
