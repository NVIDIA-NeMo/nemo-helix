// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { escapeCSV } from '@nemo/common/src/components/DataView/internal/utils/csv';
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

const MIME_TYPES: Record<DownloadFormat, string> = {
  json: 'application/json',
  jsonl: 'application/jsonl',
  csv: 'text/csv',
};

const toCell = (value: unknown): string => {
  if (value == null) return '';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
};

/** Serializes rows to CSV. Columns are the union of every row's keys, in first-seen order; nested
 *  values are written as JSON so no data is lost. */
export const toCsv = (rows: readonly CsvRow[]): string => {
  const columns = [...new Set(rows.flatMap(Object.keys))];
  return [columns, ...rows.map((row) => columns.map((column) => toCell(row[column])))]
    .map((cells) => cells.map(escapeCSV).join(','))
    .join('\n');
};

/** Strips characters that are unsafe in file names (e.g. a workspace `/`). */
export const toFileName = (name: string): string => name.replace(/[^\w.-]+/g, '_');

export const toJsonl = (rows: readonly unknown[]): string =>
  rows.map((row) => JSON.stringify(row)).join('\n');

const SERIALIZERS: Record<DownloadFormat, (payload: DownloadPayload) => string> = {
  json: (payload) => JSON.stringify(payload.json, null, 2),
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
