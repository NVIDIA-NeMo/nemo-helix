// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readParquetRows } from '@studio/api/datasets/filesetParquetRows';
import {
  formatFromFileName,
  parseCsvGrid,
  parseDataFile,
} from '@studio/components/FileRowEditor/parse';

/** Stem the dataset is stored under in the run's fileset; the extension follows its content. */
const DATASET_BASENAME = 'dataset';

/** Either the name a valid dataset is stored under, or why it was rejected. */
export interface DatasetInspection {
  storedName?: string;
  error?: string;
}

const isRecord = (value: unknown): boolean =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const storedAs = (
  records: readonly unknown[],
  format: 'json' | 'jsonl' | 'csv' | 'parquet'
): DatasetInspection =>
  records.length === 0
    ? { error: 'File contains no data' }
    : { storedName: `${DATASET_BASENAME}.${format}` };

/** The evaluator's loader takes ``.json`` and ``.jsonl`` interchangeably, sniffing a leading
 *  ``[`` to tell an array from line-delimited records — so the name follows the detected
 *  content, not the uploaded extension, and the dataset ref written into the config matches.
 *
 *  Every record must be an object: the evaluator turns each one into a row keyed by its own
 *  fields, and a file of scalars only fails once the job is running. */
const inspectJsonText = (rawText: string): DatasetInspection => {
  const text = rawText.trim();
  if (!text) return { error: 'File is empty' };

  let records: unknown[];
  let format: 'json' | 'jsonl';
  try {
    const parsed: unknown = JSON.parse(text);
    records = Array.isArray(parsed) ? parsed : [parsed];
    format = 'json';
  } catch {
    try {
      records = text.split('\n').flatMap((line) => (line.trim() ? [JSON.parse(line)] : []));
      format = 'jsonl';
    } catch {
      return { error: 'File is not valid JSON or JSONL' };
    }
  }

  if (records.length > 0 && !records.every(isRecord)) {
    return { error: 'Every dataset record must be a JSON object.' };
  }
  return storedAs(records, format);
};

const isBlankLine = (cells: readonly string[]): boolean => cells.length === 1 && cells[0] === '';

// The evaluator reads CSV with pyarrow's defaults, which refuse both of these.
const inspectCsvText = (text: string): DatasetInspection => {
  if (!text.trim()) return { error: 'File is empty' };

  const [header, ...body] = parseCsvGrid(text);
  if ([header, ...body].some((cells) => cells.some((cell) => /[\r\n]/.test(cell)))) {
    return { error: 'CSV values cannot span more than one line.' };
  }
  const raggedIndex = body.findIndex(
    (cells) => !isBlankLine(cells) && cells.length !== header.length
  );
  if (raggedIndex >= 0) {
    return {
      error: `Row ${raggedIndex + 1} has ${body[raggedIndex].length} values but the header has ${header.length}.`,
    };
  }
  return storedAs(parseDataFile(text, 'csv'), 'csv');
};

/** CSV and Parquet rows always decode as objects, so only emptiness needs checking. Reading the
 *  first row is enough to prove the file decodes with a codec the browser supports. */
const inspectParquet = async (file: File): Promise<DatasetInspection> => {
  try {
    return storedAs(await readParquetRows(file, 1), 'parquet');
  } catch (err) {
    return { error: err instanceof Error ? err.message : 'Could not read the dataset file.' };
  }
};

/** Validate a dataset and settle the name it is stored under. The file is stored as-is, so the
 *  evaluator reads it with the loader for its real format rather than a lossy conversion. */
export const inspectDatasetFile = async (file: File): Promise<DatasetInspection> => {
  switch (formatFromFileName(file.name)) {
    case 'parquet':
      return inspectParquet(file);
    case 'csv':
      return inspectCsvText(await file.text());
    default:
      return inspectJsonText(await file.text());
  }
};
