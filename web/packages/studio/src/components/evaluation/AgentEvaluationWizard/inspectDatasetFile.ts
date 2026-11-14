// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isPlainObject } from '@nemo/common/src/utils/isPlainObject';
import { parseJsonOrJsonl, type ParsedJsonRecords } from '@nemo/common/src/utils/parseRecords';
import { readParquetRows } from '@studio/api/datasets/filesetParquetRows';
import {
  formatFromFileName,
  parseCsvGrid,
  parseDataFile,
} from '@studio/components/FileRowEditor/parse';

/** Why a dataset was rejected; empty when it is valid. */
export interface DatasetInspection {
  error?: string;
}

const nonEmpty = (records: readonly unknown[]): DatasetInspection =>
  records.length === 0 ? { error: 'File contains no data' } : {};

/** The evaluator's loader takes ``.json`` and ``.jsonl`` interchangeably, sniffing a leading
 *  ``[`` to tell an array from line-delimited records, so either extension is read either way.
 *
 *  Every record must be an object: the evaluator turns each one into a row keyed by its own
 *  fields, and a file of scalars only fails once the job is running. */
const inspectJsonText = (rawText: string): DatasetInspection => {
  const text = rawText.trim();
  if (!text) return { error: 'File is empty' };

  let parsed: ParsedJsonRecords;
  try {
    parsed = parseJsonOrJsonl(text);
  } catch {
    return { error: 'File is not valid JSON or JSONL' };
  }
  const { records } = parsed;

  if (records.length > 0 && !records.every(isPlainObject)) {
    return { error: 'Every dataset record must be a JSON object.' };
  }
  return nonEmpty(records);
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
  return nonEmpty(parseDataFile(text, 'csv'));
};

/** CSV and Parquet rows always decode as objects, so only emptiness needs checking. Reading the
 *  first row is enough to prove the file decodes with a codec the browser supports. */
const inspectParquet = async (file: File): Promise<DatasetInspection> => {
  try {
    return nonEmpty(await readParquetRows(file, 1));
  } catch (err) {
    return { error: err instanceof Error ? err.message : 'Could not read the dataset file.' };
  }
};

/** Validate a dataset the way the evaluator's loader for its format will read it. */
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
