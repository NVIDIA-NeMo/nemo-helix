// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parquetMetadata, parquetReadObjects } from 'hyparquet';

/** The non-blank lines of a JSONL body, trimmed. Handles `\r\n` line endings. */
export const jsonlLines = (text: string): string[] =>
  text.split(/\r?\n/).flatMap((line) => (line.trim() ? [line.trim()] : []));

interface ParseJsonlOptions {
  /** Drop lines that are not valid JSON instead of throwing. @defaultValue false */
  skipInvalid?: boolean;
}

/** Parses JSONL into one value per non-blank line. Throws naming the first bad line unless
 *  `skipInvalid` is set. */
export const parseJsonl = <T = unknown>(
  text: string,
  { skipInvalid = false }: ParseJsonlOptions = {}
): T[] =>
  text.split(/\r?\n/).flatMap((line, index) => {
    if (!line.trim()) return [];
    try {
      return [JSON.parse(line) as T];
    } catch (error) {
      if (skipInvalid) return [];
      const reason = error instanceof Error ? error.message : String(error);
      throw new Error(`line ${index + 1}: ${reason}`);
    }
  });

export interface ParsedJsonRecords {
  records: unknown[];
  format: 'json' | 'jsonl';
}

/** Parses a JSON document (array, or one value as a single record) or, failing that, JSONL.
 *  Throws the JSONL error when the text is neither. */
export const parseJsonOrJsonl = (text: string): ParsedJsonRecords => {
  try {
    const parsed: unknown = JSON.parse(text);
    return { records: Array.isArray(parsed) ? parsed : [parsed], format: 'json' };
  } catch {
    return { records: parseJsonl(text), format: 'jsonl' };
  }
};

type ParquetSource = Blob | ArrayBuffer;

const toBuffer = (source: ParquetSource): Promise<ArrayBuffer> | ArrayBuffer =>
  source instanceof Blob ? source.arrayBuffer() : source;

/** Decodes Parquet rows `[rowStart, rowEnd)` into plain objects. INT64 columns stay BigInt. */
export const readParquetObjects = async (
  source: ParquetSource,
  { rowStart, rowEnd }: { rowStart?: number; rowEnd?: number } = {}
): Promise<Record<string, unknown>[]> =>
  parquetReadObjects({ file: await toBuffer(source), rowStart, rowEnd });

/** Reads the row count from the Parquet footer without decoding any rows. */
export const parquetRowCount = async (source: ParquetSource): Promise<number> =>
  Number(parquetMetadata(await toBuffer(source)).num_rows);
