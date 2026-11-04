// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  jsonlLines,
  parquetRowCount,
  parseJsonl,
  parseJsonOrJsonl,
  readParquetObjects,
} from '@nemo/common/src/utils/parseRecords';

/** `{ n: 'row-0' } … { n: 'row-9' }`, written by hyparquet-writer. */
const TEN_ROWS =
  'UEFSMRUGFbQBFWhcFRQVABUUFQAVABUAAABaIAUAAAByb3ctMBEJADERCQAyEQkAMxEJADQRCQA1EQkANhEJADcRCSQ4BQAAAHJvdy05FQQZLEgEcm9vdBUCABUMJQAYAW4lAAAWFBkcGRwmCBwVDBkVABkYAW4VAhYUFpQBFpQBJgg8NgAoBXJvdy05GAVyb3ctMAAZHBUGFQAVAgAAABaUARYUACgJaHlwYXJxdWV0AGQAAABQQVIx';
const tenRows = () => new Blob([Uint8Array.from(atob(TEN_ROWS), (char) => char.charCodeAt(0))]);

describe('jsonlLines', () => {
  it('drops blank lines and handles CRLF endings', () => {
    expect(jsonlLines('{"a":1}\r\n\r\n  {"a":2}  \n')).toEqual(['{"a":1}', '{"a":2}']);
  });
});

describe('parseJsonl', () => {
  it('parses one value per non-blank line', () => {
    expect(parseJsonl('{"a":1}\n\n{"a":2}\r\n')).toEqual([{ a: 1 }, { a: 2 }]);
  });

  it('names the first malformed line', () => {
    expect(() => parseJsonl('{"a":1}\n{oops')).toThrow(/^line 2: /);
  });

  it('skips malformed lines when asked', () => {
    expect(parseJsonl('{"a":1}\n{oops\n{"a":3}', { skipInvalid: true })).toEqual([
      { a: 1 },
      { a: 3 },
    ]);
  });
});

describe('parseJsonOrJsonl', () => {
  it('reads a JSON array, or a single value as one record', () => {
    expect(parseJsonOrJsonl(' [{"a":1}] ')).toEqual({ records: [{ a: 1 }], format: 'json' });
    expect(parseJsonOrJsonl('{"a":1}')).toEqual({ records: [{ a: 1 }], format: 'json' });
  });

  it('falls back to JSONL', () => {
    expect(parseJsonOrJsonl('{"a":1}\n{"a":2}')).toEqual({
      records: [{ a: 1 }, { a: 2 }],
      format: 'jsonl',
    });
  });

  it('throws when the text is neither', () => {
    expect(() => parseJsonOrJsonl('not json')).toThrow(/^line 1: /);
  });
});

describe('parquet', () => {
  it('counts rows from the footer', async () => {
    expect(await parquetRowCount(tenRows())).toBe(10);
  });

  it('reads a row range', async () => {
    expect(await readParquetObjects(tenRows(), { rowStart: 2, rowEnd: 4 })).toEqual([
      { n: 'row-2' },
      { n: 'row-3' },
    ]);
  });
});
