// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  inspectDatasetFile,
  inspectDatasetFiles,
} from '@studio/components/evaluation/inspectDatasetFile';
import { PARQUET, parquetFile } from '@studio/tests/util/parquetFixtures';

const textFile = (content: string, name: string): File => new File([content], name);

describe('inspectDatasetFile', () => {
  it('stores a JSON array as dataset.json', async () => {
    expect(await inspectDatasetFile(textFile('[{"a": 1}]', 'rows.jsonl'))).toEqual({
      storedName: 'dataset.json',
    });
  });

  it('stores line-delimited records as dataset.jsonl', async () => {
    expect(await inspectDatasetFile(textFile('{"a": 1}\n{"a": 2}\n', 'rows.json'))).toEqual({
      storedName: 'dataset.jsonl',
    });
  });

  it('rejects JSON records that are not objects', async () => {
    expect(await inspectDatasetFile(textFile('[1, 2]', 'rows.json'))).toEqual({
      error: 'Every dataset record must be a JSON object.',
    });
  });

  it('rejects text that is neither JSON nor JSONL', async () => {
    expect(await inspectDatasetFile(textFile('{"a": 1}\nnope', 'rows.jsonl'))).toEqual({
      error: 'File is not valid JSON or JSONL',
    });
  });

  it('rejects an empty JSON array', async () => {
    expect(await inspectDatasetFile(textFile('[]', 'rows.json'))).toEqual({
      error: 'File contains no data',
    });
  });

  it('stores a CSV with rows as dataset.csv', async () => {
    expect(await inspectDatasetFile(textFile('prompt,expected\nhi,hi\n', 'rows.csv'))).toEqual({
      storedName: 'dataset.csv',
    });
  });

  it('rejects a CSV value that spans lines', async () => {
    expect(
      await inspectDatasetFile(textFile('prompt,expected\n"line one\nline two",hi\n', 'rows.csv'))
    ).toEqual({ error: 'CSV values cannot span more than one line.' });
  });

  it('rejects a CSV row with more values than the header', async () => {
    expect(
      await inspectDatasetFile(textFile('prompt,expected\nhi,hi\na,b,c\n', 'rows.csv'))
    ).toEqual({ error: 'Row 2 has 3 values but the header has 2.' });
  });

  it('rejects a CSV row with fewer values than the header', async () => {
    expect(await inspectDatasetFile(textFile('prompt,expected\nhi\n', 'rows.csv'))).toEqual({
      error: 'Row 1 has 1 values but the header has 2.',
    });
  });

  it('accepts blank lines between CSV rows', async () => {
    expect(
      await inspectDatasetFile(textFile('prompt,expected\nhi,hi\n\nyo,yo\n', 'rows.csv'))
    ).toEqual({ storedName: 'dataset.csv' });
  });

  it('rejects a CSV with only a header', async () => {
    expect(await inspectDatasetFile(textFile('prompt,expected\n', 'rows.csv'))).toEqual({
      error: 'File contains no data',
    });
  });

  it('stores Parquet as dataset.parquet', async () => {
    expect(await inspectDatasetFile(parquetFile(PARQUET.twoRows))).toEqual({
      storedName: 'dataset.parquet',
    });
  });

  it('rejects Parquet with no rows', async () => {
    expect(await inspectDatasetFile(parquetFile(PARQUET.empty))).toEqual({
      error: 'File contains no data',
    });
  });

  it('rejects a .parquet file that does not decode', async () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

    const { error } = await inspectDatasetFile(textFile('{"a": 1}', 'rows.parquet'));

    expect(error).toMatch(/not valid Parquet/);
    consoleError.mockRestore();
  });

  it('rejects an empty file', async () => {
    expect(await inspectDatasetFile(textFile('  \n', 'rows.jsonl'))).toEqual({
      error: 'File is empty',
    });
  });
});

describe('inspectDatasetFiles', () => {
  it('stores a single file under its inspected name', async () => {
    const file = textFile('[{"a": 1}]', 'rows.json');

    expect(await inspectDatasetFiles([file])).toEqual({
      storedName: 'dataset.json',
      stored: [{ name: 'dataset.json', file }],
    });
  });

  it('stores nothing for a single file that fails inspection', async () => {
    expect(await inspectDatasetFiles([textFile('[]', 'rows.json')])).toEqual({
      error: 'File contains no data',
      stored: [],
    });
  });

  it('stores several Parquet files as parts read through one glob', async () => {
    const first = parquetFile(PARQUET.twoRows, 'a.parquet');
    const second = parquetFile(PARQUET.twoRows, 'b.parquet');

    expect(await inspectDatasetFiles([first, second])).toEqual({
      storedName: 'dataset/*.parquet',
      stored: [
        { name: 'dataset/part-0.parquet', file: first },
        { name: 'dataset/part-1.parquet', file: second },
      ],
    });
  });

  it('refuses to combine non-Parquet files', async () => {
    expect(
      await inspectDatasetFiles([
        parquetFile(PARQUET.twoRows, 'a.parquet'),
        textFile('{"a": 1}', 'b.jsonl'),
      ])
    ).toEqual({ error: 'Only Parquet files can be added together.', stored: [] });
  });

  it('names the Parquet part that fails inspection', async () => {
    expect(
      await inspectDatasetFiles([
        parquetFile(PARQUET.twoRows, 'a.parquet'),
        parquetFile(PARQUET.empty, 'b.parquet'),
      ])
    ).toEqual({ error: 'b.parquet: File contains no data', stored: [] });
  });
});
