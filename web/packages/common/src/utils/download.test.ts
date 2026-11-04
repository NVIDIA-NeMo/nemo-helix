// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  downloadPayload,
  hasResults,
  toCsv,
  toFileName,
  toJsonl,
} from '@nemo/common/src/utils/download';

describe('toCsv', () => {
  it('unions columns across rows in first-seen order', () => {
    expect(toCsv([{ a: 1 }, { b: 2, a: 3 }])).toBe('a,b\n1,\n3,2');
  });

  it('writes nested values as JSON and escapes quotes, commas, and formulas', () => {
    expect(toCsv([{ obj: { x: 'y' }, text: 'a,b', formula: '=SUM(A1)', empty: null }])).toBe(
      'obj,text,formula,empty\n"{""x"":""y""}","a,b",\'=SUM(A1),'
    );
  });

  it('returns an empty string for no rows', () => {
    expect(toCsv([])).toBe('');
  });

  it('leaves negative numbers alone but guards strings that look like formulas', () => {
    expect(toCsv([{ loss: -0.42, note: '-cmd' }])).toBe("loss,note\n-0.42,'-cmd");
  });

  it('writes formula-like strings unchanged when formulaSafe is off', () => {
    expect(toCsv([{ note: '=A1' }], { formulaSafe: false })).toBe('note\n=A1');
  });

  it('uses the given columns, keeping the header when there are no rows', () => {
    expect(toCsv([], { columns: ['a', 'b'] })).toBe('a,b');
    expect(toCsv([{ b: 1, c: 2 }], { columns: ['a', 'b'] })).toBe('a,b\n,1');
  });
});

describe('toFileName', () => {
  it('replaces unsafe characters', () => {
    expect(toFileName('ws/my eval:1')).toBe('ws_my_eval_1');
  });
});

describe('toJsonl', () => {
  it('writes one compact JSON record per line', () => {
    expect(toJsonl([{ a: 1, b: { c: 'x' } }, { a: 2 }])).toBe('{"a":1,"b":{"c":"x"}}\n{"a":2}');
  });

  it('writes BigInt values as strings instead of throwing', () => {
    expect(toJsonl([{ id: BigInt(7) }])).toBe('{"id":"7"}');
  });
});

describe('downloadPayload', () => {
  const empty = { json: { name: 'run', scores: [] }, jsonl: [], csv: [] };

  it('treats a payload without per-row records as having no results', () => {
    expect(hasResults(empty)).toBe(false);
    expect(hasResults({ ...empty, jsonl: [{ a: 1 }], csv: [{ a: 1 }] })).toBe(true);
  });

  it('never writes a file when there are no results, in any format', () => {
    const createObjectURL = vi.fn();
    vi.stubGlobal('URL', { ...URL, createObjectURL, revokeObjectURL: vi.fn() });
    for (const format of ['json', 'jsonl', 'csv'] as const) {
      expect(downloadPayload(empty, 'run', format)).toBe(false);
    }
    expect(createObjectURL).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });
});
