// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { sampleIndices, sampleTextLines } from '@nemo/common/src/utils/sampleTextLines';

describe('sampleTextLines', () => {
  const jsonl = ['{"a":1}', '{"b":2}', '{"c":3}', '{"d":4}'].join('\n');

  it('returns head lines up to maxRows', () => {
    expect(sampleTextLines(jsonl, 'head', 2)).toBe('{"a":1}\n{"b":2}');
  });

  it('returns tail lines up to maxRows', () => {
    expect(sampleTextLines(jsonl, 'tail', 2)).toBe('{"c":3}\n{"d":4}');
  });

  it('ignores blank lines when sampling', () => {
    const text = 'one\n\n  \ntwo\nthree';
    expect(sampleTextLines(text, 'head', 2)).toBe('one\ntwo');
  });

  it('handles CRLF line endings', () => {
    const text = 'one\r\ntwo\r\nthree';
    expect(sampleTextLines(text, 'tail', 2)).toBe('two\nthree');
  });

  it('returns empty string when there are no non-empty lines', () => {
    expect(sampleTextLines('\n  \n', 'head', 5)).toBe('');
  });

  it('caps at available rows when maxRows is larger', () => {
    expect(sampleTextLines(jsonl, 'head', 100)).toBe(jsonl);
  });

  it('returns empty string when maxRows is not positive', () => {
    expect(sampleTextLines(jsonl, 'head', 0)).toBe('');
    expect(sampleTextLines(jsonl, 'head', -5)).toBe('');
  });

  describe('random', () => {
    beforeEach(() => {
      let n = 0;
      vi.spyOn(Math, 'random').mockImplementation(() => {
        n += 1;
        return (n % 97) / 97;
      });
    });

    afterEach(() => {
      vi.restoreAllMocks();
    });

    it('returns at most maxRows distinct lines from the file', () => {
      const out = sampleTextLines(jsonl, 'random', 2);
      const lines = out.split('\n').filter(Boolean);
      expect(lines).toHaveLength(2);
      expect(new Set(lines).size).toBe(2);
      lines.forEach((line) => {
        expect(jsonl).toContain(line);
      });
    });
  });
});

describe('sampleIndices', () => {
  it('returns head indices in ascending order', () => {
    expect(sampleIndices(5, 'head', 3)).toEqual([0, 1, 2]);
  });

  it('returns tail indices in ascending order', () => {
    expect(sampleIndices(5, 'tail', 3)).toEqual([2, 3, 4]);
  });

  it('caps sampleSize at populationSize', () => {
    expect(sampleIndices(3, 'head', 10)).toEqual([0, 1, 2]);
    expect(sampleIndices(3, 'tail', 10)).toEqual([0, 1, 2]);
  });

  it('returns [] for non-positive populationSize or sampleSize', () => {
    expect(sampleIndices(0, 'head', 5)).toEqual([]);
    expect(sampleIndices(5, 'head', 0)).toEqual([]);
    expect(sampleIndices(-1, 'random', 1)).toEqual([]);
  });

  describe('random', () => {
    beforeEach(() => {
      let n = 0;
      vi.spyOn(Math, 'random').mockImplementation(() => {
        n += 1;
        return (n % 97) / 97;
      });
    });

    afterEach(() => {
      vi.restoreAllMocks();
    });

    it('returns sampleSize distinct indices in ascending order', () => {
      const out = sampleIndices(5, 'random', 3);
      expect(out).toHaveLength(3);
      expect(new Set(out).size).toBe(3);
      expect(out).toEqual([...out].sort((a, b) => a - b));
      out.forEach((idx) => {
        expect(idx).toBeGreaterThanOrEqual(0);
        expect(idx).toBeLessThan(5);
      });
    });
  });
});
